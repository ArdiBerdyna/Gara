#!/usr/bin/env python3
"""Optimized Street Cleaning solver focused on maximizing coverage."""

from __future__ import annotations

import argparse
import heapq
import math
import random
import sys
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


CAPACITY = {"S": 10, "M": 20, "L": 30}
W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
INF = 10**30
MAX_EXACT_MANDATORY_TASKS = 20
LARGE_MANDATORY_THRESHOLD = 50


@dataclass(frozen=True)
class Edge:
    idx: int
    a: int
    b: int
    direction: int
    time: int
    length: int
    category: str
    requirement: int

    @property
    def orientations(self) -> tuple[tuple[int, int], ...]:
        if self.direction == 1:
            return ((self.a, self.b),)
        return ((self.a, self.b), (self.b, self.a))


@dataclass
class Instance:
    name: str
    n: int
    m: int
    time_limit: int
    vehicle_count: int
    depot: int
    alpha: float
    coordinates: list[tuple[float, float]]
    edges: list[Edge]
    vehicles: list[str]
    raw_lines: list[str]


@dataclass
class Solution:
    routes: list[list[int]]
    cleaned: list[set[int]]
    route_times: list[int]
    score: float
    coverage: float
    efficiency: float
    waste: float


def docx_lines(path: Path) -> list[str]:
    """Extract non-empty paragraphs from a simple DOCX in document order."""
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    lines: list[str] = []
    body = root.find(f"{W_NS}body")
    if body is None:
        return lines
    for paragraph in body.iter(f"{W_NS}p"):
        text = "".join(node.text or "" for node in paragraph.iter(f"{W_NS}t"))
        if text.strip():
            lines.append(text.strip())
    return lines


def parse_lines(lines: list[str], name: str) -> Instance:
    lines = [line.strip() for line in lines if line.strip()]
    if not lines:
        raise ValueError(f"{name}: empty instance")
    header = lines[0].split()
    if len(header) != 6:
        raise ValueError(f"{name}: expected 6 header fields, got {len(header)}")
    n, m, time_limit, vehicle_count, depot = map(int, header[:5])
    alpha = float(header[5])

    if len(lines) == m + 2:
        coordinate_lines: list[str] = []
        edge_start = 1
    elif len(lines) >= n + m + 2:
        coordinate_lines = lines[1 : 1 + n]
        edge_start = 1 + n
    else:
        raise ValueError(
            f"{name}: expected {m + 2} old-format lines or at least "
            f"{n + m + 2} current-format lines, got {len(lines)}"
        )

    coordinates = [tuple(map(float, line.split())) for line in coordinate_lines]
    edges: list[Edge] = []
    for idx, line in enumerate(lines[edge_start : edge_start + m]):
        fields = line.split()
        if len(fields) != 7:
            raise ValueError(f"{name}: edge {idx} has {len(fields)} fields")
        a, b, direction, travel_time, length = map(int, fields[:5])
        category = fields[5]
        requirement = int(fields[6])
        edges.append(
            Edge(idx, a, b, direction, travel_time, length, category, requirement)
        )
    vehicles = lines[edge_start + m].split()
    if len(vehicles) != vehicle_count:
        raise ValueError(
            f"{name}: expected {vehicle_count} vehicles, got {len(vehicles)}"
        )
    return Instance(
        name,
        n,
        m,
        time_limit,
        vehicle_count,
        depot,
        alpha,
        coordinates,
        edges,
        vehicles,
        lines,
    )


def load_instance(path: Path) -> Instance:
    if path.suffix.lower() == ".docx":
        lines = docx_lines(path)
    else:
        lines = path.read_text(encoding="utf-8").splitlines()
    return parse_lines(lines, path.stem)


class Solver:
    def __init__(self, instance: Instance, seed: int = 20260620):
        self.ins = instance
        self.rng = random.Random(seed)
        self.adj: list[list[tuple[int, int, int]]] = [
            [] for _ in range(instance.n)
        ]
        self.pair_to_edge: dict[tuple[int, int], int] = {}
        self.reverse_pair_to_edge: dict[tuple[int, int], int] = {}
        for edge in instance.edges:
            self.adj[edge.a].append((edge.b, edge.time, edge.idx))
            self.pair_to_edge[(edge.a, edge.b)] = edge.idx
            if edge.direction == 2:
                self.adj[edge.b].append((edge.a, edge.time, edge.idx))
                self.pair_to_edge[(edge.b, edge.a)] = edge.idx
        self.dist: list[list[int]] = []
        self.prev_node: list[list[int]] = []
        self.prev_edge: list[list[int]] = []
        self._all_pairs_shortest_paths()
        self.mandatory_count = sum(
            1 for edge in instance.edges if edge.category == "M"
        )
        self.is_large = self.mandatory_count > LARGE_MANDATORY_THRESHOLD
        self.lmax = sum(
            edge.length for edge in instance.edges if edge.category in {"M", "O"}
        )
        self.wmax = sum(
            (30 - edge.requirement) * edge.length / 1000
            for edge in instance.edges
            if edge.category in {"M", "O"}
        )
        # Pre-compute optional edge gains for faster evaluation
        self.optional_gain_cache: dict[int, dict[int, float]] = {}
        for edge in instance.edges:
            if edge.category == "O":
                self.optional_gain_cache[edge.idx] = {}
                for vidx, vtype in enumerate(instance.vehicles):
                    cap = CAPACITY[vtype]
                    if cap >= edge.requirement:
                        waste = (cap - edge.requirement) * edge.length / 1000
                        gain = instance.alpha * edge.length / max(1, self.lmax)
                        if self.wmax > 0:
                            gain -= (1 - instance.alpha) * waste / self.wmax
                        self.optional_gain_cache[edge.idx][vidx] = gain

    def _all_pairs_shortest_paths(self) -> None:
        for source in range(self.ins.n):
            dist = [INF] * self.ins.n
            prev_node = [-1] * self.ins.n
            prev_edge = [-1] * self.ins.n
            dist[source] = 0
            queue = [(0, source)]
            while queue:
                current_distance, node = heapq.heappop(queue)
                if current_distance != dist[node]:
                    continue
                for nxt, cost, edge_idx in self.adj[node]:
                    candidate = current_distance + cost
                    if candidate < dist[nxt]:
                        dist[nxt] = candidate
                        prev_node[nxt] = node
                        prev_edge[nxt] = edge_idx
                        heapq.heappush(queue, (candidate, nxt))
            self.dist.append(dist)
            self.prev_node.append(prev_node)
            self.prev_edge.append(prev_edge)

    def shortest_path(self, source: int, target: int) -> tuple[list[int], list[int]]:
        if self.dist[source][target] >= INF:
            raise ValueError(f"No directed path from {source} to {target}")
        if source == target:
            return [source], []
        nodes = [target]
        edges: list[int] = []
        current = target
        while current != source:
            edge_idx = self.prev_edge[source][current]
            previous = self.prev_node[source][current]
            if edge_idx < 0 or previous < 0:
                raise ValueError(f"Broken shortest path from {source} to {target}")
            edges.append(edge_idx)
            nodes.append(previous)
            current = previous
        nodes.reverse()
        edges.reverse()
        return nodes, edges

    @lru_cache(maxsize=None)
    def sequence_plan(
        self, sequence: tuple[int, ...]
    ) -> tuple[int, tuple[tuple[int, int], ...]]:
        """Minimum-time closed depot tour for this ordered service-edge sequence."""
        if not sequence:
            return 0, ()
        orientation_options = [self.ins.edges[idx].orientations for idx in sequence]
        previous_costs: list[int] = []
        parents: list[list[int]] = []
        first_edge = self.ins.edges[sequence[0]]
        for start, _end in orientation_options[0]:
            previous_costs.append(self.dist[self.ins.depot][start] + first_edge.time)
        parents.append([-1] * len(orientation_options[0]))

        for position in range(1, len(sequence)):
            edge = self.ins.edges[sequence[position]]
            current_costs = [INF] * len(orientation_options[position])
            current_parents = [-1] * len(orientation_options[position])
            for current_orientation, (start, _end) in enumerate(
                orientation_options[position]
            ):
                for previous_orientation, (_prev_start, prev_end) in enumerate(
                    orientation_options[position - 1]
                ):
                    candidate = (
                        previous_costs[previous_orientation]
                        + self.dist[prev_end][start]
                        + edge.time
                    )
                    if candidate < current_costs[current_orientation]:
                        current_costs[current_orientation] = candidate
                        current_parents[current_orientation] = previous_orientation
            previous_costs = current_costs
            parents.append(current_parents)

        best_cost = INF
        best_last = -1
        for orientation, (_start, end) in enumerate(orientation_options[-1]):
            candidate = previous_costs[orientation] + self.dist[end][self.ins.depot]
            if candidate < best_cost:
                best_cost = candidate
                best_last = orientation
        if best_last < 0 or best_cost >= INF:
            return INF, ()

        choices = [0] * len(sequence)
        choices[-1] = best_last
        for position in range(len(sequence) - 1, 0, -1):
            choices[position - 1] = parents[position][choices[position]]
        plan = tuple(
            orientation_options[position][choice]
            for position, choice in enumerate(choices)
        )
        return best_cost, plan

    def route_cost(self, route: list[int]) -> int:
        cost, _ = self.sequence_plan(tuple(route))
        return cost

    def optional_gain(self, edge_idx: int, vehicle_idx: int) -> float:
        return self.optional_gain_cache.get(edge_idx, {}).get(vehicle_idx, -INF)

    def vehicle_tiers(self, requirement: int) -> list[list[int]]:
        if requirement == 30:
            tiers = [[idx for idx, vehicle in enumerate(self.ins.vehicles) if CAPACITY[vehicle] == 30]]
        elif requirement == 20:
            tiers = [
                [idx for idx, vehicle in enumerate(self.ins.vehicles) if CAPACITY[vehicle] == 20],
                [idx for idx, vehicle in enumerate(self.ins.vehicles) if CAPACITY[vehicle] == 30],
            ]
        else:
            tiers = [
                [idx for idx, vehicle in enumerate(self.ins.vehicles) if CAPACITY[vehicle] == 10],
                [idx for idx, vehicle in enumerate(self.ins.vehicles) if CAPACITY[vehicle] == 20],
                [idx for idx, vehicle in enumerate(self.ins.vehicles) if CAPACITY[vehicle] == 30],
            ]
        return [tier for tier in tiers if tier]

    def best_insertions(
        self,
        routes: list[list[int]],
        edge_idx: int,
        mandatory: bool,
        vehicle_filter: list[int] | None = None,
    ) -> list[tuple[tuple[float, ...], int, int, int]]:
        edge = self.ins.edges[edge_idx]
        candidates: list[tuple[tuple[float, ...], int, int, int]] = []
        vehicle_indices = (
            vehicle_filter
            if vehicle_filter is not None
            else list(range(len(self.ins.vehicles)))
        )
        for vehicle_idx in vehicle_indices:
            vehicle_type = self.ins.vehicles[vehicle_idx]
            capacity = CAPACITY[vehicle_type]
            if capacity < edge.requirement:
                continue
            old_cost = self.route_cost(routes[vehicle_idx])
            for position in range(len(routes[vehicle_idx]) + 1):
                new_sequence = (
                    routes[vehicle_idx][:position]
                    + [edge_idx]
                    + routes[vehicle_idx][position:]
                )
                new_cost, _ = self.sequence_plan(tuple(new_sequence))
                if new_cost > self.ins.time_limit:
                    continue
                delta = new_cost - old_cost
                excess = capacity - edge.requirement
                if mandatory:
                    mismatch = 0 if self.ins.alpha >= 0.999999 else excess
                    key = (
                        float(mismatch),
                        new_cost / self.ins.time_limit,
                        float(delta),
                        self.rng.random() * 1e-6,
                    )
                else:
                    gain = self.optional_gain(edge_idx, vehicle_idx)
                    if gain <= 1e-12:
                        continue
                    # Modified key: prioritize coverage more aggressively
                    if delta <= 0:
                        # Free insertion - always take it with high priority
                        key = (-1e9 - gain, -gain, 0.0, float(excess))
                    else:
                        # For costly insertions, weigh coverage more heavily
                        # Use a higher weight for length contribution
                        length_ratio = edge.length / max(1, self.lmax)
                        coverage_contribution = self.ins.alpha * length_ratio
                        efficiency_penalty = (1 - self.ins.alpha) * (excess * edge.length / 1000) / max(1, self.wmax)
                        net_gain = coverage_contribution - efficiency_penalty
                        if net_gain <= 0:
                            continue
                        # Scale by delta to prioritize efficient insertions
                        ratio = net_gain / max(1, delta)
                        key = (-ratio, -net_gain, float(delta), float(excess))
                candidates.append((key, vehicle_idx, position, new_cost))
        candidates.sort(key=lambda item: item[0])
        return candidates

    def subset_route_table(
        self, tasks: list[int]
    ) -> tuple[list[int], list[list[int]]]:
        """Exact Held-Karp route cost and service order for every task subset."""
        count = len(tasks)
        size = 1 << count
        state_cost: dict[tuple[int, int, int], int] = {}
        parent: dict[
            tuple[int, int, int], tuple[int, int, int] | None
        ] = {}

        for task_pos, edge_idx in enumerate(tasks):
            edge = self.ins.edges[edge_idx]
            mask = 1 << task_pos
            for orientation_idx, (start, _end) in enumerate(edge.orientations):
                key = (mask, task_pos, orientation_idx)
                state_cost[key] = self.dist[self.ins.depot][start] + edge.time
                parent[key] = None

        for mask in range(1, size):
            states = [
                (last, orientation_idx, cost)
                for (state_mask, last, orientation_idx), cost in state_cost.items()
                if state_mask == mask
            ]
            for last, orientation_idx, cost in states:
                previous_edge = self.ins.edges[tasks[last]]
                previous_end = previous_edge.orientations[orientation_idx][1]
                for nxt in range(count):
                    bit = 1 << nxt
                    if mask & bit:
                        continue
                    next_edge = self.ins.edges[tasks[nxt]]
                    next_mask = mask | bit
                    for next_orientation_idx, (start, _end) in enumerate(
                        next_edge.orientations
                    ):
                        candidate = (
                            cost + self.dist[previous_end][start] + next_edge.time
                        )
                        next_key = (next_mask, nxt, next_orientation_idx)
                        if candidate < state_cost.get(next_key, INF):
                            state_cost[next_key] = candidate
                            parent[next_key] = (mask, last, orientation_idx)

        subset_cost = [INF] * size
        subset_order: list[list[int]] = [[] for _ in range(size)]
        subset_cost[0] = 0
        for mask in range(1, size):
            best_key = None
            best_cost = INF
            for (state_mask, last, orientation_idx), cost in state_cost.items():
                if state_mask != mask:
                    continue
                end = self.ins.edges[tasks[last]].orientations[orientation_idx][1]
                candidate = cost + self.dist[end][self.ins.depot]
                if candidate < best_cost:
                    best_cost = candidate
                    best_key = (state_mask, last, orientation_idx)
            subset_cost[mask] = best_cost
            if best_key is not None:
                reverse_order: list[int] = []
                key: tuple[int, int, int] | None = best_key
                while key is not None:
                    reverse_order.append(tasks[key[1]])
                    key = parent[key]
                subset_order[mask] = list(reversed(reverse_order))
        return subset_cost, subset_order

    def partition_exact_type(
        self, tasks: list[int], vehicle_indices: list[int]
    ) -> tuple[list[list[int]], int] | None:
        if not tasks:
            return ([[] for _ in vehicle_indices], 0)
        if not vehicle_indices:
            return None
        subset_cost, subset_order = self.subset_route_table(tasks)
        full_mask = (1 << len(tasks)) - 1

        @lru_cache(maxsize=None)
        def assign(
            vehicle_pos: int, remaining: int
        ) -> tuple[int, int, tuple[int, ...]] | None:
            if vehicle_pos == len(vehicle_indices):
                return (0, 0, ()) if remaining == 0 else None
            best = None
            subset = remaining
            while True:
                cost = subset_cost[subset]
                if cost <= self.ins.time_limit:
                    tail = assign(vehicle_pos + 1, remaining ^ subset)
                    if tail is not None:
                        maximum = max(cost, tail[0])
                        total = cost + tail[1]
                        candidate = (maximum, total, (subset,) + tail[2])
                        if best is None or candidate[:2] < best[:2]:
                            best = candidate
                if subset == 0:
                    break
                subset = (subset - 1) & remaining
            return best

        assignment = assign(0, full_mask)
        if assignment is None:
            return None
        sequences = [subset_order[mask] for mask in assignment[2]]
        return sequences, assignment[1]

    def construct_mandatory_exact(self) -> list[list[int]] | None:
        """Build zero-waste mandatory routes by exact requirement class."""
        routes: list[list[int]] = [[] for _ in self.ins.vehicles]
        for requirement in (30, 20, 10):
            tasks = [
                edge.idx
                for edge in self.ins.edges
                if edge.category == "M" and edge.requirement == requirement
            ]
            if len(tasks) > MAX_EXACT_MANDATORY_TASKS:
                return None
            vehicles = [
                idx
                for idx, vehicle_type in enumerate(self.ins.vehicles)
                if CAPACITY[vehicle_type] == requirement
            ]
            partition = self.partition_exact_type(tasks, vehicles)
            if partition is None:
                return None
            sequences, _total = partition
            for vehicle_idx, sequence in zip(vehicles, sequences):
                routes[vehicle_idx] = sequence
        return routes

    def cover_state_walks(
        self, tasks: list[int]
    ) -> tuple[list[int], list[int]]:
        """Shortest depot walks for every exactly-covered subset of task edges."""
        if len(tasks) > MAX_EXACT_MANDATORY_TASKS:
            raise ValueError("too many tasks for exact cover-state DP")
        task_bit = {edge_idx: 1 << pos for pos, edge_idx in enumerate(tasks)}
        mask_count = 1 << len(tasks)
        if self.ins.n > sys.maxsize // mask_count:
            raise ValueError("cover-state DP state space too large")
        state_count = self.ins.n * mask_count
        distance = [INF] * state_count
        previous = [-1] * state_count
        start_state = self.ins.depot
        distance[start_state] = 0
        queue = [(0, start_state)]

        while queue:
            current_distance, state = heapq.heappop(queue)
            if current_distance != distance[state]:
                continue
            node = state % self.ins.n
            mask = state // self.ins.n
            for nxt, edge_time, edge_idx in self.adj[node]:
                next_mask = mask | task_bit.get(edge_idx, 0)
                next_state = next_mask * self.ins.n + nxt
                candidate = current_distance + edge_time
                if candidate < distance[next_state]:
                    distance[next_state] = candidate
                    previous[next_state] = state
                    heapq.heappush(queue, (candidate, next_state))

        depot_cost = [
            distance[mask * self.ins.n + self.ins.depot]
            for mask in range(mask_count)
        ]
        return depot_cost, previous

    def assign_cover_masks(
        self, costs: list[int], vehicle_count: int
    ) -> tuple[int, ...] | None:
        full_mask = len(costs) - 1
        candidates = [
            mask for mask, cost in enumerate(costs) if cost <= self.ins.time_limit
        ]

        @lru_cache(maxsize=None)
        def assign(
            vehicle_pos: int, covered: int
        ) -> tuple[int, int, int, tuple[int, ...]] | None:
            if vehicle_pos == vehicle_count:
                return (0, 0, 0, ()) if covered == full_mask else None
            best = None
            for mask in candidates:
                new_covered = covered | mask
                tail = assign(vehicle_pos + 1, new_covered)
                if tail is None:
                    continue
                overlap = (covered & mask).bit_count() + tail[0]
                maximum = max(costs[mask], tail[1])
                total = costs[mask] + tail[2]
                candidate = (overlap, maximum, total, (mask,) + tail[3])
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
            return best

        result = assign(0, 0)
        return None if result is None else result[3]

    def reconstruct_state_walk(self, mask: int, previous: list[int]) -> list[int]:
        if mask == 0:
            return [self.ins.depot]
        state = mask * self.ins.n + self.ins.depot
        reverse_nodes = [self.ins.depot]
        while previous[state] >= 0:
            state = previous[state]
            reverse_nodes.append(state % self.ins.n)
        reverse_nodes.reverse()
        if not reverse_nodes or reverse_nodes[0] != self.ins.depot:
            raise ValueError("Broken cover-state predecessor chain")
        return reverse_nodes

    def construct_mandatory_product(
        self,
    ) -> tuple[list[list[int]], list[set[int]]] | None:
        """Exact mandatory coverage using shortest paths in (node, covered-set)."""
        routes = [[self.ins.depot] for _ in self.ins.vehicles]
        cleaned = [set() for _ in self.ins.vehicles]
        for requirement in (30, 20, 10):
            tasks = [
                edge.idx
                for edge in self.ins.edges
                if edge.category == "M" and edge.requirement == requirement
            ]
            if not tasks:
                continue
            if len(tasks) > MAX_EXACT_MANDATORY_TASKS:
                return None
            vehicles = [
                idx
                for idx, vehicle_type in enumerate(self.ins.vehicles)
                if CAPACITY[vehicle_type] == requirement
            ]
            if not vehicles:
                return None
            costs, previous = self.cover_state_walks(tasks)
            masks = self.assign_cover_masks(costs, len(vehicles))
            if masks is None:
                return None
            claimed: set[int] = set()
            for vehicle_idx, mask in zip(vehicles, masks):
                routes[vehicle_idx] = self.reconstruct_state_walk(mask, previous)
                for task_pos, edge_idx in enumerate(tasks):
                    if mask & (1 << task_pos) and edge_idx not in claimed:
                        cleaned[vehicle_idx].add(edge_idx)
                        claimed.add(edge_idx)
            if len(claimed) != len(tasks):
                return None
        return routes, cleaned

    def add_opportunistic_nodes(
        self, routes: list[list[int]], cleaned: list[set[int]], already: set[int]
    ) -> None:
        traversed = []
        for nodes in routes:
            edge_set = set()
            for a, b in zip(nodes, nodes[1:]):
                if (a, b) in self.pair_to_edge:
                    edge_set.add(self.pair_to_edge[(a, b)])
            traversed.append(edge_set)
        for edge in self.ins.edges:
            if edge.category != "O" or edge.idx in already:
                continue
            best_vehicle = None
            best_gain = -INF
            for vehicle_idx, edge_ids in enumerate(traversed):
                if edge.idx not in edge_ids:
                    continue
                gain = self.optional_gain(edge.idx, vehicle_idx)
                if gain > best_gain and gain > 1e-12:
                    best_gain = gain
                    best_vehicle = vehicle_idx
            if best_vehicle is not None:
                cleaned[best_vehicle].add(edge.idx)
                already.add(edge.idx)

    def evaluate_node_routes(
        self, routes: list[list[int]], cleaned: list[set[int]]
    ) -> Solution:
        route_times = []
        for nodes in routes:
            route_times.append(
                sum(
                    self.ins.edges[self.pair_to_edge[(a, b)]].time
                    for a, b in zip(nodes, nodes[1:])
                    if (a, b) in self.pair_to_edge
                )
            )
        unique_cleaned = set().union(*cleaned) if cleaned else set()
        waste = 0.0
        for vehicle_idx, edge_ids in enumerate(cleaned):
            capacity = CAPACITY[self.ins.vehicles[vehicle_idx]]
            waste += sum(
                (capacity - self.ins.edges[edge_idx].requirement)
                * self.ins.edges[edge_idx].length
                / 1000
                for edge_idx in edge_ids
            )
        cleaned_length = sum(self.ins.edges[idx].length for idx in unique_cleaned)
        coverage = cleaned_length / self.lmax if self.lmax else 1.0
        efficiency = 1 - waste / self.wmax if self.wmax else 1.0
        score = self.ins.alpha * coverage + (1 - self.ins.alpha) * efficiency
        return Solution(
            routes, cleaned, route_times, score, coverage, efficiency, waste
        )

    def construct_mandatory(self, restart: int) -> list[list[int]] | None:
        routes: list[list[int]] = [[] for _ in self.ins.vehicles]
        tasks = [edge.idx for edge in self.ins.edges if edge.category == "M"]
        difficulty: dict[int, float] = {}
        for edge_idx in tasks:
            edge = self.ins.edges[edge_idx]
            individual_cost, _ = self.sequence_plan((edge_idx,))
            compatible = sum(
                CAPACITY[vehicle] >= edge.requirement for vehicle in self.ins.vehicles
            )
            difficulty[edge_idx] = (
                edge.requirement * 1_000_000
                + individual_cost * 10
                + edge.length
                - compatible * 100
            )
        if restart == 0:
            tasks.sort(key=lambda idx: difficulty[idx], reverse=True)
        else:
            tasks.sort(
                key=lambda idx: difficulty[idx]
                + self.rng.uniform(-8_000_000, 8_000_000),
                reverse=True,
            )

        for edge_idx in tasks:
            candidates = self.best_insertions(routes, edge_idx, mandatory=True)
            if not candidates:
                return None
            chosen_index = 0
            if restart and len(candidates) > 1 and self.rng.random() < 0.30:
                best_mismatch = candidates[0][0][0]
                near = [
                    i
                    for i, item in enumerate(candidates[: min(5, len(candidates))])
                    if item[0][0] == best_mismatch
                ]
                chosen_index = self.rng.choice(near)
            _, vehicle_idx, position, _ = candidates[chosen_index]
            routes[vehicle_idx].insert(position, edge_idx)
        return routes

    def construct_mandatory_by_class(self, restart: int) -> list[list[int]] | None:
        """Assign mandatory edges by requirement class to the tightest vehicle tier."""
        routes: list[list[int]] = [[] for _ in self.ins.vehicles]
        for requirement in (30, 20, 10):
            tasks = [
                edge.idx
                for edge in self.ins.edges
                if edge.category == "M" and edge.requirement == requirement
            ]
            if not tasks:
                continue
            difficulty: dict[int, float] = {}
            for edge_idx in tasks:
                edge = self.ins.edges[edge_idx]
                individual_cost, _ = self.sequence_plan((edge_idx,))
                difficulty[edge_idx] = (
                    individual_cost * 10 + edge.length + edge.time
                )
            if restart == 0:
                tasks.sort(key=lambda idx: difficulty[idx], reverse=True)
            else:
                tasks.sort(
                    key=lambda idx: difficulty[idx]
                    + self.rng.uniform(-5_000, 5_000),
                    reverse=True,
                )
            tiers = self.vehicle_tiers(requirement)
            for edge_idx in tasks:
                candidates: list[tuple[tuple[float, ...], int, int, int]] = []
                for tier in tiers:
                    tier_candidates = self.best_insertions(
                        routes, edge_idx, mandatory=True, vehicle_filter=tier
                    )
                    if tier_candidates:
                        candidates = tier_candidates
                        break
                if not candidates:
                    candidates = self.best_insertions(
                        routes, edge_idx, mandatory=True
                    )
                if not candidates:
                    return None
                chosen_index = 0
                if restart and len(candidates) > 1 and self.rng.random() < 0.25:
                    best_mismatch = candidates[0][0][0]
                    near = [
                        i
                        for i, item in enumerate(candidates[: min(5, len(candidates))])
                        if item[0][0] == best_mismatch
                    ]
                    chosen_index = self.rng.choice(near)
                _, vehicle_idx, position, _ = candidates[chosen_index]
                routes[vehicle_idx].insert(position, edge_idx)
        return routes

    def estimate_optional_priority(self, edge_idx: int) -> float:
        edge = self.ins.edges[edge_idx]
        best_gain = -INF
        for vehicle_idx, vehicle_type in enumerate(self.ins.vehicles):
            gain = self.optional_gain(edge_idx, vehicle_idx)
            if gain <= best_gain:
                continue
            route = [edge_idx]
            cost, _ = self.sequence_plan(tuple(route))
            if cost > self.ins.time_limit:
                continue
            ratio = gain / max(1, cost)
            if ratio > best_gain:
                best_gain = ratio
        return best_gain

    def refine_routes(
        self, routes: list[list[int]], cleaned: list[set[int]], rounds: int = 3
    ) -> tuple[list[list[int]], list[set[int]]]:
        """Improve vehicle matching with cheap relocations and optional pruning."""
        for _ in range(rounds):
            changed = False
            for vehicle_idx, edge_ids in enumerate(cleaned):
                for edge_idx in list(edge_ids):
                    edge = self.ins.edges[edge_idx]
                    current_capacity = CAPACITY[self.ins.vehicles[vehicle_idx]]
                    current_excess = current_capacity - edge.requirement
                    if edge.category == "O":
                        # Only prune if it has negative value
                        gain = self.optional_gain(edge_idx, vehicle_idx)
                        if gain <= -1e-9:
                            if edge_idx in routes[vehicle_idx]:
                                routes[vehicle_idx].remove(edge_idx)
                            cleaned[vehicle_idx].remove(edge_idx)
                            changed = True
                        continue
                    if current_excess <= 0:
                        continue
                    best_move: tuple[int, int, tuple[float, ...]] | None = None
                    for target_idx, vehicle_type in enumerate(self.ins.vehicles):
                        if target_idx == vehicle_idx:
                            continue
                        target_capacity = CAPACITY[vehicle_type]
                        if target_capacity < edge.requirement:
                            continue
                        target_excess = target_capacity - edge.requirement
                        if target_excess >= current_excess:
                            continue
                        trial_routes = [route[:] for route in routes]
                        if edge_idx not in trial_routes[vehicle_idx]:
                            continue
                        trial_routes[vehicle_idx].remove(edge_idx)
                        candidates = self.best_insertions(
                            trial_routes, edge_idx, mandatory=edge.category == "M"
                        )
                        for key, new_vehicle, position, _new_cost in candidates:
                            if new_vehicle != target_idx:
                                continue
                            if best_move is None or key < best_move[2]:
                                best_move = (new_vehicle, position, key)
                            break
                    if best_move is None:
                        continue
                    target_idx, position, _key = best_move
                    if edge_idx in routes[vehicle_idx]:
                        routes[vehicle_idx].remove(edge_idx)
                    cleaned[vehicle_idx].remove(edge_idx)
                    routes[target_idx].insert(position, edge_idx)
                    cleaned[target_idx].add(edge_idx)
                    changed = True
            if not changed:
                break
        return routes, cleaned

    def realize_route(
        self, sequence: list[int]
    ) -> tuple[list[int], list[int], int]:
        total_cost, orientation_plan = self.sequence_plan(tuple(sequence))
        if total_cost >= INF:
            raise ValueError("Cannot realize unreachable service sequence")
        route_nodes = [self.ins.depot]
        traversed_edges: list[int] = []
        current = self.ins.depot
        for edge_idx, (start, end) in zip(sequence, orientation_plan):
            path_nodes, path_edges = self.shortest_path(current, start)
            route_nodes.extend(path_nodes[1:])
            traversed_edges.extend(path_edges)
            route_nodes.append(end)
            traversed_edges.append(edge_idx)
            current = end
        path_nodes, path_edges = self.shortest_path(current, self.ins.depot)
        route_nodes.extend(path_nodes[1:])
        traversed_edges.extend(path_edges)
        return route_nodes, traversed_edges, total_cost

    def optimize_sequence(self, sequence: list[int], max_passes: int = 6) -> list[int]:
        """Reduce a vehicle's closed-tour time with or-opt relocation moves."""
        if len(sequence) <= 2:
            return list(sequence)
        best = list(sequence)
        best_cost, _ = self.sequence_plan(tuple(best))
        if best_cost >= INF:
            return best
        for _ in range(max_passes):
            improved = False
            for segment in (1, 2, 3):
                position = 0
                while position + segment <= len(best):
                    chunk = best[position : position + segment]
                    rest = best[:position] + best[position + segment :]
                    local_cost = best_cost
                    local_seq: list[int] | None = None
                    for insert_at in range(len(rest) + 1):
                        if insert_at == position:
                            continue
                        trial = rest[:insert_at] + chunk + rest[insert_at:]
                        cost, _ = self.sequence_plan(tuple(trial))
                        if cost < local_cost - 1e-9:
                            local_cost = cost
                            local_seq = trial
                    if local_seq is not None:
                        best = local_seq
                        best_cost = local_cost
                        improved = True
                    else:
                        position += 1
            if not improved:
                break
        return best

    def fill_optionals_greedy(
        self, mandatory_routes: list[list[int]]
    ) -> tuple[list[list[int]], list[set[int]]]:
        """Aggressively insert optional edges using a greedy approach."""
        routes = [route[:] for route in mandatory_routes]
        cleaned = [set(route) for route in routes]
        already = set().union(*cleaned) if cleaned else set()
        
        # Sort optional edges by length (longer first) to maximize coverage
        optional_edges = sorted(
            [edge for edge in self.ins.edges if edge.category == "O" and edge.idx not in already],
            key=lambda e: (e.length, e.requirement),
            reverse=True
        )
        
        # First pass: try to insert all optional edges
        for edge in optional_edges:
            edge_idx = edge.idx
            # Try vehicles with sufficient capacity, preferring those with less excess
            candidates = self.best_insertions(routes, edge_idx, mandatory=False)
            if candidates:
                _key, vehicle_idx, position, _ = candidates[0]
                routes[vehicle_idx].insert(position, edge_idx)
                cleaned[vehicle_idx].add(edge_idx)
                already.add(edge_idx)
        
        # Second pass: optimize and try to insert remaining edges
        for _ in range(3):
            # Compact routes to free up time
            routes = [self.optimize_sequence(r, max_passes=4) for r in routes]
            
            # Try to add any remaining optional edges
            remaining = [e for e in self.ins.edges if e.category == "O" and e.idx not in already]
            for edge in remaining:
                edge_idx = edge.idx
                candidates = self.best_insertions(routes, edge_idx, mandatory=False)
                if candidates:
                    _key, vehicle_idx, position, _ = candidates[0]
                    routes[vehicle_idx].insert(position, edge_idx)
                    cleaned[vehicle_idx].add(edge_idx)
                    already.add(edge_idx)
        
        # Mark no-time-cost optional streets
        self.add_opportunistic_nodes(routes, cleaned, already)
        return routes, cleaned

    def fill_optionals(
        self, mandatory_routes: list[list[int]], fast: bool = False
    ) -> tuple[list[list[int]], list[set[int]]]:
        return self.fill_optionals_greedy(mandatory_routes)

    def evaluate(
        self, routes: list[list[int]], cleaned: list[set[int]]
    ) -> Solution:
        route_nodes: list[list[int]] = []
        route_times: list[int] = []
        for sequence in routes:
            nodes, _traversed, time = self.realize_route(sequence)
            route_nodes.append(nodes)
            route_times.append(time)

        unique_cleaned: set[int] = set()
        waste = 0.0
        for vehicle_idx, edge_ids in enumerate(cleaned):
            capacity = CAPACITY[self.ins.vehicles[vehicle_idx]]
            for edge_idx in edge_ids:
                edge = self.ins.edges[edge_idx]
                unique_cleaned.add(edge_idx)
                waste += (capacity - edge.requirement) * edge.length / 1000
        total_length = sum(self.ins.edges[idx].length for idx in unique_cleaned)
        coverage = total_length / self.lmax if self.lmax else 1.0
        efficiency = 1 - waste / self.wmax if self.wmax else 1.0
        score = self.ins.alpha * coverage + (1 - self.ins.alpha) * efficiency
        return Solution(
            route_nodes, cleaned, route_times, score, coverage, efficiency, waste
        )

    def solve(self, restarts: int = 250) -> Solution:
        best: Solution | None = None
        valid_mandatory_constructions = 0

        # Try exact mandatory construction first
        product_routes = self.construct_mandatory_product()
        if product_routes is not None:
            node_routes, product_cleaned = product_routes
            already = set().union(*product_cleaned) if product_cleaned else set()
            self.add_opportunistic_nodes(node_routes, product_cleaned, already)
            best = self.evaluate_node_routes(node_routes, product_cleaned)
            valid_mandatory_constructions += 1

        # Try exact route construction
        exact_routes = self.construct_mandatory_exact()
        if exact_routes is not None:
            valid_mandatory_constructions += 1
            routes, cleaned = self.fill_optionals(exact_routes, fast=False)
            routes, cleaned = self.refine_routes(routes, cleaned, rounds=3)
            solution = self.evaluate(routes, cleaned)
            if best is None or solution.score > best.score + 1e-12:
                best = solution

        # Try heuristic construction with multiple restarts
        effective_restarts = min(restarts, 12) if self.is_large else restarts
        
        for restart in range(effective_restarts):
            # Try both construction methods
            for method in [self.construct_mandatory, self.construct_mandatory_by_class]:
                routes = method(restart)
                if routes is None:
                    continue
                valid_mandatory_constructions += 1
                routes, cleaned = self.fill_optionals(routes, fast=False)
                routes, cleaned = self.refine_routes(routes, cleaned, rounds=3)
                solution = self.evaluate(routes, cleaned)
                if best is None or solution.score > best.score + 1e-12:
                    best = solution

        if best is None:
            raise RuntimeError(
                f"No feasible mandatory assignment found in {effective_restarts} restarts"
            )
        if valid_mandatory_constructions == 0:
            raise RuntimeError("Internal error: no mandatory construction")
        return best


def validate_solution(instance: Instance, solution: Solution) -> dict[str, float | int]:
    if len(solution.routes) != instance.vehicle_count:
        raise ValueError("Wrong vehicle count in solution")
    pair_to_edge: dict[tuple[int, int], int] = {}
    for edge in instance.edges:
        pair_to_edge[(edge.a, edge.b)] = edge.idx
        if edge.direction == 2:
            pair_to_edge[(edge.b, edge.a)] = edge.idx

    mandatory = {edge.idx for edge in instance.edges if edge.category == "M"}
    cleaned_once: set[int] = set()
    waste = 0.0
    total_route_time = 0
    for vehicle_idx, nodes in enumerate(solution.routes):
        if not nodes or nodes[0] != instance.depot or nodes[-1] != instance.depot:
            raise ValueError(f"Vehicle {vehicle_idx + 1} does not start/end at depot")
        traversed: list[int] = []
        route_time = 0
        for a, b in zip(nodes, nodes[1:]):
            if (a, b) not in pair_to_edge:
                raise ValueError(f"Vehicle {vehicle_idx + 1}: invalid move {a}->{b}")
            edge_idx = pair_to_edge[(a, b)]
            traversed.append(edge_idx)
            route_time += instance.edges[edge_idx].time
        if route_time > instance.time_limit:
            raise ValueError(
                f"Vehicle {vehicle_idx + 1}: {route_time} exceeds {instance.time_limit}"
            )
        total_route_time += route_time
        capacity = CAPACITY[instance.vehicles[vehicle_idx]]
        for edge_idx in solution.cleaned[vehicle_idx]:
            edge = instance.edges[edge_idx]
            if edge_idx not in traversed:
                raise ValueError(
                    f"Vehicle {vehicle_idx + 1}: cleaned edge {edge_idx} not traversed"
                )
            if edge.category == "C":
                raise ValueError(f"Vehicle {vehicle_idx + 1}: cleans connector {edge_idx}")
            if capacity < edge.requirement:
                raise ValueError(
                    f"Vehicle {vehicle_idx + 1}: capacity {capacity} below edge "
                    f"{edge_idx} requirement {edge.requirement}"
                )
            if edge_idx in cleaned_once:
                raise ValueError(f"Edge {edge_idx} cleaned more than once")
            cleaned_once.add(edge_idx)
            waste += (capacity - edge.requirement) * edge.length / 1000
    missing = mandatory - cleaned_once
    if missing:
        raise ValueError(f"Missing mandatory edges: {sorted(missing)}")
    lmax = sum(
        edge.length for edge in instance.edges if edge.category in {"M", "O"}
    )
    wmax = sum(
        (30 - edge.requirement) * edge.length / 1000
        for edge in instance.edges
        if edge.category in {"M", "O"}
    )
    cleaned_length = sum(instance.edges[idx].length for idx in cleaned_once)
    coverage = cleaned_length / lmax if lmax else 1.0
    efficiency = 1 - waste / wmax if wmax else 1.0
    score = instance.alpha * coverage + (1 - instance.alpha) * efficiency
    return {
        "cleaned_edges": len(cleaned_once),
        "mandatory_edges": len(mandatory),
        "cleaned_length": cleaned_length,
        "total_route_time": total_route_time,
        "waste": waste,
        "coverage": coverage,
        "efficiency": efficiency,
        "score": score,
    }


def submission_text(instance: Instance, solution: Solution) -> str:
    lines = [str(instance.vehicle_count)]
    for nodes, cleaned in zip(solution.routes, solution.cleaned):
        lines.append(str(len(nodes) - 1))
        lines.append(" ".join(map(str, nodes)))
        lines.append(" ".join(map(str, sorted(cleaned))))
    return "\n".join(lines) + "\n"


def solve_file(path: Path, output_dir: Path, restarts: int) -> dict[str, float | int]:
    instance = load_instance(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    instance_path = output_dir / f"{instance.name}_instance.txt"
    instance_path.write_text("\n".join(instance.raw_lines) + "\n", encoding="utf-8")
    solver = Solver(instance)
    solution = solver.solve(restarts=restarts)
    report = validate_solution(instance, solution)
    submission_path = output_dir / f"{instance.name}_submission.txt"
    submission_path.write_text(submission_text(instance, solution), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--restarts", type=int, default=250)
    args = parser.parse_args()
    for input_path in args.inputs:
        try:
            report = solve_file(input_path, args.output_dir, args.restarts)
        except RuntimeError as error:
            print(f"{input_path.stem}: INFEASIBLE {error}")
            continue
        metrics = " ".join(
            f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in report.items()
        )
        print(f"{input_path.stem}: VALID {metrics}")


if __name__ == "__main__":
    main()