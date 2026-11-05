"""
World generation for the fixed-order EVRP.

Produces:
  - A depot, N customers, M charging stations on a 2D plane
  - A pre-computed distance matrix between ALL nodes
  - A random step-function price schedule for each station (24 hourly blocks)
  - A fixed customer visit order (random permutation of a user-specified subset)

Everything is seeded for reproducibility but the seed can be randomized per
episode during training so the agent generalizes.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Node:
    name: str
    x: float
    y: float
    kind: str          # 'depot' | 'customer' | 'station'
    index: int = -1    # position in the global node list (set by World)


@dataclass
class World:
    depot: Node
    customers: List[Node]
    stations: List[Node]
    visit_order: List[int]            # indices into self.customers
    distance_matrix: np.ndarray       # shape (total_nodes, total_nodes)
    price_schedule: np.ndarray        # shape (num_stations, 24)
    all_nodes: List[Node] = field(default_factory=list)

    # --- constants set at construction ---
    battery_capacity: float = 100.0
    consumption_per_dist: float = 0.6
    charge_rate: float = 40.0
    speed: float = 1.5
    distance_cost_per_unit: float = 0.03
    start_time: float = 6.0
    price_min: float = 0.06
    price_max: float = 0.40

    @property
    def num_customers(self):
        return len(self.customers)

    @property
    def num_stations(self):
        return len(self.stations)

    @property
    def num_nodes(self):
        return len(self.all_nodes)

    @property
    def ordered_customers(self) -> List[Node]:
        return [self.customers[i] for i in self.visit_order]

    def dist(self, a: int, b: int) -> float:
        return self.distance_matrix[a, b]

    def price_at(self, station_idx: int, t: float) -> float:
        hour = int(t % 24.0) % 24
        return float(self.price_schedule[station_idx, hour])

    def station_global_index(self, station_local: int) -> int:
        return self.stations[station_local].index

    def battery_range(self) -> float:
        return self.battery_capacity / self.consumption_per_dist

    def energy_for_dist(self, d: float) -> float:
        return d * self.consumption_per_dist

    def charge_time(self, energy: float) -> float:
        return energy / self.charge_rate

    def avg_price(self) -> float:
        return float(self.price_schedule.mean())


def generate_world(
    num_customers: int = 10,
    num_stations: int = 6,
    grid_size: float = 40.0,
    num_visit: Optional[int] = None,
    seed: Optional[int] = None,
    battery_capacity: float = 100.0,
    consumption_per_dist: float = 0.6,
) -> World:
    """Build a complete World instance.

    Args:
        num_customers: total customers placed in the world
        num_stations: charging stations
        grid_size: half-width of the square grid
        num_visit: how many customers to visit (None = all)
        seed: RNG seed (None = random)
        battery_capacity: vehicle battery in kWh
        consumption_per_dist: kWh per unit distance
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    if num_visit is None:
        num_visit = num_customers

    # --- place nodes ---
    depot = Node("Depot", 0.0, 0.0, "depot")
    customers = []
    for i in range(num_customers):
        x = rng.uniform(-grid_size, grid_size)
        y = rng.uniform(-grid_size, grid_size)
        customers.append(Node(f"C{i+1:02d}", round(x, 1), round(y, 1), "customer"))

    stations = []
    for i in range(num_stations):
        angle = 2 * math.pi * i / num_stations + rng.uniform(-0.4, 0.4)
        r = rng.uniform(grid_size * 0.25, grid_size * 0.75)
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        stations.append(Node(f"S{i+1}", round(x, 1), round(y, 1), "station"))

    # global node list: [depot, customers..., stations...]
    all_nodes = [depot] + customers + stations
    for i, node in enumerate(all_nodes):
        node.index = i

    # --- distance matrix ---
    n = len(all_nodes)
    dist_mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(all_nodes[i].x - all_nodes[j].x,
                           all_nodes[i].y - all_nodes[j].y)
            dist_mat[i, j] = d
            dist_mat[j, i] = d

    # --- visit order: random subset + random permutation ---
    chosen = rng.sample(range(num_customers), num_visit)
    rng.shuffle(chosen)
    visit_order = chosen

    # --- price schedule: random step function per station ---
    price_min, price_max = 0.06, 0.40
    price_schedule = np_rng.uniform(price_min, price_max,
                                     size=(num_stations, 24)).astype(np.float32)
    price_schedule = np.round(price_schedule, 3)

    return World(
        depot=depot,
        customers=customers,
        stations=stations,
        visit_order=visit_order,
        distance_matrix=dist_mat,
        price_schedule=price_schedule,
        all_nodes=all_nodes,
        battery_capacity=battery_capacity,
        consumption_per_dist=consumption_per_dist,
        charge_rate=40.0,
        speed=1.5,
        distance_cost_per_unit=0.03,
        start_time=6.0,
        price_min=price_min,
        price_max=price_max,
    )
