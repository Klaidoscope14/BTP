"""
World (map) definition for the fixed-order EVRP.

A World holds:
  - a depot, N customers and M charging stations on a 2D plane
  - a pre-computed distance matrix between ALL nodes (the graph is fully known)
  - a 24-hour price schedule for each station (one price per hour)
  - the customer visit order — a subset of the customers, in the exact order
    they must be visited (supplied by the user, or sampled for training)

Worlds come from three places:
  - generate_world(...)        random synthetic map (for experiments)
  - load_world("map.json")     a map + visit order supplied by the user
  - with_visit_order(w, [...]) the same map with a different visit order

JSON format accepted by load_world / world_from_dict:

    {
      "depot":     {"name": "Depot", "x": 0, "y": 0},
      "customers": [{"name": "C01", "x": 12.5, "y": -3.0,
                     "window": [0, 40]}, ...],              # optional
      "stations":  [{"name": "S1", "x": 20, "y": 5,
                     "prices": [24 hourly $/kWh values]}, ...],
      "visit_order": ["C07", "C02", "C11"],      # names (or 0-based indices)
      "price_seed": 7,                           # optional: used for stations
                                                 #   that have no "prices"
      "params": {"battery_capacity": 100, ...}   # optional, see WORLD_PARAMS
    }

"window" is an optional delivery time window [earliest, latest] in hours
after the vehicle leaves the depot (either end may be null). Arriving before
`earliest` means waiting; arriving after `latest` is late. Customers without
a window can be reached at any time.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence, Union

import numpy as np

# Vehicle / cost constants and their defaults. Any of these can be overridden
# through the "params" block of a JSON map.
WORLD_PARAMS = {
    "battery_capacity": 100.0,       # kWh
    "consumption_per_dist": 0.6,     # kWh per distance unit
    "charge_rate": 40.0,             # kWh per hour
    "speed": 1.5,                    # distance units per hour
    "distance_cost_per_unit": 0.03,  # $ per distance unit (wear and tear)
    "start_time": 6.0,               # hour of day the vehicle leaves the depot
    "late_penalty_per_hour": 5.0,    # $ per hour a customer is reached late
    "late_penalty_fixed": 10.0,      # $ per customer reached late
}
DEFAULT_PRICE_MIN = 0.06
DEFAULT_PRICE_MAX = 0.40


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
    # (num_customers, 2) [earliest, latest] in hours after departure, or None.
    # Missing ends are -inf / +inf.
    time_windows: Optional[np.ndarray] = None

    battery_capacity: float = WORLD_PARAMS["battery_capacity"]
    consumption_per_dist: float = WORLD_PARAMS["consumption_per_dist"]
    charge_rate: float = WORLD_PARAMS["charge_rate"]
    speed: float = WORLD_PARAMS["speed"]
    distance_cost_per_unit: float = WORLD_PARAMS["distance_cost_per_unit"]
    start_time: float = WORLD_PARAMS["start_time"]
    late_penalty_per_hour: float = WORLD_PARAMS["late_penalty_per_hour"]
    late_penalty_fixed: float = WORLD_PARAMS["late_penalty_fixed"]
    price_min: float = DEFAULT_PRICE_MIN
    price_max: float = DEFAULT_PRICE_MAX

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
        return float(self.distance_matrix[a, b])

    def price_at(self, station_idx: int, t: float) -> float:
        hour = int(t % 24.0) % 24
        return float(self.price_schedule[station_idx, hour])

    def station_global_index(self, station_local: int) -> int:
        return self.stations[station_local].index

    def station_local_index(self, global_idx: int) -> int:
        return global_idx - 1 - self.num_customers

    def battery_range(self) -> float:
        return self.battery_capacity / self.consumption_per_dist

    def energy_for_dist(self, d: float) -> float:
        return d * self.consumption_per_dist

    def charge_time(self, energy: float) -> float:
        return energy / self.charge_rate

    def avg_price(self) -> float:
        return float(self.price_schedule.mean())

    @property
    def has_time_windows(self) -> bool:
        return self.time_windows is not None and bool(
            np.isfinite(self.time_windows).any())

    def window(self, customer_local: int):
        """(earliest, latest) in hours after departure; (-inf, inf) if none."""
        if self.time_windows is None:
            return -math.inf, math.inf
        lo, hi = self.time_windows[customer_local]
        return float(lo), float(hi)

    def validate(self) -> None:
        """Raise ValueError if some leg can't be driven even on a full battery
        and with the best possible station detour."""
        seq = ([self.depot.index] + [self.customers[i].index for i in self.visit_order]
               + [self.depot.index])
        for a, b in zip(seq, seq[1:]):
            if self.energy_for_dist(self.dist(a, b)) <= self.battery_capacity:
                continue
            ok = any(
                self.energy_for_dist(self.dist(a, s.index)) <= self.battery_capacity
                and self.energy_for_dist(self.dist(s.index, b)) <= self.battery_capacity
                for s in self.stations
            )
            if not ok:
                raise ValueError(
                    f"Leg {self.all_nodes[a].name} -> {self.all_nodes[b].name} "
                    f"({self.dist(a, b):.1f} units) is longer than the battery "
                    f"range ({self.battery_range():.1f}) and no station bridges it."
                )


# ──────────────────────────────────────────────────────────────────────────
# Construction helpers
# ──────────────────────────────────────────────────────────────────────────

def _build_world(depot, customers, stations, visit_order, price_schedule,
                 params: Optional[dict] = None, time_windows=None) -> World:
    all_nodes = [depot] + customers + stations
    for i, node in enumerate(all_nodes):
        node.index = i

    n = len(all_nodes)
    dist_mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(all_nodes[i].x - all_nodes[j].x,
                           all_nodes[i].y - all_nodes[j].y)
            dist_mat[i, j] = dist_mat[j, i] = d

    p = dict(WORLD_PARAMS)
    if params:
        unknown = set(params) - set(WORLD_PARAMS)
        if unknown:
            raise ValueError(f"Unknown world params: {sorted(unknown)}")
        p.update({k: float(v) for k, v in params.items()})

    price_schedule = np.asarray(price_schedule, dtype=np.float32)
    if time_windows is not None:
        time_windows = np.asarray(time_windows, dtype=np.float64)
    return World(
        depot=depot, customers=customers, stations=stations,
        visit_order=list(visit_order), distance_matrix=dist_mat,
        price_schedule=price_schedule, all_nodes=all_nodes,
        time_windows=time_windows,
        price_min=min(DEFAULT_PRICE_MIN, float(price_schedule.min())),
        price_max=max(DEFAULT_PRICE_MAX, float(price_schedule.max())),
        **p,
    )


def random_price_schedule(num_stations: int, seed: Optional[int] = None) -> np.ndarray:
    """Random hourly step-function prices, one row per station."""
    np_rng = np.random.RandomState(seed)
    sched = np_rng.uniform(DEFAULT_PRICE_MIN, DEFAULT_PRICE_MAX,
                           size=(num_stations, 24))
    return np.round(sched, 3).astype(np.float32)


def generate_world(
    num_customers: int = 10,
    num_stations: int = 6,
    grid_size: float = 40.0,
    num_visit: Optional[int] = None,
    seed: Optional[int] = None,
    battery_capacity: float = WORLD_PARAMS["battery_capacity"],
    consumption_per_dist: float = WORLD_PARAMS["consumption_per_dist"],
) -> World:
    """Random synthetic map with a random visit order over `num_visit` customers."""
    rng = random.Random(seed)
    if num_visit is None:
        num_visit = num_customers

    depot = Node("Depot", 0.0, 0.0, "depot")
    customers = [
        Node(f"C{i+1:02d}", round(rng.uniform(-grid_size, grid_size), 1),
             round(rng.uniform(-grid_size, grid_size), 1), "customer")
        for i in range(num_customers)
    ]
    stations = []
    for i in range(num_stations):
        angle = 2 * math.pi * i / num_stations + rng.uniform(-0.4, 0.4)
        r = rng.uniform(grid_size * 0.25, grid_size * 0.75)
        stations.append(Node(f"S{i+1}", round(r * math.cos(angle), 1),
                             round(r * math.sin(angle), 1), "station"))

    visit_order = rng.sample(range(num_customers), num_visit)
    rng.shuffle(visit_order)

    return _build_world(
        depot, customers, stations, visit_order,
        random_price_schedule(num_stations, seed),
        params={"battery_capacity": battery_capacity,
                "consumption_per_dist": consumption_per_dist},
    )


def _resolve_order(customers: List[Node], order: Sequence[Union[str, int]]) -> List[int]:
    names = {c.name: i for i, c in enumerate(customers)}
    out = []
    for item in order:
        if isinstance(item, str):
            if item not in names:
                raise ValueError(f"Unknown customer '{item}' in visit order. "
                                 f"Known: {', '.join(names)}")
            out.append(names[item])
        else:
            i = int(item)
            if not 0 <= i < len(customers):
                raise ValueError(f"Customer index {i} out of range 0..{len(customers)-1}")
            out.append(i)
    if len(set(out)) != len(out):
        raise ValueError("Visit order lists the same customer more than once.")
    if not out:
        raise ValueError("Visit order is empty.")
    return out


def with_visit_order(world: World, order: Sequence[Union[str, int]]) -> World:
    """Same map, new visit order (customer names or 0-based indices)."""
    w = replace(world, visit_order=_resolve_order(world.customers, order))
    w.validate()
    return w


def with_time_windows(world: World, windows: Optional[np.ndarray]) -> World:
    """Same map and order, different time windows (None removes them)."""
    if windows is not None:
        windows = np.asarray(windows, dtype=np.float64)
        if windows.shape != (world.num_customers, 2):
            raise ValueError(f"time windows need shape ({world.num_customers}, 2), "
                             f"got {windows.shape}")
    return replace(world, time_windows=windows)


def _parse_window(spec, name: str):
    if spec is None:
        return -math.inf, math.inf
    if not isinstance(spec, (list, tuple)) or len(spec) != 2:
        raise ValueError(f"Customer {name}: 'window' must be [earliest, latest]")
    lo = -math.inf if spec[0] is None else float(spec[0])
    hi = math.inf if spec[1] is None else float(spec[1])
    if lo > hi:
        raise ValueError(f"Customer {name}: window earliest {lo} > latest {hi}")
    return lo, hi


def world_from_dict(spec: dict, visit_order: Optional[Sequence] = None) -> World:
    """Build a World from a user spec (see module docstring for the format)."""
    def node(d, kind, default_name):
        return Node(str(d.get("name", default_name)), float(d["x"]), float(d["y"]), kind)

    depot = node(spec.get("depot", {"x": 0.0, "y": 0.0}), "depot", "Depot")
    customers = [node(c, "customer", f"C{i+1:02d}")
                 for i, c in enumerate(spec["customers"])]
    station_specs = spec["stations"]
    stations = [node(s, "station", f"S{i+1}") for i, s in enumerate(station_specs)]
    if not customers or not stations:
        raise ValueError("A map needs at least one customer and one station.")

    fallback = random_price_schedule(len(stations), spec.get("price_seed", 0))
    prices = []
    for i, s in enumerate(station_specs):
        if "prices" in s:
            row = [float(v) for v in s["prices"]]
            if len(row) != 24:
                raise ValueError(f"Station {stations[i].name}: 'prices' needs 24 "
                                 f"hourly values, got {len(row)}")
            prices.append(row)
        else:
            prices.append(fallback[i].tolist())

    order_spec = visit_order if visit_order is not None else spec.get(
        "visit_order", [c.name for c in customers])
    order = _resolve_order(customers, order_spec)

    windows = None
    if any("window" in c for c in spec["customers"]):
        windows = [_parse_window(c.get("window"), customers[i].name)
                   for i, c in enumerate(spec["customers"])]

    w = _build_world(depot, customers, stations, order, prices, spec.get("params"),
                     time_windows=windows)
    w.validate()
    return w


def load_world(path: str, visit_order: Optional[Sequence] = None) -> World:
    with open(path) as f:
        return world_from_dict(json.load(f), visit_order)


def world_to_dict(world: World) -> dict:
    def cust(i, c):
        d = {"name": c.name, "x": c.x, "y": c.y}
        if world.time_windows is not None:
            lo, hi = world.window(i)
            if math.isfinite(lo) or math.isfinite(hi):
                d["window"] = [round(lo, 2) if math.isfinite(lo) else None,
                               round(hi, 2) if math.isfinite(hi) else None]
        return d

    return {
        "depot": {"name": world.depot.name, "x": world.depot.x, "y": world.depot.y},
        "customers": [cust(i, c) for i, c in enumerate(world.customers)],
        "stations": [{"name": s.name, "x": s.x, "y": s.y,
                      "prices": [round(float(p), 3) for p in world.price_schedule[i]]}
                     for i, s in enumerate(world.stations)],
        "visit_order": [world.customers[i].name for i in world.visit_order],
        "params": {k: getattr(world, k) for k in WORLD_PARAMS},
    }


def save_world(world: World, path: str) -> None:
    with open(path, "w") as f:
        json.dump(world_to_dict(world), f, indent=2)
