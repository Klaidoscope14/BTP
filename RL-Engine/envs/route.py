"""
Leg dynamics of the fixed-order EVRP, shared by the Gym env, the baselines
and the exact solver so that all of them price a route identically.

A route is the mandatory sequence  depot -> c_1 -> ... -> c_N -> depot.
At every leg the vehicle either drives straight to the next stop (action 0)
or detours through charging station k and charges to one of L target levels
(actions 1 .. M*L). The action is decoded as

    station = (action - 1) // L,   level = (action - 1) % L

Charge targets (partial charging)
---------------------------------
For a detour via station s on the leg towards stop y, with battery b_s on
arrival at s:

    lo = e(s, y) + need[y]              least charge that still lets the
                                        route be finished
    hi = min(capacity,                  most charge that can ever be used:
             e(s, y) + E_direct(y))     enough to drive the rest of the route
                                        without stopping again

    target(level) = lo + level / (L - 1) * (hi - lo)

so level 0 buys only what is needed to stay feasible (the vehicle will
charge again later, hopefully somewhere cheaper) and level L-1 fills up to
full *or* to exactly what the rest of the route needs, whichever is smaller.
No level ever buys energy the route can't use. A level whose target is below
b_s would mean not charging at all, so it is masked out.

With charge_levels = 0 the legacy behaviour is kept: one level, always
charge to 100 %.

Time windows
------------
Customer windows [earliest, latest] are in hours after departure. Arriving
early means waiting until `earliest`; arriving after `latest` costs
late_penalty_fixed + late_penalty_per_hour * hours_late.

Deadlines are treated as hard through the action mask. At reset a backward
table is built:

    T[j][b] = latest time the vehicle may arrive at stop j with battery b
              and still meet every later deadline (charging included)

computed over a 1 %-of-capacity battery grid, rounding battery *down*, so it
is conservative: an action is allowed only if it arrives at the next stop
at time t with battery b and t <= T[next][b]. Following the mask therefore
reaches every customer on time whenever the windows allow it at all; only
when no action can meet them is the time check dropped and lateness paid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from utils.world import World

EPS = 1e-6


@dataclass
class RouteContext:
    world: World
    seq: List[int]                 # global node index of each mandatory stop
    cust: List[int]                # customer-local index per stop (-1 = depot)
    need: List[float]              # least battery on arrival to finish the route
    e_direct: List[float]          # energy to drive stop j -> ... -> depot directly
    win_open: List[float]          # absolute times on the env clock
    win_close: List[float]
    latest: List[float]            # latest arrival meeting later deadlines, driving direct
    charge_levels: int             # 0 = legacy "always charge to 100 %"
    late_table: Optional[np.ndarray] = None   # T[j][b], see module docstring
    grid_step: float = 1.0                    # kWh per late_table column

    @property
    def n_levels(self) -> int:
        return max(1, self.charge_levels)

    @property
    def n_actions(self) -> int:
        return 1 + self.world.num_stations * self.n_levels

    @property
    def n_legs(self) -> int:
        return len(self.seq)

    @property
    def has_windows(self) -> bool:
        return any(math.isfinite(c) for c in self.win_close) or any(
            math.isfinite(o) for o in self.win_open)


@dataclass
class State:
    leg: int          # index of the next mandatory stop in ctx.seq
    node: int         # global index of the current node
    battery: float
    time: float       # absolute env clock (hours; starts at world.start_time)


@dataclass
class Outcome:
    """Result of taking one action from a state (nothing is mutated)."""
    drivable: bool = True          # False -> the battery runs flat on this leg
    valid_charge: bool = True      # False -> charge target below current battery
    station: Optional[int] = None  # station-local index of the detour
    target: float = 0.0
    segments: List[Tuple[int, int, float, float, float]] = field(default_factory=list)
    # (from, to, distance, battery after driving, time after driving)
    charge: Optional[Tuple[str, float, float, float, float]] = None
    # (station name, hour of day, $/kWh, kWh, $)
    charge_time: float = 0.0
    distance: float = 0.0
    dist_cost: float = 0.0
    charge_cost: float = 0.0
    arrive_battery: float = 0.0    # at the next mandatory stop
    arrive_time: float = 0.0       # at the next mandatory stop, before waiting
    wait: float = 0.0
    late_hours: float = 0.0
    late_cost: float = 0.0
    dead_at: Optional[Tuple[int, int, float, float]] = None   # (from, to, need, have)

    @property
    def cost(self) -> float:
        return self.dist_cost + self.charge_cost + self.late_cost

    def next_state(self, state: State, ctx: RouteContext) -> State:
        return State(state.leg + 1, ctx.seq[state.leg], self.arrive_battery,
                     self.arrive_time + self.wait)


# ──────────────────────────────────────────────────────────────────────────
# Context construction
# ──────────────────────────────────────────────────────────────────────────

def build_context(world: World, charge_levels: int = 5) -> RouteContext:
    w = world
    seq = [w.customers[i].index for i in w.visit_order] + [w.depot.index]
    cust = list(w.visit_order) + [-1]
    n = len(seq)

    e_direct = [0.0] * n
    for j in range(n - 2, -1, -1):
        e_direct[j] = e_direct[j + 1] + w.energy_for_dist(w.dist(seq[j], seq[j + 1]))

    win_open, win_close = [], []
    for c in cust:
        lo, hi = (-math.inf, math.inf) if c < 0 else w.window(c)
        win_open.append(w.start_time + lo)
        win_close.append(w.start_time + hi)

    latest = [math.inf] * n
    for j in range(n - 2, -1, -1):
        travel = w.dist(seq[j], seq[j + 1]) / w.speed
        if max(win_open[j], -1e18) + travel > latest[j + 1] + EPS:
            latest[j] = -math.inf          # infeasible no matter how early
        else:
            latest[j] = min(win_close[j], latest[j + 1] - travel)

    ctx = RouteContext(
        world=w, seq=seq, cust=cust, need=_min_battery(w, seq), e_direct=e_direct,
        win_open=win_open, win_close=win_close, latest=latest,
        charge_levels=int(charge_levels),
    )
    if ctx.has_windows:
        _build_late_table(ctx)
    return ctx


def _build_late_table(ctx: RouteContext, grid: int = 100) -> None:
    """Fill ctx.late_table (see module docstring), vectorised over battery.

    T[last][b] = +inf (the depot has no deadline). Going backwards, from stop
    j with battery b, an action reaching stop j+1 after dt hours with battery
    b' is usable if the vehicle leaves j by  T[j+1][b'] - dt. Leaving j
    happens at max(arrival, earliest_j), so
        D[j][b] = max over actions of (T[j+1][b'] - dt)
        T[j][b] = min(latest_j, D[j][b])   if earliest_j <= D[j][b], else -inf
    T is non-decreasing in b, so rounding b' down never overestimates it.
    """
    w = ctx.world
    cap = w.battery_capacity
    step = cap / grid
    b = np.arange(grid + 1) * step
    n = ctx.n_legs
    T = np.full((n, grid + 1), -np.inf)
    T[n - 1, :] = np.inf

    def lookup(row, bp):
        idx = np.floor(bp / step + 1e-9).astype(int)
        out = np.full(bp.shape, -np.inf)
        ok = idx >= 0
        out[ok] = row[np.minimum(idx[ok], grid)]
        return out

    for j in range(n - 2, -1, -1):
        x, y, leg = ctx.seq[j], ctx.seq[j + 1], j + 1
        row = T[j + 1]
        d = w.dist(x, y)
        bp = b - w.energy_for_dist(d)
        D = np.where(bp >= -EPS, lookup(row, np.maximum(bp, 0.0)) - d / w.speed, -np.inf)
        for k in range(w.num_stations):
            sg = w.station_global_index(k)
            d1, d2 = w.dist(x, sg), w.dist(sg, y)
            e1, e2 = w.energy_for_dist(d1), w.energy_for_dist(d2)
            bs = b - e1
            for level in range(ctx.n_levels):
                tgt = charge_target(ctx, leg, k, level)
                after = np.maximum(bs, tgt)
                ok = (bs >= -EPS) & (tgt >= bs - 1e-4) & (e2 <= after + EPS)
                if not ok.any():
                    continue
                dt = (d1 + d2) / w.speed + np.maximum(0.0, tgt - bs) / w.charge_rate
                val = lookup(row, np.maximum(after - e2, 0.0)) - dt
                D = np.maximum(D, np.where(ok, val, -np.inf))
        T[j] = np.where(ctx.win_open[j] <= D + EPS, np.minimum(ctx.win_close[j], D), -np.inf)

    ctx.late_table = T
    ctx.grid_step = step


def latest_arrival(ctx: RouteContext, leg: int, battery: float) -> float:
    """Latest time to arrive at stop `leg` with `battery` and still meet
    every later deadline (+inf without windows)."""
    if ctx.late_table is None or leg >= ctx.n_legs:
        return math.inf
    if battery < -EPS:
        return -math.inf
    idx = int(math.floor(max(battery, 0.0) / ctx.grid_step + 1e-9))
    return float(ctx.late_table[leg, min(idx, ctx.late_table.shape[1] - 1)])


def _min_battery(w: World, seq: List[int]) -> List[float]:
    """need[j] = least battery on arriving at stop j to still finish the route.

    Worked backwards from the depot. Charging to full is the most permissive
    option, so from stop X (next stop Y) the route can be finished with
    battery b iff
        b >= e(X,Y) + need[Y]                         (go direct), or
        b >= e(X,s) for a station s with
             capacity - e(s,Y) >= need[Y]             (detour via s).
    Every partial-charge level >= lo keeps that guarantee, and the top level
    reaches it whenever full charging would.
    """
    cap = w.battery_capacity
    need = [0.0] * len(seq)
    for j in range(len(seq) - 2, -1, -1):
        x, y = seq[j], seq[j + 1]
        best = w.energy_for_dist(w.dist(x, y)) + need[j + 1]
        for s in w.stations:
            if cap - w.energy_for_dist(w.dist(s.index, y)) >= need[j + 1] - EPS:
                best = min(best, w.energy_for_dist(w.dist(x, s.index)))
        need[j] = best if best <= cap + EPS else math.inf
    return need


def initial_state(ctx: RouteContext) -> State:
    w = ctx.world
    return State(0, w.depot.index, w.battery_capacity, w.start_time)


# ──────────────────────────────────────────────────────────────────────────
# Transition
# ──────────────────────────────────────────────────────────────────────────

def decode(ctx: RouteContext, action: int) -> Tuple[Optional[int], int]:
    if action == 0:
        return None, 0
    return (action - 1) // ctx.n_levels, (action - 1) % ctx.n_levels


def encode(ctx: RouteContext, station: int, level: int) -> int:
    return 1 + station * ctx.n_levels + level


def charge_bounds(ctx: RouteContext, leg: int, station: int) -> Tuple[float, float]:
    """(lo, hi) charge targets for a detour via `station` on leg `leg`."""
    w = ctx.world
    cap = w.battery_capacity
    e2 = w.energy_for_dist(w.dist(w.station_global_index(station), ctx.seq[leg]))
    hi = min(cap, e2 + ctx.e_direct[leg])
    lo = min(e2 + ctx.need[leg], hi)
    return lo, hi


def charge_target(ctx: RouteContext, leg: int, station: int, level: int) -> float:
    if ctx.charge_levels == 0:
        return ctx.world.battery_capacity
    lo, hi = charge_bounds(ctx, leg, station)
    L = ctx.n_levels
    f = level / (L - 1) if L > 1 else 1.0
    return lo + f * (hi - lo)


def simulate(ctx: RouteContext, s: State, action: int) -> Outcome:
    """What happens if `action` is taken in state `s` (pure function)."""
    w = ctx.world
    out = Outcome()
    y = ctx.seq[s.leg]
    station, level = decode(ctx, action)
    out.station = station
    battery, t, node = s.battery, s.time, s.node

    hops = [y] if station is None else [w.station_global_index(station), y]
    for to in hops:
        d = w.dist(node, to)
        e = w.energy_for_dist(d)
        if e > battery + EPS:
            out.drivable = False
            out.dead_at = (node, to, e, battery)
            return out
        battery -= e
        t += d / w.speed
        out.distance += d
        out.dist_cost += d * w.distance_cost_per_unit
        out.segments.append((node, to, d, battery, t))

        if w.all_nodes[to].kind == "station":
            target = charge_target(ctx, s.leg, station, level)
            out.target = target
            if target < battery - 1e-4:
                out.valid_charge = False
            kwh = max(0.0, target - battery)
            price = w.price_at(station, t)
            cost = price * kwh
            out.charge = (w.all_nodes[to].name, t % 24, price, kwh, cost)
            out.charge_cost = cost
            out.charge_time = w.charge_time(kwh)
            battery = max(battery, target)
            t += out.charge_time
        node = to

    out.arrive_battery = battery
    out.arrive_time = t
    j = s.leg
    if t < ctx.win_open[j]:
        out.wait = ctx.win_open[j] - t
    if t > ctx.win_close[j] + EPS:
        out.late_hours = t - ctx.win_close[j]
        out.late_cost = w.late_penalty_fixed + w.late_penalty_per_hour * out.late_hours
    return out


def action_mask(ctx: RouteContext, s: State, use_time: bool = True) -> np.ndarray:
    """Valid actions.

    1. Battery (exact): drivable, a real charge, and arrival at the next stop
       with at least need[] — following the mask never strands the vehicle.
    2. Time windows: arrival (t, b) at the next stop with t <= T[next][b],
       i.e. every later deadline can still be met (charging included).
    If no action passes both, time is dropped (lateness is then unavoidable
    and simply paid for); if still none, anything drivable; else action 0.
    """
    mask = np.zeros(ctx.n_actions, dtype=bool)
    if s.leg >= ctx.n_legs:
        mask[0] = True
        return mask
    need = ctx.need[s.leg]
    battery_ok = np.zeros_like(mask)
    drivable = np.zeros_like(mask)
    for a in range(ctx.n_actions):
        o = simulate(ctx, s, a)
        if not o.drivable:
            continue
        drivable[a] = True
        if o.valid_charge and o.arrive_battery >= need - EPS:
            battery_ok[a] = True
            mask[a] = (not use_time) or (
                o.arrive_time <= latest_arrival(ctx, s.leg, o.arrive_battery) + EPS)
    if mask.any():
        return mask
    if battery_ok.any():
        return battery_ok
    if drivable.any():
        return drivable
    mask[0] = True
    return mask


# ──────────────────────────────────────────────────────────────────────────
# Helpers for baselines, bounds and window generation
# ──────────────────────────────────────────────────────────────────────────

def direct_lateness(ctx: RouteContext, s: State) -> float:
    """Lateness cost if the vehicle drove straight through the rest of the
    route with no charging stops — a lower bound on the real lateness cost,
    because detours and charging only make arrivals later."""
    w = ctx.world
    t, node, cost = s.time, s.node, 0.0
    for j in range(s.leg, ctx.n_legs):
        t += w.dist(node, ctx.seq[j]) / w.speed
        if t > ctx.win_close[j] + EPS:
            cost += w.late_penalty_fixed + w.late_penalty_per_hour * (t - ctx.win_close[j])
        t = max(t, ctx.win_open[j])
        node = ctx.seq[j]
    return cost


def forced_rollout(ctx: RouteContext) -> List[float]:
    """Arrival times (env clock) at each stop for the simple policy 'go
    direct; when the mask forbids it, charge fully at the nearest valid
    station'. Used to generate time windows that are guaranteed feasible."""
    w = ctx.world
    s = initial_state(ctx)
    arrivals = []
    top = ctx.n_levels - 1
    while s.leg < ctx.n_legs:
        mask = action_mask(ctx, s, use_time=False)
        if mask[0]:
            a = 0
        else:
            cands = [k for k in range(w.num_stations) if mask[encode(ctx, k, top)]]
            if not cands:
                cands_a = [a for a in range(1, ctx.n_actions) if mask[a]]
                a = cands_a[0] if cands_a else 0
            else:
                k = min(cands, key=lambda k: w.dist(s.node, w.station_global_index(k)))
                a = encode(ctx, k, top)
        o = simulate(ctx, s, a)
        arrivals.append(o.arrive_time)
        s = o.next_state(s, ctx)
    return arrivals


def generate_time_windows(world: World, rng: np.random.Generator,
                          slack: Tuple[float, float] = (2.0, 24.0),
                          width: float = 24.0, charge_levels: int = 5) -> np.ndarray:
    """Random windows for the customers in world.visit_order that a simple
    feasible plan meets: each latest = (arrival under the forced-charging
    plan) + slack ~ U(slack); earliest = that arrival - U(0, width).
    Customers outside the visit order get no window."""
    ctx = build_context(world, charge_levels)
    ref = forced_rollout(ctx)
    tw = np.full((world.num_customers, 2), [-np.inf, np.inf], dtype=np.float64)
    for j, c in enumerate(ctx.cust[:-1]):
        rel = ref[j] - world.start_time
        tw[c, 1] = round(rel + rng.uniform(*slack), 2)
        tw[c, 0] = round(max(0.0, rel - rng.uniform(0.0, width)), 2)
    return tw
