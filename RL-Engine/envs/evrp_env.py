"""
Fixed-Order EVRP Gymnasium Environment
=======================================
The customer visit order is fixed (given by the user). The graph — node
positions, distances and every station's 24-hour price schedule — is fully
known. At each step the agent decides how to drive the next leg:

  * go directly to the next mandatory stop, or
  * detour to charging station k and charge up to a chosen level, then
    continue to the next stop (partial charging).

Action space: Discrete(1 + M * L)
  0                    = go directly to the next mandatory stop
  1 + k * L + l        = detour to station k, charge to level l
                         (k in 0..M-1, l in 0..L-1)

Charge levels (`charge_levels`, default (0.0, 0.25, 0.5, 0.75, 1.0)) are
target battery levels as a fraction of capacity, on leaving the station.
The target is never below "just enough": the battery needed to reach the next
stop and still be able to finish the route. So level 0.0 means "charge the
minimum", and 1.0 means "charge to full". A level the vehicle already has
(nothing to charge) is masked out.

One step = one leg (current position -> next mandatory stop, optionally via a
station). An episode has len(visit_order) + 1 steps: one per customer plus
the return to the depot.

Time windows (optional): customer j may have a window [earliest, latest]
(absolute hours; the vehicle leaves at 06:00 on day 0). Arriving early means
waiting until `earliest`. Arriving after `latest` is late and costs
    late_penalty_fixed + late_penalty_per_hour * hours_late.
The action mask keeps only actions that reach the next stop on time — with
"on time" tightened by the deadlines of the stops after it — whenever any
such action exists, so the vehicle is late only when it can't be avoided.
`time_windows` picks where windows come from: "map" (the user's JSON),
"random" (new feasible windows each episode, with probability `tw_prob`) or
"off" (ignore windows).

Reward = -(distance cost + charging cost + lateness penalty) of the leg, i.e.
the episode return is exactly  completion_bonus - objective  when the route
is completed, so maximising reward is the same as minimising the objective
reported by evaluate.py (= total cost when there are no time windows).
Running out of battery ends the episode with a penalty larger than the cost
of any feasible completion.

Where worlds come from (checked in this order on every reset):
  1. fixed_world + randomize_order=False -> exactly that map and order
  2. fixed_world + randomize_order=True  -> that map, a new random subset and
                                            order of its customers each episode
  3. world_seed_pool                     -> random maps, cycling through the
                                            given seeds (used for eval/test)
  4. randomize_world=True                -> a fresh random map each episode,
                                            seed drawn from TRAIN_SEED_RANGE
  5. otherwise                           -> one random map from `seed`
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from utils.world import World, generate_world, with_visit_order

# Random-map seeds are split so training never sees evaluation/test maps.
TRAIN_SEED_RANGE = (0, 1_000_000)
EVAL_SEED_START = 2_000_000    # EvalCallback during training
TEST_SEED_START = 3_000_000    # evaluate.py

DEFAULT_CHARGE_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
EPS = 1e-6


def _squash(x: float, scale: float) -> float:
    """Map [0, inf) to [0, 1) smoothly, without the hard clip losing information."""
    x = max(0.0, x)
    return x / (x + scale)


class Leg(NamedTuple):
    """Result of driving one leg with one action (see EVRPEnv.simulate)."""
    station: Optional[int]       # local station index, or None for direct
    target: int                  # global index of the mandatory stop reached
    distance: float
    distance_cost: float
    energy_charged: float
    charge_price: float
    charge_cost: float
    charge_hour: float           # time of arrival at the station (abs hours)
    battery: float               # battery on arrival at `target`
    arrival: float               # time the vehicle reaches `target`
    time: float                  # time it can leave (after waiting for the window)
    late_hours: float = 0.0
    penalty: float = 0.0         # lateness penalty, $

    @property
    def cost(self) -> float:
        return self.distance_cost + self.charge_cost

    @property
    def wait(self) -> float:
        return self.time - self.arrival

    @property
    def objective(self) -> float:
        return self.cost + self.penalty


class EVRPEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_customers: int = 10,
        num_stations: int = 6,
        num_visit: Optional[int] = None,
        grid_size: float = 40.0,
        seed: Optional[int] = None,
        price_lookahead: int = 4,
        randomize_world: bool = True,
        fixed_world: Optional[World] = None,
        randomize_order: bool = False,
        visit_range: Optional[Tuple[int, int]] = None,
        world_seed_pool: Optional[Sequence[int]] = None,
        completion_bonus: float = 50.0,
        charge_levels: Sequence[float] = DEFAULT_CHARGE_LEVELS,
        time_windows: str = "map",
        tw_prob: float = 1.0,
        tw_slack: Tuple[float, float] = (0.5, 6.0),
        tw_width: Tuple[float, float] = (4.0, 16.0),
        late_penalty_fixed: float = 5.0,
        late_penalty_per_hour: float = 2.0,
    ):
        super().__init__()

        if fixed_world is not None:
            num_customers = fixed_world.num_customers
            num_stations = fixed_world.num_stations
        self.num_customers = num_customers
        self.num_stations = num_stations
        self.num_visit = num_visit or num_customers
        self.grid_size = grid_size
        self.base_seed = seed
        self.price_lookahead = price_lookahead
        self.randomize_world = randomize_world
        self.fixed_world = fixed_world
        self.randomize_order = randomize_order
        self.visit_range = visit_range
        self.world_seed_pool = list(world_seed_pool) if world_seed_pool else None
        self._pool_pos = 0
        self.completion_bonus = completion_bonus
        if time_windows not in ("map", "random", "off"):
            raise ValueError("time_windows must be 'map', 'random' or 'off'")
        self.time_windows = time_windows
        self.tw_prob = tw_prob
        self.tw_slack = tuple(tw_slack)
        self.tw_width = tuple(tw_width)
        self.late_penalty_fixed = late_penalty_fixed
        self.late_penalty_per_hour = late_penalty_per_hour

        levels = sorted(float(l) for l in charge_levels)
        if not levels or levels[0] < 0.0 or levels[-1] != 1.0:
            raise ValueError("charge_levels must lie in [0, 1] and include 1.0 (full)")
        self.charge_levels = tuple(levels)
        self.num_levels = len(levels)

        M = self.num_stations
        K = self.price_lookahead
        self.action_space = spaces.Discrete(1 + M * self.num_levels)
        self.obs_dim = 11 + M * (5 + K)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )

        self.world: Optional[World] = None
        self._base_world: Optional[World] = None
        self.reset(seed=seed)

    # ------------------------------------------------------------------ #
    # Action encoding
    # ------------------------------------------------------------------ #
    def encode(self, station: int, level: int) -> int:
        return 1 + station * self.num_levels + level

    def decode(self, action: int) -> Tuple[Optional[int], Optional[int]]:
        """action -> (station, level), or (None, None) for 'go direct'."""
        if action == 0:
            return None, None
        return divmod(action - 1, self.num_levels)

    @property
    def full_level(self) -> int:
        return self.num_levels - 1

    def set_visit_range(self, visit_range: Tuple[int, int]) -> None:
        """Change how many customers sampled visit orders have (curriculum)."""
        self.visit_range = tuple(visit_range)

    # ------------------------------------------------------------------ #
    # World selection
    # ------------------------------------------------------------------ #
    def _sample_num_visit(self, n_total: int, rng=None) -> int:
        if self.visit_range is None:
            return min(self.num_visit, n_total)
        rng = rng if rng is not None else self.np_random
        lo, hi = self.visit_range
        lo, hi = max(1, lo), min(hi, n_total)
        lo = min(lo, hi)
        return int(rng.integers(lo, hi + 1))

    def _next_pool_seed(self) -> int:
        s = self.world_seed_pool[self._pool_pos % len(self.world_seed_pool)]
        self._pool_pos += 1
        return s

    def _next_world(self):
        """(world, rng) for the next episode; rng also draws random windows,
        so pooled (eval/test) episodes get repeatable windows too."""
        if self.fixed_world is not None:
            if not self.randomize_order:
                rng = (np.random.default_rng(self._next_pool_seed())
                       if self.world_seed_pool else self.np_random)
                return self.fixed_world, rng
            # a pool makes the sampled orders repeatable (eval/test)
            rng = (np.random.default_rng(self._next_pool_seed())
                   if self.world_seed_pool else self.np_random)
            n = self.fixed_world.num_customers
            k = self._sample_num_visit(n, rng)
            order = rng.permutation(n)[:k]
            return with_visit_order(self.fixed_world, [int(i) for i in order]), rng

        if self.world_seed_pool:
            world_seed = self._next_pool_seed()
            rng = np.random.default_rng(world_seed)
        elif self.randomize_world:
            world_seed = int(self.np_random.integers(*TRAIN_SEED_RANGE))
            rng = self.np_random
        else:
            if self.world is not None and self._base_world is not None:
                return self._base_world, self.np_random
            world_seed = self.base_seed
            rng = np.random.default_rng(world_seed)

        world = generate_world(
            num_customers=self.num_customers,
            num_stations=self.num_stations,
            grid_size=self.grid_size,
            num_visit=self._sample_num_visit(self.num_customers, rng),
            seed=world_seed,
        )
        self._base_world = world
        return world, rng

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.world, rng = self._next_world()

        w = self.world
        self.current_node_idx = w.depot.index
        self.battery = w.battery_capacity
        self.time = w.start_time
        self.leg_index = 0
        self.total_distance = 0.0
        self.total_charge_cost = 0.0
        self.total_distance_cost = 0.0
        self.total_energy_charged = 0.0
        self.total_penalty = 0.0
        self.late_stops = 0
        self.late_hours = 0.0
        self.wait_hours = 0.0
        self.done = False
        self.log = []
        self.path = [w.depot.index]       # every node driven through, for plots
        self.charges = []                 # (station name, hour, price, kWh, $)
        self._mask_cache = None

        # mandatory stops: [cust_1, ..., cust_N, depot]
        self.mandatory_sequence = (
            [w.customers[i].index for i in w.visit_order] + [w.depot.index]
        )
        self.min_battery_on_arrival = self._compute_min_battery()
        self._set_windows(rng)
        return self._get_obs(), {}

    # ------------------------------------------------------------------ #
    # Time windows
    # ------------------------------------------------------------------ #
    def _set_windows(self, rng) -> None:
        """Fill tw_earliest / tw_latest (one entry per mandatory stop; the
        final depot has none) and the tightened latest_arrival table."""
        w = self.world
        n = len(self.mandatory_sequence)
        self.tw_earliest = [0.0] * n
        self.tw_latest = [float("inf")] * n
        mode = self.time_windows
        if mode == "map" and w.has_windows:
            for j, ci in enumerate(w.visit_order):
                self.tw_earliest[j], self.tw_latest[j] = w.window(ci)
        elif mode == "random" and rng.random() < self.tw_prob:
            self._random_windows(rng)
        self._compute_latest_arrival()

    def _random_windows(self, rng) -> None:
        """Random windows that are always achievable: drive a reference plan
        (fastest valid action every leg) and put each deadline a random
        slack after its reference arrival; the window opens a random width
        before the deadline (so arriving early sometimes means waiting)."""
        self.latest_arrival = [float("inf")] * len(self.mandatory_sequence)
        node, b, t = self.current_node_idx, self.battery, self.time
        for j in range(len(self.mandatory_sequence) - 1):
            legs = self.valid_legs(node, b, t, j)
            if not legs:
                break
            _, r = min(legs, key=lambda x: x[1].arrival)
            latest = r.arrival + float(rng.uniform(*self.tw_slack))
            earliest = max(self.world.start_time,
                           latest - float(rng.uniform(*self.tw_width)))
            self.tw_earliest[j], self.tw_latest[j] = earliest, latest
            node, b, t = r.target, r.battery, max(r.arrival, earliest)

    def _compute_latest_arrival(self) -> None:
        """latest_arrival[j] = latest time to reach stop j and still be able
        to reach every later stop on time (driving direct, the fastest way).
        A necessary condition, used to keep the mask from walking into a
        deadline it can no longer meet."""
        w = self.world
        seq = self.mandatory_sequence
        la = list(self.tw_latest)
        for j in range(len(seq) - 2, -1, -1):
            nxt = la[j + 1] - w.dist(seq[j], seq[j + 1]) / w.speed
            if nxt < self.tw_earliest[j]:
                nxt = float("inf")     # the next stop is late whatever we do here
            la[j] = min(la[j], nxt)
        self.latest_arrival = la

    @property
    def has_windows(self) -> bool:
        return any(np.isfinite(l) for l in self.tw_latest)

    # ------------------------------------------------------------------ #
    # Transition model (shared by step, the mask, baselines and the solver)
    # ------------------------------------------------------------------ #
    def charge_target(self, station: int, level: int, leg: int) -> float:
        """Battery the vehicle leaves `station` with, for charge level `level`
        on leg `leg`: the chosen fraction of capacity, but never less than
        what is needed to reach the next stop and still finish the route."""
        w = self.world
        cap = w.battery_capacity
        target = self.mandatory_sequence[leg]
        enough = (w.energy_for_dist(w.dist(w.station_global_index(station), target))
                  + self.min_battery_on_arrival[leg])
        return min(cap, max(self.charge_levels[level] * cap, enough))

    def simulate(self, node: int, battery: float, time: float, leg: int,
                 action: int) -> Optional[Leg]:
        """Drive leg `leg` from (node, battery, time) using `action`.
        Pure function of its arguments; returns None if the leg can't be
        driven (the battery would run flat on the way)."""
        w = self.world
        target = self.mandatory_sequence[leg]
        station, level = self.decode(int(action))

        if station is None:
            d = w.dist(node, target)
            e = w.energy_for_dist(d)
            if e > battery + EPS:
                return None
            return self._arrive(Leg(None, target, d, d * w.distance_cost_per_unit,
                                    0.0, 0.0, 0.0, time,
                                    battery - e, time + d / w.speed, time + d / w.speed), leg)

        s_glob = w.station_global_index(station)
        d1, d2 = w.dist(node, s_glob), w.dist(s_glob, target)
        e1, e2 = w.energy_for_dist(d1), w.energy_for_dist(d2)
        if e1 > battery + EPS:
            return None
        b_station = battery - e1
        t_station = time + d1 / w.speed
        leave = max(b_station, self.charge_target(station, level, leg))
        if e2 > leave + EPS:
            return None
        energy = leave - b_station
        price = w.price_at(station, t_station)
        t_leave = t_station + w.charge_time(energy)
        d = d1 + d2
        return self._arrive(Leg(station, target, d, d * w.distance_cost_per_unit,
                                energy, price, price * energy, t_station,
                                leave - e2, t_leave + d2 / w.speed,
                                t_leave + d2 / w.speed), leg)

    def _arrive(self, r: Leg, leg: int) -> Leg:
        """Apply the time window of stop `leg` to a leg that reaches it."""
        early, late = self.tw_earliest[leg], self.tw_latest[leg]
        t = max(r.arrival, early)
        late_h = r.arrival - late
        if late_h > EPS:
            pen = self.late_penalty_fixed + self.late_penalty_per_hour * late_h
            return r._replace(time=t, late_hours=late_h, penalty=pen)
        return r._replace(time=t)

    def valid_legs(self, node: int, battery: float, time: float,
                   leg: int) -> List[Tuple[int, Leg]]:
        """Every action that keeps the route finishable, with its outcome.
        Detours that would charge nothing are left out (going direct is
        always at least as good then)."""
        need = self.min_battery_on_arrival[leg]
        out = []
        for a in range(self.action_space.n):
            r = self.simulate(node, battery, time, leg, a)
            if r is None or r.battery < need - EPS:
                continue
            if r.station is not None and r.energy_charged <= EPS:
                continue
            out.append((a, r))
        return out

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #
    def step(self, action: int):
        assert not self.done, "Episode is done — call reset()"
        action = int(action)
        w = self.world
        leg = self.simulate(self.current_node_idx, self.battery, self.time,
                            self.leg_index, action)
        self._mask_cache = None

        if leg is None:
            station, _ = self.decode(action)
            to = (w.station_global_index(station) if station is not None
                  else self.mandatory_sequence[self.leg_index])
            self.done = True
            self.log.append(f"DEAD: {w.all_nodes[self.current_node_idx].name} -> "
                            f"{w.all_nodes[to].name}, battery {self.battery:.1f} kWh "
                            f"is not enough")
            return self._get_obs(), -self._death_penalty(), True, False, {"dead": True}

        frm = w.all_nodes[self.current_node_idx].name
        dst = w.all_nodes[leg.target].name
        if leg.station is not None:
            s_glob = w.station_global_index(leg.station)
            s_name = w.all_nodes[s_glob].name
            d1 = w.dist(self.current_node_idx, s_glob)
            self.log.append(f"{frm} -> {s_name}: {d1:.1f}u, "
                            f"bat={self.battery - w.energy_for_dist(d1):.1f}, "
                            f"t={leg.charge_hour:.1f}h")
            self.log.append(f"  CHARGE {s_name}: +{leg.energy_charged:.1f}kWh "
                            f"@${leg.charge_price:.3f} = ${leg.charge_cost:.2f}")
            self.charges.append((s_name, leg.charge_hour % 24, leg.charge_price,
                                 leg.energy_charged, leg.charge_cost))
            self.path.append(s_glob)
            frm = s_name
            d2 = leg.distance - d1
        else:
            d2 = leg.distance
        note = ""
        if leg.late_hours > 0:
            note = f"  LATE by {leg.late_hours:.1f}h (-${leg.penalty:.2f})"
        elif leg.wait > EPS:
            note = f"  early, waits {leg.wait:.1f}h"
        if np.isfinite(self.tw_latest[self.leg_index]):
            note = (f"  window [{self.tw_earliest[self.leg_index]:.1f}, "
                    f"{self.tw_latest[self.leg_index]:.1f}]" + note)
        self.log.append(f"{frm} -> {dst}: {d2:.1f}u, bat={leg.battery:.1f}, "
                        f"t={leg.arrival:.1f}h{note}")
        self.path.append(leg.target)

        self.battery = leg.battery
        self.time = leg.time
        self.current_node_idx = leg.target
        self.total_distance += leg.distance
        self.total_distance_cost += leg.distance_cost
        self.total_charge_cost += leg.charge_cost
        self.total_energy_charged += leg.energy_charged
        self.total_penalty += leg.penalty
        self.wait_hours += leg.wait
        if leg.late_hours > 0:
            self.late_stops += 1
            self.late_hours += leg.late_hours

        reward = -leg.objective
        info = {}
        self.leg_index += 1
        terminated = self.leg_index >= len(self.mandatory_sequence)
        if terminated:
            self.done = True
            reward += self.completion_bonus
            info.update(self.summary())
        return self._get_obs(), reward, terminated, False, info

    def summary(self) -> dict:
        w = self.world
        avg_paid = (self.total_charge_cost / self.total_energy_charged
                    if self.total_energy_charged > 0 else float("nan"))
        cost = self.total_distance_cost + self.total_charge_cost
        n_windows = sum(1 for l in self.tw_latest if np.isfinite(l))
        return {
            "objective": cost + self.total_penalty,   # what the agent minimises
            "total_cost": cost,
            "lateness_penalty": self.total_penalty,
            "late_stops": self.late_stops,
            "late_hours": self.late_hours,
            "wait_hours": self.wait_hours,
            "windowed_stops": n_windows,
            "total_distance": self.total_distance,
            "total_distance_cost": self.total_distance_cost,
            "total_charge_cost": self.total_charge_cost,
            "energy_charged": self.total_energy_charged,
            "final_battery": self.battery,
            "num_charges": len(self.charges),
            "avg_price_paid": avg_paid,
            "market_avg_price": w.avg_price(),
            "finish_time": self.time,
        }

    def _death_penalty(self) -> float:
        # Worse than the most expensive possible completion of the remaining
        # legs, so the agent can never "prefer" running out of battery.
        w = self.world
        remaining = len(self.mandatory_sequence) - self.leg_index
        return 100.0 + remaining * (w.battery_capacity * w.price_max
                                    + 2 * w.distance_matrix.max() * w.distance_cost_per_unit)

    # ------------------------------------------------------------------ #
    # Feasibility helpers
    # ------------------------------------------------------------------ #
    def _compute_min_battery(self) -> list:
        """need[j] = least battery the vehicle must have on arriving at
        mandatory stop j to be able to finish the route (charging allowed).

        Worked backwards from the depot. Charging to full is the most a
        station can give, so partial charging doesn't change this bound: from
        stop X (next stop Y) the route can be finished with battery b iff
            b >= e(X,Y) + need[Y]                       (go direct), or
            b >= e(X,s) for a station s with
                 capacity - e(s,Y) >= need[Y]           (detour via s).
        """
        w = self.world
        cap = w.battery_capacity
        seq = self.mandatory_sequence
        need = [0.0] * len(seq)
        for j in range(len(seq) - 2, -1, -1):
            x, y = seq[j], seq[j + 1]
            best = w.energy_for_dist(w.dist(x, y)) + need[j + 1]
            for s in w.stations:
                if cap - w.energy_for_dist(w.dist(s.index, y)) >= need[j + 1] - EPS:
                    best = min(best, w.energy_for_dist(w.dist(x, s.index)))
            need[j] = best if best <= cap + EPS else float("inf")
        return need

    def action_masks(self) -> np.ndarray:
        """Valid actions (used by MaskablePPO and the baselines).

        An action is valid only if the vehicle arrives at the next stop with
        enough charge to still finish the whole route. Following the mask
        therefore never runs the battery flat. With time windows, actions
        that arrive on time (see latest_arrival) are kept whenever there are
        any; otherwise lateness can't be avoided and every safe action is kept.
        """
        if self._mask_cache is not None:
            return self._mask_cache.copy()
        n = self.action_space.n
        mask = np.zeros(n, dtype=bool)
        if self.leg_index >= len(self.mandatory_sequence):
            mask[0] = True
            return mask

        args = (self.current_node_idx, self.battery, self.time, self.leg_index)
        legs = self.valid_legs(*args)
        deadline = self.latest_arrival[self.leg_index] + EPS
        on_time = [a for a, r in legs if r.arrival <= deadline]
        for a in (on_time or [a for a, _ in legs]):
            mask[a] = True
        if not mask.any():
            # route already infeasible: allow anything drivable
            for a in range(n):
                mask[a] = self.simulate(*args, a) is not None
        if not mask.any():
            mask[0] = True          # stranded: let the episode end with the penalty
        self._mask_cache = mask
        return mask.copy()

    def station_mask(self, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """(M,) bool — stations with at least one valid charge level."""
        mask = self.action_masks() if mask is None else mask
        return mask[1:].reshape(self.num_stations, self.num_levels).any(axis=1)

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #
    def _get_obs(self) -> np.ndarray:
        w = self.world
        M = w.num_stations
        K = self.price_lookahead
        cap = w.battery_capacity
        max_dist = float(w.distance_matrix.max()) + EPS
        p_span = w.price_max - w.price_min + EPS

        def pnorm(p):
            return (p - w.price_min) / p_span

        seq = self.mandatory_sequence
        has_next = self.leg_index < len(seq)
        next_idx = seq[self.leg_index] if has_next else self.current_node_idx

        battery_soc = self.battery / cap
        progress = self.leg_index / len(seq)
        dist_to_next = w.dist(self.current_node_idx, next_idx)
        time_norm = (self.time % 24.0) / 24.0

        # energy to finish the whole remaining route without detours
        remaining = [self.current_node_idx] + seq[self.leg_index:]
        rem_energy = sum(w.energy_for_dist(w.dist(a, b))
                         for a, b in zip(remaining, remaining[1:]))
        shortfall = max(0.0, rem_energy - self.battery)
        following = (w.dist(next_idx, seq[self.leg_index + 1])
                     if self.leg_index + 1 < len(seq) else 0.0)
        legs_left = len(seq) - self.leg_index

        # time windows: slack to the (tightened) deadline of the next stop if
        # driven direct, and how long the vehicle would wait there
        tw_active, tw_slack, tw_wait = 0.0, 1.0, 0.0
        if has_next:
            tw_active = float(any(np.isfinite(l) for l in self.tw_latest[self.leg_index:]))
            t_direct = self.time + dist_to_next / w.speed
            la = self.latest_arrival[self.leg_index]
            if np.isfinite(la):
                tw_slack = 0.5 + 0.5 * float(np.tanh((la - t_direct) / 12.0))
            tw_wait = _squash(self.tw_earliest[self.leg_index] - t_direct, 6.0)

        current_prices = np.zeros(M, dtype=np.float32)
        arrival_prices = np.zeros(M, dtype=np.float32)
        future_prices = np.zeros(M * K, dtype=np.float32)
        detour_dists = np.zeros(M, dtype=np.float32)
        min_charge = np.zeros(M, dtype=np.float32)
        for s in range(M):
            s_global = w.station_global_index(s)
            d_to_s = w.dist(self.current_node_idx, s_global)
            current_prices[s] = pnorm(w.price_at(s, self.time))
            arrival_prices[s] = pnorm(w.price_at(s, self.time + d_to_s / w.speed))
            for k in range(K):
                future_prices[s * K + k] = pnorm(w.price_at(s, self.time + k + 1))
            if has_next:
                detour = d_to_s + w.dist(s_global, next_idx) - dist_to_next
                detour_dists[s] = detour / max_dist
                # least energy a charge at s must add ("just enough" level)
                b_at_s = self.battery - w.energy_for_dist(d_to_s)
                min_charge[s] = (self.charge_target(s, 0, self.leg_index) - b_at_s) / cap

        reachable = (self.station_mask().astype(np.float32)
                     if has_next else np.zeros(M, dtype=np.float32))

        obs = np.concatenate([
            np.array([
                battery_soc,
                progress,
                dist_to_next / max_dist,
                time_norm,
                _squash(rem_energy, 2.0 * cap),
                _squash(shortfall, cap),
                following / max_dist,
                _squash(legs_left, 10.0),
                tw_active,
                tw_slack,
                tw_wait,
            ], dtype=np.float32),
            current_prices,
            arrival_prices,
            future_prices,
            detour_dists,
            min_charge,
            reachable,
        ])
        return np.clip(obs, 0.0, 1.0).astype(np.float32)
