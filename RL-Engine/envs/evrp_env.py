"""
Fixed-Order EVRP Gymnasium Environment
=======================================
The customer visit order is fixed (given by the user). The graph — node
positions, distances and every station's 24-hour price schedule — is fully
known. At each leg the agent decides whether to drive straight to the next
mandatory stop, or to detour through one of M charging stations and how
much to charge there (partial charging, see envs/route.py).

Action space: Discrete(1 + M * L)
  0                 = go directly to the next mandatory stop
  1 + k * L + l     = detour via station k, charge to level l (0 = least
                      needed to stay feasible, L-1 = full / enough to finish)
  With charge_levels=0 (legacy) L = 1 and every charge fills to 100 %.

One step = one leg. An episode has len(visit_order) + 1 steps.

Reward = -(distance cost + charging cost + lateness cost) of the leg, plus
completion_bonus at the end, so the episode return is exactly
completion_bonus - total_cost. Running out of battery ends the episode with
a penalty larger than the cost of any feasible completion.

Time windows (time_windows=...):
  "none"       ignore any windows (default)
  "map"        use the windows stored in the map (JSON "window" fields)
  "generated"  every episode draws fresh windows the current visit order can
               meet (see route.generate_time_windows) — used for training

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

import math
from typing import Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs import route as R
from utils.world import World, generate_world, with_time_windows, with_visit_order

# Random-map seeds are split so training never sees evaluation/test maps.
TRAIN_SEED_RANGE = (0, 1_000_000)
EVAL_SEED_START = 2_000_000    # evaluation during training
TEST_SEED_START = 3_000_000    # evaluate.py

N_SCALARS = 11
N_PER_STATION = 7              # + price_lookahead


def obs_size(num_stations: int, price_lookahead: int) -> int:
    return N_SCALARS + num_stations * (N_PER_STATION + price_lookahead)


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
        charge_levels: int = 5,
        time_windows: str = "none",
        tw_slack: Tuple[float, float] = (2.0, 24.0),
        tw_width: float = 24.0,
    ):
        super().__init__()
        if time_windows not in ("none", "map", "generated"):
            raise ValueError(f"time_windows must be none/map/generated, got {time_windows!r}")

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
        self.charge_levels = int(charge_levels)
        self.time_windows = time_windows
        self.tw_slack = tuple(tw_slack)
        self.tw_width = tw_width

        M, K = self.num_stations, self.price_lookahead
        self.n_levels = max(1, self.charge_levels)
        self.action_space = spaces.Discrete(1 + M * self.n_levels)
        self.obs_dim = obs_size(M, K)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )

        self.world: Optional[World] = None
        self.reset(seed=seed)

    # ------------------------------------------------------------------ #
    # World selection
    # ------------------------------------------------------------------ #
    def set_visit_range(self, lo: int, hi: int) -> None:
        """Change the route-length range sampled from the next reset on
        (used by curriculum training)."""
        self.visit_range = (int(lo), int(hi))

    def _sample_num_visit(self, n_total: int, rng=None) -> int:
        if self.visit_range is None:
            return min(self.num_visit, n_total)
        rng = rng if rng is not None else self.np_random
        lo, hi = self.visit_range
        lo, hi = max(1, min(lo, n_total)), min(hi, n_total)
        return int(rng.integers(lo, hi + 1))

    def _next_pool_seed(self) -> int:
        s = self.world_seed_pool[self._pool_pos % len(self.world_seed_pool)]
        self._pool_pos += 1
        return s

    def _next_world(self) -> Tuple[World, np.random.Generator]:
        """Returns the world and the rng used for anything else random in this
        episode (time windows), so pooled episodes are fully repeatable."""
        if self.fixed_world is not None:
            if not self.randomize_order:
                return self.fixed_world, self.np_random
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
            if self.world is not None:
                return self.world, self.np_random
            world_seed = self.base_seed
            rng = np.random.default_rng(world_seed)

        return generate_world(
            num_customers=self.num_customers,
            num_stations=self.num_stations,
            grid_size=self.grid_size,
            num_visit=self._sample_num_visit(self.num_customers, rng),
            seed=world_seed,
        ), rng

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        world, rng = self._next_world()
        if self.time_windows == "none":
            world = with_time_windows(world, None)
        elif self.time_windows == "generated":
            world = with_time_windows(world, R.generate_time_windows(
                world, rng, self.tw_slack, self.tw_width, self.charge_levels))
        self.world = world

        self.ctx = R.build_context(world, self.charge_levels)
        self.state = R.initial_state(self.ctx)
        self.total_distance = 0.0
        self.total_charge_cost = 0.0
        self.total_distance_cost = 0.0
        self.total_energy_charged = 0.0
        self.total_late_cost = 0.0
        self.total_late_hours = 0.0
        self.num_late = 0
        self.total_wait = 0.0
        self.done = False
        self.log = []
        self.path = [world.depot.index]   # every node driven through, for plots
        self.charges = []                 # (station name, hour, price, kWh, $)
        self.actions = []
        self._mask_cache = None
        return self._get_obs(), {}

    # Backwards-compatible views of the state ---------------------------
    @property
    def current_node_idx(self) -> int:
        return self.state.node

    @property
    def battery(self) -> float:
        return self.state.battery

    @property
    def time(self) -> float:
        return self.state.time

    @property
    def leg_index(self) -> int:
        return self.state.leg

    @property
    def mandatory_sequence(self):
        return self.ctx.seq

    @property
    def min_battery_on_arrival(self):
        return self.ctx.need

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #
    def step(self, action: int):
        assert not self.done, "Episode is done — call reset()"
        action = int(action)
        w = self.world
        o = R.simulate(self.ctx, self.state, action)
        self.actions.append(action)
        self._mask_cache = None

        for frm, to, d, bat, t in o.segments:
            self.path.append(to)
            self.log.append(f"{w.all_nodes[frm].name} -> {w.all_nodes[to].name}: "
                            f"{d:.1f}u, bat={bat:.1f}, t={t:.1f}h")
            if o.charge is not None and w.all_nodes[to].kind == "station":
                name, hour, price, kwh, cost = o.charge
                self.log.append(f"  CHARGE {name}: +{kwh:.1f}kWh @${price:.3f} = "
                                f"${cost:.2f} (to {o.target:.1f} kWh), "
                                f"t={t + o.charge_time:.1f}h")
        self.total_distance += o.distance
        self.total_distance_cost += o.dist_cost

        if not o.drivable:
            frm, to, need, have = o.dead_at
            self._record_charge(o)
            self.log.append(f"DEAD: {w.all_nodes[frm].name} -> {w.all_nodes[to].name}, "
                            f"need {need:.1f} kWh, have {have:.1f}")
            self.done = True
            reward = -(o.dist_cost + o.charge_cost) - self._death_penalty()
            return self._get_obs(), reward, True, False, {"dead": True}

        self._record_charge(o)
        if o.wait > 0:
            self.total_wait += o.wait
            self.log.append(f"  WAIT {o.wait:.1f}h for the window to open")
        if o.late_hours > 0:
            self.num_late += 1
            self.total_late_hours += o.late_hours
            self.total_late_cost += o.late_cost
            self.log.append(f"  LATE by {o.late_hours:.1f}h (+${o.late_cost:.2f})")

        self.state = o.next_state(self.state, self.ctx)
        reward = -o.cost
        info = {}
        terminated = self.state.leg >= self.ctx.n_legs
        if terminated:
            self.done = True
            reward += self.completion_bonus
            info.update(self.summary())
        return self._get_obs(), reward, terminated, False, info

    def _record_charge(self, o: R.Outcome) -> None:
        if o.charge is not None:
            self.charges.append(o.charge)
            self.total_charge_cost += o.charge_cost
            self.total_energy_charged += o.charge[3]

    def summary(self) -> dict:
        w = self.world
        avg_paid = (self.total_charge_cost / self.total_energy_charged
                    if self.total_energy_charged > 0 else float("nan"))
        # Energy bought but still in the battery at the end: the vehicle
        # starts full for free, so only charge above that counts as wasted.
        unused = max(0.0, min(self.state.battery, self.total_energy_charged))
        return {
            "total_cost": (self.total_distance_cost + self.total_charge_cost
                           + self.total_late_cost),
            "total_distance": self.total_distance,
            "total_distance_cost": self.total_distance_cost,
            "total_charge_cost": self.total_charge_cost,
            "total_late_cost": self.total_late_cost,
            "late_customers": self.num_late,
            "late_hours": self.total_late_hours,
            "wait_hours": self.total_wait,
            "energy_charged": self.total_energy_charged,
            "energy_unused": unused,
            "num_charges": len(self.charges),
            "avg_price_paid": avg_paid,
            "market_avg_price": w.avg_price(),
            "finish_time": self.state.time,
        }

    def _death_penalty(self) -> float:
        # Worse than the most expensive possible completion of the remaining
        # legs, so the agent can never "prefer" running out of battery.
        w = self.world
        remaining = self.ctx.n_legs - self.state.leg
        late = (w.late_penalty_fixed + w.late_penalty_per_hour * 1000.0
                if self.ctx.has_windows else 0.0)
        return 100.0 + remaining * (w.battery_capacity * w.price_max + late
                                    + 2 * w.distance_matrix.max() * w.distance_cost_per_unit)

    # ------------------------------------------------------------------ #
    # Masks
    # ------------------------------------------------------------------ #
    def action_masks(self) -> np.ndarray:
        """Valid actions (used by MaskablePPO, the baselines and the solver).
        See route.action_mask."""
        if self._mask_cache is None:
            self._mask_cache = R.action_mask(self.ctx, self.state)
        return self._mask_cache.copy()

    def simulate(self, action: int) -> R.Outcome:
        """Preview an action without taking it."""
        return R.simulate(self.ctx, self.state, int(action))

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #
    def _get_obs(self) -> np.ndarray:
        """Every value in [0, 1]; the size depends only on M and K, so one
        model handles visit orders of any length.

        scalars (11): battery SoC, progress, distance to next stop,
          time of day, energy to finish (direct), shortfall, length of the
          following leg, legs remaining, battery needed on arrival at the
          next stop, time slack at the next stop (going direct, against
          the deadline table), windows-active flag
        per station (M each): price now, price on arrival, K future hourly
          prices, detour distance, any valid action, least kWh to buy,
          most useful kWh to buy, time slack if detouring via the station
        """
        w, ctx, s = self.world, self.ctx, self.state
        M, K = w.num_stations, self.price_lookahead
        cap = w.battery_capacity
        max_dist = float(w.distance_matrix.max()) + 1e-6
        p_span = w.price_max - w.price_min + 1e-6

        def pnorm(p):
            return (p - w.price_min) / p_span

        def slack_norm(sl):          # hours of slack -> [0, 1]; 0.5 = on time
            if not math.isfinite(sl):
                return 1.0 if sl > 0 else 0.0
            return 0.5 + 0.5 * math.tanh(sl / 24.0)

        has_next = s.leg < ctx.n_legs
        nxt = ctx.seq[s.leg] if has_next else s.node
        dist_next = w.dist(s.node, nxt)
        rem = (w.energy_for_dist(dist_next) + ctx.e_direct[s.leg]) if has_next else 0.0
        shortfall = max(0.0, rem - s.battery)
        following = (w.dist(nxt, ctx.seq[s.leg + 1])
                     if s.leg + 1 < ctx.n_legs else 0.0)
        legs_left = ctx.n_legs - s.leg
        need_next = ctx.need[s.leg] if has_next else 0.0
        b_direct = s.battery - w.energy_for_dist(dist_next)
        slack_direct = (R.latest_arrival(ctx, s.leg, b_direct)
                        - (s.time + dist_next / w.speed)) if has_next else math.inf

        scalars = [
            s.battery / cap,
            s.leg / ctx.n_legs,
            dist_next / max_dist,
            (s.time % 24.0) / 24.0,
            rem / (rem + cap),
            min(shortfall, 3 * cap) / (3 * cap),
            following / max_dist,
            legs_left / (legs_left + 5.0),
            min(need_next, cap) / cap,
            slack_norm(slack_direct) if ctx.has_windows else 1.0,
            1.0 if ctx.has_windows else 0.0,
        ]

        mask = self.action_masks() if has_next else np.zeros(1 + M * self.n_levels, bool)
        per = np.zeros((M, N_PER_STATION + K), dtype=np.float32)
        for k in range(M):
            g = w.station_global_index(k)
            d_to = w.dist(s.node, g)
            t_arr = s.time + d_to / w.speed
            per[k, 0] = pnorm(w.price_at(k, s.time))
            per[k, 1] = pnorm(w.price_at(k, t_arr))
            for h in range(K):
                per[k, 2 + h] = pnorm(w.price_at(k, s.time + h + 1))
            if not has_next:
                continue
            base = 2 + K
            per[k, base] = (d_to + w.dist(g, nxt) - dist_next) / max_dist
            lv = slice(1 + k * self.n_levels, 1 + (k + 1) * self.n_levels)
            per[k, base + 1] = float(mask[lv].any())
            b_arr = s.battery - w.energy_for_dist(d_to)
            lo, hi = R.charge_bounds(ctx, s.leg, k)
            if self.charge_levels == 0:
                lo = hi = cap
            per[k, base + 2] = max(0.0, lo - b_arr) / cap
            per[k, base + 3] = max(0.0, hi - b_arr) / cap
            if ctx.has_windows:
                # slack after the top-level charge at this station
                o = R.simulate(ctx, s, R.encode(ctx, k, self.n_levels - 1))
                per[k, base + 4] = (slack_norm(R.latest_arrival(ctx, s.leg, o.arrive_battery)
                                               - o.arrive_time) if o.drivable else 0.0)
            else:
                per[k, base + 4] = 1.0

        obs = np.concatenate([np.asarray(scalars, dtype=np.float32), per.ravel()])
        return np.clip(obs, 0.0, 1.0).astype(np.float32)
