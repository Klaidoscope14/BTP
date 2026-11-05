"""
Fixed-Order EVRP Gymnasium Environment
=======================================
The customer visit order is fixed (given by the user). The graph — node
positions, distances and every station's 24-hour price schedule — is fully
known. The agent's only decision at each step is whether to go directly to
the next mandatory stop, or detour to one of M charging stations first
(charge to full, then continue).

Action space: Discrete(M + 1)
  0       = go directly to the next mandatory stop
  1 .. M  = detour to station k, charge to full, then go to the next stop

One step = one leg (current position -> next mandatory stop, optionally via a
station). An episode has len(visit_order) + 1 steps: one per customer plus
the return to the depot.

Reward = -(distance cost + charging cost) of the leg, i.e. the episode return
is exactly  completion_bonus - total_cost  when the route is completed, so
maximising reward is the same as minimising the cost reported by evaluate.py.
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

from typing import Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from utils.world import World, generate_world, with_visit_order

# Random-map seeds are split so training never sees evaluation/test maps.
TRAIN_SEED_RANGE = (0, 1_000_000)
EVAL_SEED_START = 2_000_000    # EvalCallback during training
TEST_SEED_START = 3_000_000    # evaluate.py


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

        M = self.num_stations
        K = self.price_lookahead
        self.action_space = spaces.Discrete(M + 1)
        self.obs_dim = 7 + 4 * M + M * K
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )

        self.world: Optional[World] = None
        self.reset(seed=seed)

    # ------------------------------------------------------------------ #
    # World selection
    # ------------------------------------------------------------------ #
    def _sample_num_visit(self, n_total: int, rng=None) -> int:
        if self.visit_range is None:
            return min(self.num_visit, n_total)
        rng = rng if rng is not None else self.np_random
        lo, hi = self.visit_range
        lo, hi = max(1, lo), min(hi, n_total)
        return int(rng.integers(lo, hi + 1))

    def _next_pool_seed(self) -> int:
        s = self.world_seed_pool[self._pool_pos % len(self.world_seed_pool)]
        self._pool_pos += 1
        return s

    def _next_world(self) -> World:
        if self.fixed_world is not None:
            if not self.randomize_order:
                return self.fixed_world
            # a pool makes the sampled orders repeatable (eval/test)
            rng = (np.random.default_rng(self._next_pool_seed())
                   if self.world_seed_pool else self.np_random)
            n = self.fixed_world.num_customers
            k = self._sample_num_visit(n, rng)
            order = rng.permutation(n)[:k]
            return with_visit_order(self.fixed_world, [int(i) for i in order])

        if self.world_seed_pool:
            world_seed = self._next_pool_seed()
        elif self.randomize_world:
            world_seed = int(self.np_random.integers(*TRAIN_SEED_RANGE))
        else:
            if self.world is not None:
                return self.world
            world_seed = self.base_seed

        return generate_world(
            num_customers=self.num_customers,
            num_stations=self.num_stations,
            grid_size=self.grid_size,
            num_visit=self._sample_num_visit(self.num_customers),
            seed=world_seed,
        )

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.world = self._next_world()

        w = self.world
        self.current_node_idx = w.depot.index
        self.battery = w.battery_capacity
        self.time = w.start_time
        self.leg_index = 0
        self.total_distance = 0.0
        self.total_charge_cost = 0.0
        self.total_distance_cost = 0.0
        self.total_energy_charged = 0.0
        self.done = False
        self.log = []
        self.path = [w.depot.index]       # every node driven through, for plots
        self.charges = []                 # (station name, hour, price, kWh, $)

        # mandatory stops: [cust_1, ..., cust_N, depot]
        self.mandatory_sequence = (
            [w.customers[i].index for i in w.visit_order] + [w.depot.index]
        )
        self.min_battery_on_arrival = self._compute_min_battery()
        return self._get_obs(), {}

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #
    def step(self, action: int):
        assert not self.done, "Episode is done — call reset()"
        action = int(action)
        w = self.world
        next_mandatory = self.mandatory_sequence[self.leg_index]

        if action == 0:
            path = [(self.current_node_idx, next_mandatory)]
        else:
            station_global = w.station_global_index(action - 1)
            path = [(self.current_node_idx, station_global),
                    (station_global, next_mandatory)]

        step_cost = 0.0
        for from_idx, to_idx in path:
            d = w.dist(from_idx, to_idx)
            energy = w.energy_for_dist(d)

            if energy > self.battery + 1e-6:
                self.done = True
                self.log.append(
                    f"DEAD: {w.all_nodes[from_idx].name} -> "
                    f"{w.all_nodes[to_idx].name}, need {energy:.1f} kWh, "
                    f"have {self.battery:.1f}"
                )
                reward = -step_cost - self._death_penalty()
                return self._get_obs(), reward, True, False, {"dead": True}

            self.battery -= energy
            self.time += d / w.speed
            self.total_distance += d
            dist_cost = d * w.distance_cost_per_unit
            self.total_distance_cost += dist_cost
            step_cost += dist_cost
            self.path.append(to_idx)
            self.log.append(
                f"{w.all_nodes[from_idx].name} -> {w.all_nodes[to_idx].name}: "
                f"{d:.1f}u, bat={self.battery:.1f}, t={self.time:.1f}h"
            )

            if w.all_nodes[to_idx].kind == "station":
                s_local = w.station_local_index(to_idx)
                price = w.price_at(s_local, self.time)
                energy_charged = w.battery_capacity - self.battery
                cost = price * energy_charged
                self.charges.append((w.all_nodes[to_idx].name, self.time % 24,
                                     price, energy_charged, cost))
                self.battery = w.battery_capacity
                self.time += w.charge_time(energy_charged)
                self.total_charge_cost += cost
                self.total_energy_charged += energy_charged
                step_cost += cost
                self.log.append(
                    f"  CHARGE {w.all_nodes[to_idx].name}: "
                    f"+{energy_charged:.1f}kWh @${price:.3f} = ${cost:.2f}, "
                    f"t={self.time:.1f}h"
                )

            self.current_node_idx = to_idx

        reward = -step_cost
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
        return {
            "total_cost": self.total_distance_cost + self.total_charge_cost,
            "total_distance": self.total_distance,
            "total_distance_cost": self.total_distance_cost,
            "total_charge_cost": self.total_charge_cost,
            "energy_charged": self.total_energy_charged,
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
    def _arrival_battery(self, node: int, battery: float, target: int,
                         action: int) -> Optional[float]:
        """Battery on arrival at `target` from `node` using `action`, or None
        if the leg can't be driven."""
        w = self.world
        cap = w.battery_capacity
        if action == 0:
            e = w.energy_for_dist(w.dist(node, target))
            return battery - e if e <= battery + 1e-6 else None
        s = w.station_global_index(action - 1)
        e1 = w.energy_for_dist(w.dist(node, s))
        e2 = w.energy_for_dist(w.dist(s, target))
        if e1 > battery + 1e-6 or e2 > cap + 1e-6:
            return None
        return cap - e2

    def _compute_min_battery(self) -> list:
        """need[j] = least battery the vehicle must have on arriving at
        mandatory stop j to be able to finish the route (charging allowed).

        Worked backwards from the depot. Because a station always charges to
        full, from stop X (next stop Y) the route can be finished with
        battery b iff
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
                if cap - w.energy_for_dist(w.dist(s.index, y)) >= need[j + 1] - 1e-6:
                    best = min(best, w.energy_for_dist(w.dist(x, s.index)))
            need[j] = best if best <= cap + 1e-6 else float("inf")
        return need

    def action_masks(self) -> np.ndarray:
        """Valid actions (used by MaskablePPO and the baselines).

        An action is valid only if the vehicle arrives at the next stop with
        enough charge to still finish the whole route. Following the mask
        therefore never runs the battery flat.
        """
        M = self.world.num_stations
        mask = np.zeros(M + 1, dtype=bool)
        if self.leg_index >= len(self.mandatory_sequence):
            mask[0] = True
            return mask

        target = self.mandatory_sequence[self.leg_index]
        need = self.min_battery_on_arrival[self.leg_index]
        drivable = np.zeros(M + 1, dtype=bool)
        for a in range(M + 1):
            b = self._arrival_battery(self.current_node_idx, self.battery, target, a)
            drivable[a] = b is not None
            mask[a] = b is not None and b >= need - 1e-6

        if mask.any():
            return mask
        if drivable.any():          # route already infeasible: allow anything drivable
            return drivable
        mask[0] = True              # stranded: let the episode end with the penalty
        return mask

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #
    def _get_obs(self) -> np.ndarray:
        w = self.world
        M = w.num_stations
        K = self.price_lookahead
        cap = w.battery_capacity
        max_dist = float(w.distance_matrix.max()) + 1e-6
        p_span = w.price_max - w.price_min + 1e-6

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

        current_prices = np.zeros(M, dtype=np.float32)
        arrival_prices = np.zeros(M, dtype=np.float32)
        future_prices = np.zeros(M * K, dtype=np.float32)
        detour_dists = np.zeros(M, dtype=np.float32)
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

        reachable = (self.action_masks()[1:].astype(np.float32)
                     if has_next else np.zeros(M, dtype=np.float32))

        obs = np.concatenate([
            np.array([
                battery_soc,
                progress,
                dist_to_next / max_dist,
                time_norm,
                rem_energy / (3.0 * cap),
                shortfall / cap,
                following / max_dist,
            ], dtype=np.float32),
            current_prices,
            arrival_prices,
            future_prices,
            detour_dists,
            reachable,
        ])
        return np.clip(obs, 0.0, 1.0).astype(np.float32)
