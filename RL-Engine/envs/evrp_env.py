"""
Fixed-Order EVRP Gymnasium Environment
=======================================
The customer visit order is fixed. The agent's only decision at each step is
whether to go directly to the next customer, or detour to one of M charging
stations first (charge to full, then continue).

Action space: Discrete(M + 1)
  0       = go directly to the next mandatory stop
  1 .. M  = detour to station k, charge, then go to next mandatory stop

One step = one leg of the journey (current position → next mandatory stop,
optionally via a station).  The episode has len(visit_order) + 1 steps
(N customer legs + 1 return-to-depot leg).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from utils.world import World, generate_world


class EVRPEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_customers: int = 10,
        num_stations: int = 6,
        num_visit: int | None = None,
        grid_size: float = 40.0,
        seed: int | None = None,
        price_lookahead: int = 4,
        randomize_world: bool = True,
        fixed_world: World | None = None,
    ):
        super().__init__()

        self.num_customers = num_customers
        self.num_stations = num_stations
        self.num_visit = num_visit or num_customers
        self.grid_size = grid_size
        self.base_seed = seed
        self.price_lookahead = price_lookahead
        self.randomize_world = randomize_world
        self.fixed_world = fixed_world

        M = self.num_stations
        K = self.price_lookahead

        # action: 0 = direct, 1..M = detour via station k
        self.action_space = spaces.Discrete(M + 1)

        # observation: flat vector
        # [battery_soc, progress, dist_to_next_norm, time_of_day_norm,
        #  current_prices(M), future_prices(M*K),
        #  station_detour_dists_norm(M), station_reachable(M)]
        obs_dim = 4 + M + M * K + M + M
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self.world: World | None = None
        self._episode_seed = seed or 0
        self.reset()

    # --------------------------------------------------------------------- #
    # Reset
    # --------------------------------------------------------------------- #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.fixed_world is not None:
            self.world = self.fixed_world
        elif self.randomize_world:
            self._episode_seed = (seed if seed is not None
                                  else self._episode_seed + 1)
            self.world = generate_world(
                num_customers=self.num_customers,
                num_stations=self.num_stations,
                grid_size=self.grid_size,
                num_visit=self.num_visit,
                seed=self._episode_seed,
            )
        elif self.world is None:
            self.world = generate_world(
                num_customers=self.num_customers,
                num_stations=self.num_stations,
                grid_size=self.grid_size,
                num_visit=self.num_visit,
                seed=self.base_seed,
            )

        w = self.world
        self.current_node_idx = w.depot.index
        self.battery = w.battery_capacity
        self.time = w.start_time
        self.leg_index = 0  # which leg we're on (0 .. len(visit_order))
        self.total_distance = 0.0
        self.total_charge_cost = 0.0
        self.total_distance_cost = 0.0
        self.done = False
        self.log = []

        # build the full mandatory stop sequence: [cust_0, cust_1, ..., depot]
        self.mandatory_sequence = (
            [w.customers[i].index for i in w.visit_order] + [w.depot.index]
        )

        return self._get_obs(), {}

    # --------------------------------------------------------------------- #
    # Step
    # --------------------------------------------------------------------- #
    def step(self, action: int):
        assert not self.done, "Episode is done — call reset()"
        w = self.world
        reward = 0.0
        info = {}
        truncated = False

        next_mandatory = self.mandatory_sequence[self.leg_index]

        # --- decode action ---
        if action == 0:
            # go directly to next mandatory stop
            path = [(self.current_node_idx, next_mandatory)]
        else:
            station_local = action - 1
            station_global = w.station_global_index(station_local)
            path = [
                (self.current_node_idx, station_global),    # to station
                (station_global, next_mandatory),           # station to customer
            ]

        # --- simulate driving ---
        for from_idx, to_idx in path:
            d = w.dist(from_idx, to_idx)
            energy = w.energy_for_dist(d)

            if energy > self.battery + 1e-6:
                # battery dead
                self.done = True
                reward = -100.0
                self.log.append(
                    f"DEAD: {w.all_nodes[from_idx].name} -> "
                    f"{w.all_nodes[to_idx].name}, need {energy:.1f} kWh, "
                    f"have {self.battery:.1f}"
                )
                return self._get_obs(), reward, True, False, {"dead": True}

            self.battery -= energy
            self.time += d / w.speed
            self.total_distance += d
            self.total_distance_cost += d * w.distance_cost_per_unit

            self.log.append(
                f"{w.all_nodes[from_idx].name} -> {w.all_nodes[to_idx].name}: "
                f"{d:.1f}u, bat={self.battery:.1f}, t={self.time:.1f}h"
            )

            # charge if we arrived at a station
            if w.all_nodes[to_idx].kind == "station":
                station_local = to_idx - 1 - w.num_customers  # offset past depot+customers
                price = w.price_at(station_local, self.time)
                energy_charged = w.battery_capacity - self.battery
                cost = price * energy_charged
                charge_t = w.charge_time(energy_charged)
                self.battery = w.battery_capacity
                self.time += charge_t
                self.total_charge_cost += cost

                # arbitrage reward: bonus if price < average
                avg = w.avg_price()
                if price < avg:
                    reward += (avg - price) * energy_charged * 0.5

                reward -= cost  # penalize charging cost

                self.log.append(
                    f"  CHARGE {w.all_nodes[to_idx].name}: "
                    f"+{energy_charged:.1f}kWh @${price:.3f} = ${cost:.2f}, "
                    f"t={self.time:.1f}h"
                )

            self.current_node_idx = to_idx

        # distance penalty
        leg_dist = sum(w.dist(a, b) for a, b in path)
        reward -= leg_dist * w.distance_cost_per_unit

        # advance to next leg
        self.leg_index += 1
        terminated = self.leg_index >= len(self.mandatory_sequence)

        if terminated:
            self.done = True
            reward += 50.0  # completion bonus
            info["total_cost"] = self.total_distance_cost + self.total_charge_cost
            info["total_distance"] = self.total_distance
            info["total_charge_cost"] = self.total_charge_cost

        return self._get_obs(), reward, terminated, truncated, info

    # --------------------------------------------------------------------- #
    # Observation
    # --------------------------------------------------------------------- #
    def _get_obs(self) -> np.ndarray:
        w = self.world
        M = w.num_stations
        K = self.price_lookahead

        battery_soc = self.battery / w.battery_capacity
        progress = (self.leg_index / len(self.mandatory_sequence)
                    if self.mandatory_sequence else 1.0)

        if self.leg_index < len(self.mandatory_sequence):
            next_idx = self.mandatory_sequence[self.leg_index]
            dist_to_next = w.dist(self.current_node_idx, next_idx)
        else:
            dist_to_next = 0.0
        max_dist = w.distance_matrix.max() + 1e-6
        dist_norm = min(dist_to_next / max_dist, 1.0)

        time_norm = (self.time % 24.0) / 24.0

        # current + future prices, normalized
        current_prices = np.zeros(M, dtype=np.float32)
        future_prices = np.zeros(M * K, dtype=np.float32)
        for s in range(M):
            p = w.price_at(s, self.time)
            current_prices[s] = (p - w.price_min) / (w.price_max - w.price_min + 1e-6)
            for k in range(K):
                pf = w.price_at(s, self.time + k + 1)
                future_prices[s * K + k] = (
                    (pf - w.price_min) / (w.price_max - w.price_min + 1e-6)
                )

        # station detour distances (normalized) and reachability
        detour_dists = np.zeros(M, dtype=np.float32)
        reachable = np.zeros(M, dtype=np.float32)
        if self.leg_index < len(self.mandatory_sequence):
            next_idx = self.mandatory_sequence[self.leg_index]
            direct_dist = w.dist(self.current_node_idx, next_idx)
            for s in range(M):
                s_global = w.station_global_index(s)
                d_to_s = w.dist(self.current_node_idx, s_global)
                d_s_to_next = w.dist(s_global, next_idx)
                detour = (d_to_s + d_s_to_next) - direct_dist
                detour_dists[s] = min(max(detour / max_dist, 0.0), 1.0)
                if w.energy_for_dist(d_to_s) <= self.battery:
                    reachable[s] = 1.0

        obs = np.concatenate([
            np.array([battery_soc, progress, dist_norm, time_norm],
                     dtype=np.float32),
            current_prices,
            future_prices,
            detour_dists,
            reachable,
        ])
        return np.clip(obs, 0.0, 1.0)

    # --------------------------------------------------------------------- #
    # Action mask (for SB3 MaskablePPO)
    # --------------------------------------------------------------------- #
    def action_masks(self) -> np.ndarray:
        w = self.world
        M = w.num_stations
        mask = np.ones(M + 1, dtype=bool)

        # action 0 (direct) is always valid if battery allows
        if self.leg_index < len(self.mandatory_sequence):
            next_idx = self.mandatory_sequence[self.leg_index]
            direct_energy = w.energy_for_dist(
                w.dist(self.current_node_idx, next_idx)
            )
            if direct_energy > self.battery + 1e-6:
                mask[0] = False

            # each station: reachable AND station-to-next is reachable after charge
            for s in range(M):
                s_global = w.station_global_index(s)
                d_to_s = w.dist(self.current_node_idx, s_global)
                d_s_next = w.dist(s_global, next_idx)
                if (w.energy_for_dist(d_to_s) > self.battery + 1e-6 or
                        w.energy_for_dist(d_s_next) > w.battery_capacity + 1e-6):
                    mask[s + 1] = False
        else:
            mask[1:] = False

        # if nothing is valid (stranded), allow direct to trigger death penalty
        if not mask.any():
            mask[0] = True

        return mask
