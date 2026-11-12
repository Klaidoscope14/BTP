"""
Exact solver for small fixed-order routes.

Finds the cheapest possible way to drive a route using the *same* decisions
the RL agent has (go direct, or detour to station k and charge to level l),
so the agent's cost can be compared with the true optimum:

    gap = agent_cost / optimal_cost - 1

Method: depth-first branch and bound over the env's own transition model
(EVRPEnv.simulate), so costs are exactly what the env would charge.

  * Lower bound for a partial route: cost so far
        + distance cost of the remaining route driven directly
          (detours only ever add distance — triangle inequality)
        + energy still to buy (remaining direct energy - battery)
          priced at the cheapest price anywhere on the map.
  * Branches that can't finish the route are cut using the env's exact
    min-battery table (EVRPEnv.valid_legs).
  * Memo on (leg, battery, time): a state reached again at no lower cost is
    skipped. Charge levels that collapse to the same target are merged.

The result is proven optimal unless the time limit / node limit is hit, in
which case the best route found so far is returned with proven=False.
Search effort grows exponentially with route length; routes of up to ~8
customers usually solve in seconds.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import List, Optional

EPS = 1e-9


@dataclass
class SolveResult:
    cost: float                      # objective of the best route found
    actions: List[int] = field(default_factory=list)
    proven: bool = False             # True if the search finished
    nodes: int = 0
    seconds: float = 0.0

    @property
    def feasible(self) -> bool:
        return bool(self.actions) and self.cost < float("inf")


def leg_objective(leg) -> float:
    """What the solver minimises for one leg (cost plus any penalties)."""
    return getattr(leg, "objective", leg.cost)


def solve_optimal(env, time_limit: float = 60.0,
                  max_nodes: Optional[int] = None) -> SolveResult:
    """Optimal plan for the route `env` was last reset to, starting from the
    env's current state. Does not modify the env."""
    w = env.world
    seq = env.mandatory_sequence
    n_legs = len(seq)
    dpc = w.distance_cost_per_unit
    p_min = float(w.price_schedule.min())

    # suffix[j] = direct distance seq[j] -> seq[j+1] -> ... -> depot
    suffix = [0.0] * (n_legs + 1)
    for j in range(n_legs - 2, -1, -1):
        suffix[j] = w.dist(seq[j], seq[j + 1]) + suffix[j + 1]

    def lower_bound(node, battery, leg, cost):
        if leg >= n_legs:
            return cost
        d = w.dist(node, seq[leg]) + suffix[leg]
        energy = w.energy_for_dist(d)
        return cost + d * dpc + max(0.0, energy - battery) * p_min

    best = SolveResult(float("inf"))
    memo = {}
    path: List[int] = []
    start = _time.perf_counter()
    stop = [False]
    nodes = [0]

    def dfs(node, battery, t, leg, cost):
        if stop[0]:
            return
        nodes[0] += 1
        if nodes[0] % 2048 == 0:
            if _time.perf_counter() - start > time_limit or (
                    max_nodes is not None and nodes[0] > max_nodes):
                stop[0] = True
                return
        if leg >= n_legs:
            if cost < best.cost - EPS:
                best.cost = cost
                best.actions = list(path)
            return

        key = (leg, round(battery, 6), round(t, 6))
        seen = memo.get(key)
        if seen is not None and seen <= cost + EPS:
            return
        memo[key] = cost

        children, outcomes = [], set()
        for a, r in env.valid_legs(node, battery, t, leg):
            c = cost + leg_objective(r)
            sig = (round(r.battery, 6), round(r.time, 6), round(c, 9))
            if sig in outcomes:
                continue                     # same outcome as a cheaper-indexed action
            outcomes.add(sig)
            lb = lower_bound(r.target, r.battery, leg + 1, c)
            if lb < best.cost - EPS:
                children.append((lb, a, r, c))
        children.sort(key=lambda x: x[0])
        for lb, a, r, c in children:
            if lb >= best.cost - EPS:
                break
            path.append(a)
            dfs(r.target, r.battery, r.time, leg + 1, c)
            path.pop()

    dfs(env.current_node_idx, env.battery, env.time, env.leg_index, 0.0)
    best.proven = not stop[0]
    best.nodes = nodes[0]
    best.seconds = _time.perf_counter() - start
    return best
