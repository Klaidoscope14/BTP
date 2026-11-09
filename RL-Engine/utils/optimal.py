"""
Exact solver for small fixed-order routes.

Finds the cheapest possible sequence of actions — over exactly the same
action set, action mask and cost model as the RL agent (envs/route.py) —
by depth-first branch and bound:

  * children are tried cheapest-first, so a good route is found early;
  * a branch is cut when   cost so far + lower bound on the rest   is no
    better than the best complete route found so far. The lower bound adds
      - distance cost of driving the rest of the route directly,
      - energy shortfall x the cheapest price anywhere at any hour,
      - the cheapest single detour, if any charging is still needed,
      - lateness if the rest were driven directly (detours only delay);
    each term can only underestimate, so no optimal route is ever cut;
  * identical states (same leg, battery and clock) reached at a higher cost
    are cut too.

The result is the true optimum of the discrete problem the agent faces, so
(agent cost - optimal cost) / optimal cost is the agent's optimality gap.
The search is exponential in the worst case; it is meant for routes of up
to ~8 customers. A node or time limit can be set, and the result says
whether optimality was proven.

Usage:
    from utils.optimal import solve
    res = solve(env.ctx)            # or solve(build_context(world, levels))
    res.cost, res.actions, res.optimal
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional

from envs import route as R


@dataclass
class SolveResult:
    cost: float
    actions: List[int] = field(default_factory=list)
    optimal: bool = True            # False if a limit stopped the search early
    nodes: int = 0
    seconds: float = 0.0


def solve(ctx: R.RouteContext, upper_bound: float = math.inf,
          node_limit: Optional[int] = None, time_limit: Optional[float] = None,
          ) -> SolveResult:
    w = ctx.world
    n = ctx.n_legs
    dcost = w.distance_cost_per_unit
    min_price = float(w.price_schedule.min())
    station_idx = [w.station_global_index(k) for k in range(w.num_stations)]

    # distance from stop j to the depot along the fixed order
    d_direct = [0.0] * n
    for j in range(n - 2, -1, -1):
        d_direct[j] = d_direct[j + 1] + w.dist(ctx.seq[j], ctx.seq[j + 1])

    # cheapest detour (extra distance) on leg j, and its suffix minimum
    def leg_detour(frm: int, to: int) -> float:
        base = w.dist(frm, to)
        return max(0.0, min(w.dist(frm, s) + w.dist(s, to) for s in station_idx) - base)

    det = [leg_detour(ctx.seq[j - 1], ctx.seq[j]) for j in range(1, n)]
    det_suffix = [math.inf] * (n + 1)     # det_suffix[j] = min detour on legs > j
    for j in range(n - 1, 0, -1):
        det_suffix[j - 1] = min(det[j - 1], det_suffix[j])

    def lower_bound(s: R.State) -> float:
        to_next = w.dist(s.node, ctx.seq[s.leg])
        dist = to_next + d_direct[s.leg]
        short = w.energy_for_dist(dist) - s.battery
        lb = dist * dcost
        if short > 1e-6:
            cur = leg_detour(s.node, ctx.seq[s.leg])
            lb += short * min_price + min(cur, det_suffix[s.leg]) * dcost
        if ctx.has_windows:
            lb += R.direct_lateness(ctx, s)
        return lb

    best_cost = upper_bound
    best_actions: List[int] = []
    seen = {}
    nodes = 0
    stopped = False
    t0 = time.time()
    path: List[int] = []

    def dfs(s: R.State, cost: float) -> None:
        nonlocal best_cost, best_actions, nodes, stopped
        if stopped:
            return
        nodes += 1
        if (node_limit and nodes > node_limit) or (
                time_limit and nodes % 512 == 0 and time.time() - t0 > time_limit):
            stopped = True
            return
        if s.leg >= n:
            if cost < best_cost - 1e-9:
                best_cost, best_actions = cost, list(path)
            return
        key = (s.leg, round(s.battery, 6), round(s.time, 6))
        if seen.get(key, math.inf) <= cost + 1e-9:
            return
        seen[key] = cost

        mask = R.action_mask(ctx, s)
        children = []
        for a in range(ctx.n_actions):
            if not mask[a]:
                continue
            o = R.simulate(ctx, s, a)
            if not o.drivable:
                continue
            ns = o.next_state(s, ctx)
            c = cost + o.cost
            bound = c + (lower_bound(ns) if ns.leg < n else 0.0)
            if bound < best_cost - 1e-9:
                children.append((bound, a, ns, c))
        children.sort(key=lambda x: x[0])
        for bound, a, ns, c in children:
            if bound >= best_cost - 1e-9:
                break
            path.append(a)
            dfs(ns, c)
            path.pop()

    dfs(R.initial_state(ctx), 0.0)
    return SolveResult(cost=best_cost, actions=best_actions, optimal=not stopped,
                       nodes=nodes, seconds=time.time() - t0)


def brute_force(ctx: R.RouteContext) -> float:
    """Plain enumeration of every masked action sequence (no pruning).
    Only for testing solve() on tiny routes."""
    best = math.inf

    def rec(s: R.State, cost: float) -> None:
        nonlocal best
        if s.leg >= ctx.n_legs:
            best = min(best, cost)
            return
        mask = R.action_mask(ctx, s)
        for a in range(ctx.n_actions):
            if mask[a]:
                o = R.simulate(ctx, s, a)
                if o.drivable:
                    rec(o.next_state(s, ctx), cost + o.cost)

    rec(R.initial_state(ctx), 0.0)
    return best
