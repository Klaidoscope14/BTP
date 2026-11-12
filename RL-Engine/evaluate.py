"""
Evaluate the trained agent against baseline heuristics.

Run the user's own route (map + fixed visit order from a JSON file):
    python evaluate.py --map examples/sample_map.json
    python evaluate.py --map examples/sample_map.json --order C05,C12,C03,C08
    python evaluate.py --map examples/sample_map.json --plot outputs/

Benchmark over many test visit orders on a known map:
    python evaluate.py --map examples/sample_map.json --random-orders --episodes 50

Benchmark over random test maps (never seen in training):
    python evaluate.py --nc 10 --ns 6 --episodes 50

A trained model is picked up from models/saved_weights/ automatically when it
exists; pass --model / --norm to choose another one.
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.evrp_env import DEFAULT_CHARGE_LEVELS, TEST_SEED_START, EVRPEnv
from utils.world import load_world

DEFAULT_MODEL = "models/saved_weights/ppo_evrp_final.zip"
DEFAULT_NORM = "models/saved_weights/vec_normalize.pkl"


# ────────────────────────────────────────────────────────────────────────────
# Baselines
#
# All baselines respect env.action_masks(), which already forbids moves that
# would leave the vehicle stranded on the following leg. On top of their own
# rule, every baseline charges whenever going direct is not allowed — so they
# never run out of battery and the comparison with the agent is fair.
# ────────────────────────────────────────────────────────────────────────────

def _station_arrival(env: EVRPEnv, s: int):
    w = env.world
    d = w.dist(env.current_node_idx, w.station_global_index(s))
    return d, env.time + d / w.speed


def _pick_station(env: EVRPEnv, mask, key) -> int:
    """Best valid station by `key`, charged as full as the mask allows."""
    st_ok = env.station_mask(mask)
    valid = [s for s in range(env.world.num_stations) if st_ok[s]]
    if not valid:
        return 0
    s = min(valid, key=lambda s: key(env, s))
    for lvl in range(env.full_level, -1, -1):
        if mask[env.encode(s, lvl)]:
            return env.encode(s, lvl)
    return 0


def _nearest(env, s):
    return _station_arrival(env, s)[0]


def _cheapest(env, s):
    _, t = _station_arrival(env, s)
    return env.world.price_at(s, t)


def _detour_plus_charge(env, s):
    w = env.world
    nxt = env.mandatory_sequence[env.leg_index]
    d_to_s, t = _station_arrival(env, s)
    detour = d_to_s + w.dist(w.station_global_index(s), nxt) - w.dist(env.current_node_idx, nxt)
    energy = w.battery_capacity - (env.battery - w.energy_for_dist(d_to_s))
    return max(detour, 0.0) * w.distance_cost_per_unit + w.price_at(s, t) * energy


def _can_finish_without_charging(env: EVRPEnv) -> bool:
    w = env.world
    stops = [env.current_node_idx] + env.mandatory_sequence[env.leg_index:]
    need = sum(w.energy_for_dist(w.dist(a, b)) for a, b in zip(stops, stops[1:]))
    return need <= env.battery + 1e-6


def _first_valid(mask) -> int:
    return int(np.flatnonzero(mask)[0])


def policy_charge_when_forced(env: EVRPEnv) -> int:
    """Go direct; charge at the nearest station only when direct isn't allowed."""
    mask = env.action_masks()
    if mask[0]:
        return 0
    a = _pick_station(env, mask, _nearest)
    return a if mask[a] else _first_valid(mask)


def make_threshold_policy(key, low_battery_frac=0.3):
    def policy(env: EVRPEnv) -> int:
        mask = env.action_masks()
        soc = env.battery / env.world.battery_capacity
        if mask[0] and (soc >= low_battery_frac or _can_finish_without_charging(env)):
            return 0
        a = _pick_station(env, mask, key)
        return a if mask[a] else _first_valid(mask)
    return policy


def policy_smart(env: EVRPEnv, low_battery_frac=0.3) -> int:
    """Threshold rule plus opportunistic charging when a station is cheap."""
    w = env.world
    mask = env.action_masks()
    soc = env.battery / w.battery_capacity
    if mask[0] and _can_finish_without_charging(env):
        return 0
    need = (not mask[0]) or soc < low_battery_frac
    remaining_legs = len(env.mandatory_sequence) - env.leg_index
    st_ok = env.station_mask(mask)
    if not need and remaining_legs > 2 and soc < 0.7:
        cheap = [s for s in range(w.num_stations)
                 if st_ok[s] and _cheapest(env, s) < 0.65 * w.avg_price()]
        need = bool(cheap)
    if not need:
        return 0
    a = _pick_station(env, mask, _detour_plus_charge)
    return a if mask[a] else _first_valid(mask)


BASELINES = {
    "Charge When Forced": policy_charge_when_forced,
    "Greedy Nearest": make_threshold_policy(_nearest),
    "Greedy Cheapest": make_threshold_policy(_cheapest),
    "Smart Rule-Based": policy_smart,
}


def make_agent_policy(model, vec_norm):
    def policy(env: EVRPEnv) -> int:
        obs = env._get_obs()
        if vec_norm is not None:
            obs = vec_norm.normalize_obs(obs)
        action, _ = model.predict(obs, action_masks=env.action_masks(),
                                  deterministic=True)
        return int(action)
    return policy


def make_optimal_policy(time_limit=60.0):
    """Exact branch-and-bound plan (utils/optimal_solver.py), solved once at
    the start of each episode and then replayed leg by leg."""
    from utils.optimal_solver import solve_optimal

    def policy(env: EVRPEnv) -> int:
        if env.leg_index == 0 or getattr(env, "_opt_plan", None) is None:
            res = solve_optimal(env, time_limit=time_limit)
            env._opt_plan, env._opt_result = res.actions, res
        return env._opt_plan[env.leg_index]
    return policy


OPTIMAL = "Optimal (exact)"


def run_episode(env: EVRPEnv, policy):
    env.reset()
    env._opt_plan = None
    info, done = {}, False
    while not done:
        _, _, terminated, truncated, info = env.step(policy(env))
        done = terminated or truncated
    res = getattr(env, "_opt_result", None)
    if res is not None:
        info["proven"] = res.proven
        info["solve_seconds"] = res.seconds
    return info


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def load_agent(model_path, norm_path, probe_env: EVRPEnv):
    if not model_path or not os.path.exists(model_path):
        return None
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(model_path)
    expected = model.observation_space.shape[0]
    n_act = model.action_space.n
    if expected != probe_env.obs_dim or n_act != probe_env.action_space.n:
        sys.exit(
            f"Model {model_path} expects {expected} observation values and "
            f"{n_act} actions, but this map gives {probe_env.obs_dim} and "
            f"{probe_env.action_space.n}. Train with the same number of stations "
            f"({probe_env.num_stations}), --lookahead and --charge-levels, or "
            f"retrain: models from before partial charging are not compatible."
        )
    vec_norm = None
    if norm_path and os.path.exists(norm_path):
        vec_norm = VecNormalize.load(norm_path, DummyVecEnv([lambda: probe_env]))
        vec_norm.training = False
    print(f"Loaded agent from {model_path}"
          + (f" (+ {norm_path})" if vec_norm is not None else ""))
    return make_agent_policy(model, vec_norm)


def env_settings(args) -> dict:
    """EVRPEnv settings the model was trained with (from train_config.json
    next to the model), overridden by anything given on the command line."""
    cfg = {}
    cfg_path = os.path.join(os.path.dirname(args.model or ""), "train_config.json")
    if args.model and os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
    ek = dict(cfg.get("env_kwargs", {}))
    ek.setdefault("price_lookahead", cfg.get("lookahead", 4))
    ek.setdefault("charge_levels", cfg.get("charge_levels", list(DEFAULT_CHARGE_LEVELS)))
    if args.lookahead is not None:
        ek["price_lookahead"] = args.lookahead
    if args.charge_levels is not None:
        ek["charge_levels"] = [float(x) for x in args.charge_levels.split(",")]
    # windows: the map's own, or random test windows with --tw-prob
    if args.no_windows:
        ek["time_windows"] = "off"
    elif args.tw_prob > 0:
        ek.update(time_windows="random", tw_prob=args.tw_prob)
    else:
        ek["time_windows"] = "map"
    return ek


def print_route_detail(env_factory, methods, plot_dir=None):
    env = env_factory()
    w = env.world
    order = " -> ".join(c.name for c in w.ordered_customers)
    print(f"Visit order ({len(w.visit_order)} of {w.num_customers} customers): "
          f"Depot -> {order} -> Depot\n")

    if plot_dir:
        from utils.visualizer import plot_price_heatmap, plot_route
        os.makedirs(plot_dir, exist_ok=True)
        plot_price_heatmap(w, os.path.join(plot_dir, "prices.png"))

    opt_cost = None
    for name, policy in methods.items():
        env = env_factory()
        info = run_episode(env, policy)
        print(f"--- {name} ---")
        if info.get("dead"):
            print("  FAILED (battery died)")
        else:
            print(f"  Total cost: ${info['total_cost']:.2f} "
                  f"(distance ${info['total_distance_cost']:.2f}, "
                  f"charging ${info['total_charge_cost']:.2f} for "
                  f"{info['energy_charged']:.1f} kWh, "
                  f"{info['num_charges']} charge stops, "
                  f"{info['final_battery']:.1f} kWh left at the end)")
            if info["windowed_stops"]:
                print(f"  Time windows: {info['late_stops']}/{info['windowed_stops']} "
                      f"customers late ({info['late_hours']:.1f}h, penalty "
                      f"${info['lateness_penalty']:.2f}), waited {info['wait_hours']:.1f}h"
                      f" -> objective ${info['objective']:.2f}")
            if name == OPTIMAL:
                opt_cost = info["objective"]
                print(f"  {'proven optimal' if info.get('proven') else 'best found (time limit hit)'}"
                      f" in {info.get('solve_seconds', 0):.2f}s")
            elif opt_cost:
                print(f"  Gap to optimal: {100 * (info['objective'] / opt_cost - 1):+.1f}%")
        for line in env.log:
            print(f"    {line}")
        print()
        if plot_dir:
            fname = (name.lower().replace(" ", "_").replace("-", "_")
                     .replace("(", "").replace(")", "") + ".png")
            cost = "died" if info.get("dead") else f"${info['total_cost']:.2f}"
            plot_route(w, env.path, title=f"{name} — {cost}",
                       filename=os.path.join(plot_dir, fname))


def benchmark(env_factory, methods, episodes):
    """Run every method on the same test episodes. When the exact solver is
    included, the 'Gap' column is each method's mean extra cost over the
    optimum, on the episodes the solver proved optimal."""
    results = {n: {"obj": {}, "paid": [], "charges": [], "deaths": 0,
                   "late": 0, "windowed": 0} for n in methods}
    proven = set()
    for ep in range(episodes):
        for name, policy in methods.items():
            info = run_episode(env_factory(ep), policy)
            r = results[name]
            if info.get("dead"):
                r["deaths"] += 1
                continue
            r["obj"][ep] = info["objective"]
            r["late"] += info["late_stops"]
            r["windowed"] += info["windowed_stops"]
            r["charges"].append(info["num_charges"])
            if not math.isnan(info["avg_price_paid"]):
                r["paid"].append(info["avg_price_paid"])
            if name == OPTIMAL and info.get("proven"):
                proven.add(ep)

    opt = results.get(OPTIMAL, {}).get("obj", {})
    tw = any(r["windowed"] for r in results.values())
    print(f"  {'Method':<20} {'Avg Cost':>9} {'Std':>7} {'$/kWh':>7} "
          f"{'Stops':>6} {'Deaths':>7} {'Feasible':>9}"
          + (f" {'Late':>7}" if tw else "") + (f" {'Gap':>8}" if opt else ""))
    print(f"  {'-' * (69 + (8 if tw else 0) + (9 if opt else 0))}")
    for name, r in results.items():
        costs = list(r["obj"].values())
        n_ok = len(costs)
        gap = ""
        if tw:
            gap += (f" {100 * r['late'] / r['windowed']:>6.1f}%" if r["windowed"]
                    else f" {'-':>7}")
        if opt:
            g = [r["obj"][e] / opt[e] - 1 for e in proven if e in r["obj"] and opt[e] > 0]
            gap += f" {100 * np.mean(g):>7.1f}%" if g else f" {'-':>8}"
        if n_ok:
            paid = f"{np.mean(r['paid']):.3f}" if r["paid"] else "  -  "
            print(f"  {name:<20} ${np.mean(costs):>8.2f} ${np.std(costs):>6.2f} "
                  f"{paid:>7} {np.mean(r['charges']):>6.1f} {r['deaths']:>7d} "
                  f"{n_ok:>5}/{episodes}" + gap)
        else:
            print(f"  {name:<20} {'N/A':>9} {'N/A':>7} {'-':>7} {'-':>6} "
                  f"{r['deaths']:>7d} {n_ok:>5}/{episodes}" + gap)
    if tw:
        print("\n  Avg Cost includes lateness penalties; Late = share of "
              "time-windowed customers reached late")
    if opt:
        print(f"\n  Gap = extra cost over the exact optimum "
              f"(on the {len(proven)}/{episodes} episodes solved to proven optimality)")
    return results


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Evaluate EVRP agents")
    p.add_argument("--map", type=str, default=None, help="JSON map with the user's route")
    p.add_argument("--order", type=str, default=None,
                   help="Comma-separated customer names overriding the map's visit_order")
    p.add_argument("--random-orders", action="store_true",
                   help="With --map: benchmark over random test visit orders")
    p.add_argument("--min-visit", type=int, default=None)
    p.add_argument("--max-visit", type=int, default=None)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--norm", type=str, default=DEFAULT_NORM)
    p.add_argument("--nc", type=int, default=10)
    p.add_argument("--ns", type=int, default=6)
    p.add_argument("--nv", type=int, default=None, help="Customers visited per random test map")
    p.add_argument("--grid", type=float, default=40.0)
    p.add_argument("--lookahead", type=int, default=None,
                   help="Default: the value in the model's train_config.json, else 4")
    p.add_argument("--charge-levels", type=str, default=None,
                   help="Default: the value in the model's train_config.json")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--tw-prob", type=float, default=0.0,
                   help="Give this fraction of test routes random time windows "
                        "(replaces the map's own windows)")
    p.add_argument("--no-windows", action="store_true",
                   help="Ignore the map's time windows")
    p.add_argument("--plot", type=str, default=None, help="Directory to save route plots")
    p.add_argument("--optimal", action="store_true",
                   help="Also run the exact solver and report each method's gap to it")
    p.add_argument("--optimal-time", type=float, default=60.0,
                   help="Solver time limit per route, in seconds")
    args = p.parse_args()

    ek = env_settings(args)
    order = [s.strip() for s in args.order.split(",")] if args.order else None
    user_map = load_world(args.map, order) if args.map else None

    if user_map is not None:
        n = user_map.num_customers
        vr = (args.min_visit or max(1, n // 2), args.max_visit or n)

        def env_factory(ep=None):
            if ep is None or not args.random_orders:
                return EVRPEnv(fixed_world=user_map, **ek)
            return EVRPEnv(fixed_world=user_map, randomize_order=True, visit_range=vr,
                           world_seed_pool=[TEST_SEED_START + ep], **ek)
    else:
        def env_factory(ep=0):
            return EVRPEnv(num_customers=args.nc, num_stations=args.ns, num_visit=args.nv,
                           grid_size=args.grid, world_seed_pool=[TEST_SEED_START + ep],
                           **ek)

    methods = dict(BASELINES)
    agent = load_agent(args.model, args.norm, env_factory())
    if agent is not None:
        methods = {"PPO Agent": agent, **methods}
    else:
        print(f"No trained model at {args.model} — running baselines only")
    if args.optimal:
        methods = {OPTIMAL: make_optimal_policy(args.optimal_time), **methods}

    single_route = user_map is not None and not args.random_orders
    print(f"\n{'=' * 72}")
    if single_route:
        print(f"USER ROUTE on {args.map}")
        print(f"{'=' * 72}")
        print_route_detail(env_factory, methods, args.plot)
        return

    what = (f"random visit orders on {args.map}" if user_map is not None
            else f"random test maps ({args.nc} customers, {args.ns} stations)")
    print(f"BENCHMARK: {args.episodes} episodes, {what}")
    print(f"{'=' * 72}")
    benchmark(env_factory, methods, args.episodes)

    print(f"\n{'=' * 72}\nDETAIL OF FIRST TEST EPISODE\n{'=' * 72}")
    print_route_detail(lambda: env_factory(0), methods, args.plot)


if __name__ == "__main__":
    main()
