"""
Evaluate the trained PPO agent against baseline heuristics.

Usage:
    python evaluate.py                              # default model path
    python evaluate.py --model models/saved_weights/ppo_evrp_final.zip
    python evaluate.py --episodes 50 --seed 777
"""

import argparse
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.evrp_env import EVRPEnv
from utils.world import generate_world, World


# ────────────────────────────────────────────────────────────────────────────
# Baselines
# ────────────────────────────────────────────────────────────────────────────

def run_baseline_always_direct(env: EVRPEnv):
    """Never charge voluntarily — only go direct."""
    obs, _ = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        obs, r, terminated, truncated, info = env.step(0)
        total_reward += r
        done = terminated or truncated
    return info, total_reward, env.log


def run_baseline_greedy_nearest(env: EVRPEnv, low_battery_frac=0.3):
    """Charge at nearest reachable station when battery < threshold."""
    obs, _ = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        w = env.world
        soc = env.battery / w.battery_capacity
        mask = env.action_masks()

        if soc < low_battery_frac:
            # pick nearest reachable station
            best_action = 0
            best_dist = float("inf")
            for s in range(w.num_stations):
                if not mask[s + 1]:
                    continue
                s_global = w.station_global_index(s)
                d = w.dist(env.current_node_idx, s_global)
                if d < best_dist:
                    best_dist = d
                    best_action = s + 1
        else:
            best_action = 0

        if not mask[best_action]:
            best_action = 0

        obs, r, terminated, truncated, info = env.step(best_action)
        total_reward += r
        done = terminated or truncated
    return info, total_reward, env.log


def run_baseline_greedy_cheapest(env: EVRPEnv, low_battery_frac=0.3):
    """When battery < threshold, charge at cheapest reachable station."""
    obs, _ = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        w = env.world
        soc = env.battery / w.battery_capacity
        mask = env.action_masks()

        if soc < low_battery_frac:
            best_action = 0
            best_price = float("inf")
            for s in range(w.num_stations):
                if not mask[s + 1]:
                    continue
                p = w.price_at(s, env.time)
                if p < best_price:
                    best_price = p
                    best_action = s + 1
        else:
            best_action = 0

        if not mask[best_action]:
            best_action = 0

        obs, r, terminated, truncated, info = env.step(best_action)
        total_reward += r
        done = terminated or truncated
    return info, total_reward, env.log


def run_baseline_smart(env: EVRPEnv, low_battery_frac=0.3):
    """Score stations by detour_cost + charge_cost; opportunistic cheap charging."""
    obs, _ = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        w = env.world
        soc = env.battery / w.battery_capacity
        mask = env.action_masks()
        avg_p = w.avg_price()

        best_action = 0
        best_score = float("inf")
        need_charge = soc < low_battery_frac

        if env.leg_index < len(env.mandatory_sequence):
            next_idx = env.mandatory_sequence[env.leg_index]
            direct_dist = w.dist(env.current_node_idx, next_idx)

            for s in range(w.num_stations):
                if not mask[s + 1]:
                    continue
                s_global = w.station_global_index(s)
                d_to_s = w.dist(env.current_node_idx, s_global)
                d_s_next = w.dist(s_global, next_idx)
                detour = (d_to_s + d_s_next) - direct_dist
                detour_cost = max(detour, 0.0) * w.distance_cost_per_unit

                t_at_s = env.time + d_to_s / w.speed
                price = w.price_at(s, t_at_s)
                energy = w.battery_capacity - (env.battery - w.energy_for_dist(d_to_s))
                charge_cost = price * energy
                score = detour_cost + charge_cost

                # opportunistic: cheap price + will need charge later
                remaining_legs = len(env.mandatory_sequence) - env.leg_index
                if price < avg_p * 0.65 and remaining_legs > 2 and soc < 0.7:
                    need_charge = True

                if need_charge and score < best_score:
                    best_score = score
                    best_action = s + 1

        if not mask[best_action]:
            best_action = 0

        obs, r, terminated, truncated, info = env.step(best_action)
        total_reward += r
        done = terminated or truncated
    return info, total_reward, env.log


def run_ppo_agent(env: EVRPEnv, model: PPO, vec_norm=None):
    """Run the trained PPO agent."""
    obs, _ = env.reset()
    if vec_norm is not None:
        obs = vec_norm.normalize_obs(obs)
    total_reward = 0.0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        mask = env.action_masks()
        if not mask[int(action)]:
            # fallback: pick best valid action
            valid = np.where(mask)[0]
            action = valid[0] if len(valid) > 0 else 0

        obs, r, terminated, truncated, info = env.step(int(action))
        if vec_norm is not None:
            obs = vec_norm.normalize_obs(obs)
        total_reward += r
        done = terminated or truncated
    return info, total_reward, env.log


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate EVRP agents")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained PPO model .zip")
    parser.add_argument("--norm", type=str, default=None,
                        help="Path to VecNormalize .pkl")
    parser.add_argument("--nc", type=int, default=10)
    parser.add_argument("--ns", type=int, default=6)
    parser.add_argument("--nv", type=int, default=None)
    parser.add_argument("--grid", type=float, default=40.0)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=5000)
    args = parser.parse_args()

    num_visit = args.nv or args.nc

    # load PPO if available
    ppo_model = None
    vec_norm = None
    if args.model and os.path.exists(args.model):
        ppo_model = PPO.load(args.model)
        print(f"Loaded PPO model from {args.model}")
        if args.norm and os.path.exists(args.norm):
            dummy = DummyVecEnv([lambda: EVRPEnv(
                num_customers=args.nc, num_stations=args.ns,
                num_visit=num_visit, grid_size=args.grid, seed=0,
                randomize_world=False)])
            vec_norm = VecNormalize.load(args.norm, dummy)
            vec_norm.training = False
            print(f"Loaded VecNormalize from {args.norm}")
    else:
        print("No PPO model provided — running baselines only")

    methods = {
        "Always Direct": run_baseline_always_direct,
        "Greedy Nearest": run_baseline_greedy_nearest,
        "Greedy Cheapest": run_baseline_greedy_cheapest,
        "Smart Rule-Based": run_baseline_smart,
    }

    results = {name: {"costs": [], "deaths": 0} for name in methods}
    if ppo_model is not None:
        results["PPO Agent"] = {"costs": [], "deaths": 0}

    for ep in range(args.episodes):
        ep_seed = args.seed + ep

        for name, fn in methods.items():
            env = EVRPEnv(
                num_customers=args.nc, num_stations=args.ns,
                num_visit=num_visit, grid_size=args.grid,
                seed=ep_seed, randomize_world=False,
            )
            info, reward, log = fn(env)
            if info.get("dead"):
                results[name]["deaths"] += 1
            elif "total_cost" in info:
                results[name]["costs"].append(info["total_cost"])

        if ppo_model is not None:
            env = EVRPEnv(
                num_customers=args.nc, num_stations=args.ns,
                num_visit=num_visit, grid_size=args.grid,
                seed=ep_seed, randomize_world=False,
            )
            info, reward, log = run_ppo_agent(env, ppo_model, vec_norm)
            if info.get("dead"):
                results["PPO Agent"]["deaths"] += 1
            elif "total_cost" in info:
                results["PPO Agent"]["costs"].append(info["total_cost"])

    # print results
    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS ({args.episodes} episodes, "
          f"{args.nc} customers, {args.ns} stations, visiting {num_visit})")
    print(f"{'='*70}")
    print(f"  {'Method':<22} {'Avg Cost':>10} {'Std':>8} "
          f"{'Deaths':>8} {'Feasible':>10}")
    print(f"  {'-'*58}")

    for name in ["PPO Agent", "Smart Rule-Based", "Greedy Cheapest",
                  "Greedy Nearest", "Always Direct"]:
        if name not in results:
            continue
        r = results[name]
        n_ok = len(r["costs"])
        if n_ok > 0:
            avg = np.mean(r["costs"])
            std = np.std(r["costs"])
            print(f"  {name:<22} ${avg:>8.2f} ${std:>6.2f} "
                  f"{r['deaths']:>7d}   {n_ok}/{args.episodes}")
        else:
            print(f"  {name:<22} {'N/A':>10} {'N/A':>8} "
                  f"{r['deaths']:>7d}   {n_ok}/{args.episodes}")

    # detailed single-episode comparison
    print(f"\n{'='*70}")
    print("SINGLE EPISODE DETAIL (seed={})".format(args.seed))
    print(f"{'='*70}")
    env = EVRPEnv(
        num_customers=args.nc, num_stations=args.ns,
        num_visit=num_visit, grid_size=args.grid,
        seed=args.seed, randomize_world=False,
    )
    w = env.world
    print(f"Visit order: {' -> '.join(w.all_nodes[w.customers[i].index].name for i in w.visit_order)}")
    print()

    for name, fn in methods.items():
        env = EVRPEnv(
            num_customers=args.nc, num_stations=args.ns,
            num_visit=num_visit, grid_size=args.grid,
            seed=args.seed, randomize_world=False,
        )
        info, reward, log = fn(env)
        print(f"--- {name} ---")
        if info.get("dead"):
            print("  FAILED (battery died)")
        else:
            print(f"  Total cost: ${info.get('total_cost', 0):.2f} "
                  f"(dist=${info.get('total_distance', 0) * w.distance_cost_per_unit:.2f}, "
                  f"charge=${info.get('total_charge_cost', 0):.2f})")
        for line in log:
            print(f"    {line}")
        print()


if __name__ == "__main__":
    main()
