"""
Does longer training keep lowering the cost? Evaluate every saved checkpoint
(ppo_evrp_<N>_steps.zip + its VecNormalize stats) on one fixed set of test
routes and report the mean cost and the gap to the exact optimum.

    python learning_curve.py                         # every 200k steps
    python learning_curve.py --every 100000 --episodes 50 --plot outputs/learning_curve.png

Test routes are random visit orders on --map (seeds disjoint from training),
evaluated twice: without time windows and with random time windows. The
optimum of each route is solved once (utils/optimal_solver.py), so the
routes must stay small enough to solve (--max-visit, default 12).
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from envs.evrp_env import TEST_SEED_START, EVRPEnv
from evaluate import BASELINES, load_agent, make_optimal_policy, run_episode
from utils.world import load_world


def checkpoints(directory, every):
    found = {}
    for f in os.listdir(directory):
        m = re.fullmatch(r"ppo_evrp_(\d+)_steps\.zip", f)
        if m:
            found[int(m.group(1))] = f
    steps = sorted(found)
    if every:
        keep, nxt = [], every
        for s in steps:
            if s >= nxt:
                keep.append(s)
                nxt = (s // every + 1) * every
        if steps and steps[-1] not in keep:
            keep.append(steps[-1])
        steps = keep
    out = []
    for s in steps:
        norm = os.path.join(directory, f"ppo_evrp_vecnormalize_{s}_steps.pkl")
        out.append((s, os.path.join(directory, found[s]),
                    norm if os.path.exists(norm) else None))
    return out


def main():
    p = argparse.ArgumentParser(description="Cost vs training steps")
    p.add_argument("--dir", default="models/saved_weights")
    p.add_argument("--map", default="examples/sample_map.json")
    p.add_argument("--min-visit", type=int, default=4)
    p.add_argument("--max-visit", type=int, default=12)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--every", type=int, default=200_000,
                   help="Evaluate one checkpoint per this many steps (0 = all)")
    p.add_argument("--plot", default=None, help="Save a chart to this .png path")
    p.add_argument("--json", default=None, help="Save the numbers to this .json path")
    args = p.parse_args()

    with open(os.path.join(args.dir, "train_config.json")) as f:
        cfg = json.load(f)
    ek = cfg.get("env_kwargs", {"price_lookahead": cfg.get("lookahead", 4)})
    world = load_world(args.map)
    vr = (args.min_visit, args.max_visit)

    settings = {"no windows": {"time_windows": "off"},
                "time windows": {"time_windows": "random"}}

    def factory(ep, tw):
        return EVRPEnv(fixed_world=world, randomize_order=True, visit_range=vr,
                       world_seed_pool=[TEST_SEED_START + ep], **ek, **settings[tw])

    def evaluate(policy, tw):
        return [run_episode(factory(ep, tw), policy)["objective"]
                for ep in range(args.episodes)]

    print(f"Test set: {args.episodes} routes of {vr[0]}-{vr[1]} customers on {args.map}")
    print("Solving every test route exactly...")
    opt = {tw: evaluate(make_optimal_policy(120), tw) for tw in settings}
    smart = {tw: evaluate(BASELINES["Smart Rule-Based"], tw) for tw in settings}

    def gap(costs, tw):
        return 100 * float(np.mean([c / o - 1 for c, o in zip(costs, opt[tw]) if o > 0]))

    rows = []
    for steps, model_path, norm_path in checkpoints(args.dir, args.every):
        policy = load_agent(model_path, norm_path, factory(0, "no windows"))
        row = {"steps": steps}
        for tw in settings:
            costs = evaluate(policy, tw)
            row[tw] = {"cost": float(np.mean(costs)), "gap": gap(costs, tw)}
        rows.append(row)

    print(f"\n  {'Steps':>9} | {'cost (no TW)':>12} {'gap':>7} | {'cost (TW)':>10} {'gap':>7}")
    print(f"  {'-' * 54}")
    for tw_label, ref in (("Optimal", opt), ("Smart Rule-Based", smart)):
        print(f"  {tw_label:>9.9} | ${np.mean(ref['no windows']):>11.2f} "
              f"{gap(ref['no windows'], 'no windows'):>6.1f}% | "
              f"${np.mean(ref['time windows']):>9.2f} "
              f"{gap(ref['time windows'], 'time windows'):>6.1f}%")
    for r in rows:
        a, b = r["no windows"], r["time windows"]
        print(f"  {r['steps']:>9,} | ${a['cost']:>11.2f} {a['gap']:>6.1f}% | "
              f"${b['cost']:>9.2f} {b['gap']:>6.1f}%")
    print("\n  gap = mean extra cost over the exact optimum of each test route")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"rows": rows,
                       "optimal": {k: float(np.mean(v)) for k, v in opt.items()},
                       "smart": {k: float(np.mean(v)) for k, v in smart.items()}},
                      f, indent=2)
    if args.plot and rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = [r["steps"] for r in rows]
        for tw, color in (("no windows", "tab:blue"), ("time windows", "tab:orange")):
            ax.plot(x, [r[tw]["gap"] for r in rows], marker="o", color=color,
                    label=f"PPO agent ({tw})")
            ax.axhline(gap(smart[tw], tw), color=color, linestyle=":",
                       label=f"Smart rule-based ({tw})")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("training steps")
        ax.set_ylabel("extra cost over optimum (%)")
        ax.set_title("Gap to the exact optimum vs training steps")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(args.plot)), exist_ok=True)
        fig.savefig(args.plot, dpi=150)
        print(f"Saved {args.plot}")


if __name__ == "__main__":
    main()
