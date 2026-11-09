"""
Train a MaskablePPO agent on the fixed-order EVRP environment.

Two training modes:

  1. Known map (the professor's setting): the graph is fixed and known, and
     every episode the agent gets a different user-style visit order — a
     random subset of the map's customers in a random order.

        python train.py --map examples/sample_map.json --min-visit 4 --max-visit 12

  2. Random maps: a new synthetic map every episode (tests generalisation).

        python train.py --nc 10 --ns 6 --steps 500000

Options:
  --charge-levels L   partial charging with L target levels (default 5);
                      0 = legacy "always charge to 100 %"
  --time-windows generated
                      every episode gets random delivery windows that the
                      route can meet; lateness is penalised
  --curriculum 5,15,30
                      train on routes of up to 5 customers, then 15, then 30,
                      continuing the same network (stage k samples route
                      lengths in [ceil(n_k / 3), n_k]); --steps is split
                      evenly over the stages unless --stage-steps is given
  --steps 2000000     long runs: the learning curve (mean test cost every
                      --eval-freq steps) goes to cost_curve.csv; plot it with
                      learning_curve.py

The agent's observation size depends only on the number of stations, so a
model trained here can be run on any visit order (any length) over a map with
the same number of stations.
"""

import argparse
import csv
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from envs.evrp_env import EVAL_SEED_START, EVRPEnv
from utils.world import load_world


def env_kwargs(args, base_map, visit_range, seed, world_seed_pool=None):
    common = dict(
        visit_range=visit_range,
        price_lookahead=args.lookahead,
        charge_levels=args.charge_levels,
        time_windows=args.time_windows,
        tw_slack=args.tw_slack,
        tw_width=args.tw_width,
        world_seed_pool=world_seed_pool,
        seed=seed,
    )
    if base_map is not None:
        return dict(fixed_world=base_map, randomize_order=True, **common)
    return dict(num_customers=args.nc, num_stations=args.ns, grid_size=args.grid,
                randomize_world=True, **common)


def make_env(kwargs):
    def _init():
        return Monitor(EVRPEnv(**kwargs))
    return _init


class CostEvalCallback(BaseCallback):
    """Every `eval_freq` steps, run the deterministic policy on a fixed set
    of held-out episodes and log the mean route cost (in $, not reward) to
    cost_curve.csv. Keeps the best model of the current stage."""

    def __init__(self, train_env: VecNormalize, eval_kwargs: list, eval_freq: int,
                 save_dir: str, csv_path: str, stage: str, verbose: int = 1):
        super().__init__(verbose)
        self.train_env = train_env
        self.eval_kwargs = eval_kwargs
        self.eval_freq = eval_freq
        self.save_dir = save_dir
        self.csv_path = csv_path
        self.stage = stage
        self.last_eval = None
        self.best = math.inf

    def _on_training_start(self) -> None:
        self.last_eval = self.num_timesteps
        self.t0 = time.time()

    def evaluate(self) -> dict:
        costs, late, unused, deaths = [], [], [], 0
        for kw in self.eval_kwargs:
            env = EVRPEnv(**kw)
            obs, _ = env.reset()
            done, info = False, {}
            while not done:
                a, _ = self.model.predict(self.train_env.normalize_obs(obs),
                                          action_masks=env.action_masks(),
                                          deterministic=True)
                obs, _, done, _, info = env.step(int(a))
            if info.get("dead"):
                deaths += 1
            else:
                costs.append(info["total_cost"])
                late.append(info["late_customers"])
                unused.append(info["energy_unused"])
        return {
            "mean_cost": float(np.mean(costs)) if costs else math.nan,
            "std_cost": float(np.std(costs)) if costs else math.nan,
            "deaths": deaths,
            "mean_late": float(np.mean(late)) if late else math.nan,
            "mean_unused_kwh": float(np.mean(unused)) if unused else math.nan,
        }

    def _log(self) -> None:
        r = self.evaluate()
        new = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            wr = csv.writer(f)
            if new:
                wr.writerow(["timesteps", "stage", "mean_cost", "std_cost", "deaths",
                             "mean_late", "mean_unused_kwh", "wall_seconds"])
            wr.writerow([self.num_timesteps, self.stage, f"{r['mean_cost']:.4f}",
                         f"{r['std_cost']:.4f}", r["deaths"], f"{r['mean_late']:.3f}",
                         f"{r['mean_unused_kwh']:.2f}", f"{time.time() - self.t0:.0f}"])
        score = r["mean_cost"] + 1000.0 * r["deaths"]
        tag = ""
        if score < self.best:
            self.best = score
            self.model.save(os.path.join(self.save_dir, "best_model"))
            self.train_env.save(os.path.join(self.save_dir, "vec_normalize_best.pkl"))
            tag = "  (new best)"
        if self.verbose:
            print(f"[eval] stage {self.stage} step {self.num_timesteps}: "
                  f"mean cost ${r['mean_cost']:.2f} ± {r['std_cost']:.2f}, "
                  f"deaths {r['deaths']}, late {r['mean_late']:.2f}{tag}", flush=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval >= self.eval_freq:
            self.last_eval = self.num_timesteps
            self._log()
        return True

    def _on_training_end(self) -> None:
        self._log()


def stage_range(n: int):
    return (max(1, math.ceil(n / 3)), n)


def main():
    parser = argparse.ArgumentParser(description="Train MaskablePPO on fixed-order EVRP")
    parser.add_argument("--map", type=str, default=None,
                        help="JSON map (fixed known graph). Omit to train on random maps.")
    parser.add_argument("--nc", type=int, default=10, help="Customers per random map")
    parser.add_argument("--ns", type=int, default=6, help="Stations per random map")
    parser.add_argument("--grid", type=float, default=40.0, help="Random map half-size")
    parser.add_argument("--min-visit", type=int, default=None,
                        help="Fewest customers in a sampled visit order (default: half)")
    parser.add_argument("--max-visit", type=int, default=None,
                        help="Most customers in a sampled visit order (default: all)")
    parser.add_argument("--curriculum", type=str, default=None,
                        help="Comma-separated route sizes, e.g. 5,15,30")
    parser.add_argument("--stage-steps", type=str, default=None,
                        help="Comma-separated steps per curriculum stage")
    parser.add_argument("--charge-levels", type=int, default=5,
                        help="Partial-charge target levels per station (0 = always full)")
    parser.add_argument("--time-windows", choices=["none", "generated"], default="none")
    parser.add_argument("--tw-slack", type=str, default="2,24",
                        help="Generated windows: latest = feasible arrival + U(lo, hi) hours")
    parser.add_argument("--tw-width", type=float, default=24.0,
                        help="Generated windows: earliest = feasible arrival - U(0, width)")
    parser.add_argument("--lookahead", type=int, default=4, help="Hours of future prices observed")
    parser.add_argument("--steps", type=int, default=500_000, help="Total training steps")
    parser.add_argument("--envs", type=int, default=4, help="Parallel environments")
    parser.add_argument("--subproc", action="store_true",
                        help="Run the environments in subprocesses")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument("--eval-freq", type=int, default=25_000, help="Steps between evaluations")
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--save-dir", type=str, default="models/saved_weights")
    parser.add_argument("--threads", type=int, default=1,
                        help="Torch CPU threads (1 is fastest for this small network "
                             "and lets several runs share a machine)")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    args.tw_slack = tuple(float(x) for x in args.tw_slack.split(","))

    base_map = load_world(args.map) if args.map else None
    n_stations = base_map.num_stations if base_map else args.ns

    if args.curriculum:
        sizes = [int(x) for x in args.curriculum.split(",")]
        if base_map is None:
            args.nc = max(args.nc, max(sizes))
        elif max(sizes) > base_map.num_customers:
            sys.exit(f"--curriculum goes up to {max(sizes)} customers but {args.map} "
                     f"has only {base_map.num_customers}. Use a bigger map "
                     f"(e.g. examples/large_map.json).")
        stages = [(str(n), stage_range(n)) for n in sizes]
        if args.stage_steps:
            steps = [int(x) for x in args.stage_steps.split(",")]
            if len(steps) != len(stages):
                sys.exit("--stage-steps needs one value per curriculum stage")
        else:
            steps = [args.steps // len(stages)] * len(stages)
    else:
        n_total = base_map.num_customers if base_map else args.nc
        hi = min(args.max_visit or n_total, n_total)
        lo = max(1, min(args.min_visit or max(1, n_total // 2), hi))
        args.min_visit, args.max_visit = lo, hi
        stages = [("all", (lo, hi))]
        steps = [args.steps]
    args.steps = sum(steps)

    os.makedirs(args.save_dir, exist_ok=True)
    csv_path = os.path.join(args.save_dir, "cost_curve.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)

    where = f"map {args.map}" if base_map else f"random maps ({args.nc} customers)"
    print(f"Training MaskablePPO on {where}, {n_stations} stations, "
          f"charge levels={args.charge_levels}, time windows={args.time_windows}, "
          f"{args.steps} steps")
    for (name, (lo, hi)), st in zip(stages, steps):
        print(f"  stage {name}: routes of {lo}-{hi} customers, {st} steps")

    first_range = stages[0][1]
    vec_cls = SubprocVecEnv if args.subproc else DummyVecEnv
    train_env = VecNormalize(
        vec_cls([make_env(env_kwargs(args, base_map, first_range, args.seed + i))
                 for i in range(args.envs)]),
        norm_obs=True, norm_reward=True, clip_obs=10.0,
    )

    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        seed=args.seed,
        device="cpu",
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.envs, 1),
        save_path=args.save_dir,
        name_prefix="ppo_evrp",
        save_vecnormalize=True,
    )

    for i, ((name, vr), st) in enumerate(zip(stages, steps)):
        if i > 0:
            # running episodes finish on the old range; new ones use the new range
            train_env.env_method("set_visit_range", *vr)
        # fixed held-out episodes for this stage (disjoint from training/test seeds)
        eval_kwargs = [env_kwargs(args, base_map, vr, seed=0,
                                  world_seed_pool=[EVAL_SEED_START + k])
                       for k in range(args.eval_episodes)]
        eval_cb = CostEvalCallback(train_env, eval_kwargs, args.eval_freq,
                                   args.save_dir, csv_path, name)
        print(f"\n=== Stage {name}: routes of {vr[0]}-{vr[1]} customers, {st} steps ===",
              flush=True)
        model.learn(total_timesteps=st, callback=[checkpoint_cb, eval_cb],
                    reset_num_timesteps=(i == 0), progress_bar=False)
        if len(stages) > 1:
            model.save(os.path.join(args.save_dir, f"stage_{name}_final"))
            train_env.save(os.path.join(args.save_dir, f"stage_{name}_vec_normalize.pkl"))

    final_path = os.path.join(args.save_dir, "ppo_evrp_final")
    model.save(final_path)
    train_env.save(os.path.join(args.save_dir, "vec_normalize.pkl"))
    cfg = {**vars(args), "num_stations": n_stations,
           "stages": [{"name": n, "visit_range": list(vr), "steps": s}
                      for (n, vr), s in zip(stages, steps)]}
    with open(os.path.join(args.save_dir, "train_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\nTraining complete. Model saved to {final_path}.zip; "
          f"learning curve in {csv_path}")


if __name__ == "__main__":
    main()
