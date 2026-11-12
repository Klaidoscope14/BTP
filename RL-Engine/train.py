"""
Train a MaskablePPO agent on the fixed-order EVRP environment.

Two training modes:

  1. Known map (the professor's setting): the graph is fixed and known, and
     every episode the agent gets a different user-style visit order — a
     random subset of the map's customers in a random order.

        python train.py --map examples/sample_map.json --min-visit 4 --max-visit 12

  2. Random maps: a new synthetic map every episode (tests generalisation).

        python train.py --nc 10 --ns 6 --steps 500000

Curriculum (bigger routes step by step): train on short routes first, then
longer ones, keeping the same network and normalisation statistics:

        python train.py --map examples/large_map.json --curriculum 5,15,30 --steps 2000000

Time windows: --tw-prob 0.5 gives half the training routes random customer
time windows (always achievable), so one model handles routes with and
without windows.

Stage i trains on visit orders of ceil(frac * size_i) .. size_i customers
(--stage-min-frac, default 0.5). Steps are split between stages in
proportion to their size unless --stage-steps gives them explicitly. Each
stage saves its own model and best checkpoint in <save-dir>/stage<i>_<size>/.

The agent's observation size depends only on the number of stations, so a
model trained here can be run on any visit order (any length) over a map with
the same number of stations.
"""

import argparse
import json
import math
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from envs.evrp_env import DEFAULT_CHARGE_LEVELS, EVAL_SEED_START, EVRPEnv
from utils.world import load_world


def env_kwargs(args) -> dict:
    """EVRPEnv settings that the trained model depends on. Saved in
    train_config.json so evaluate.py rebuilds the same env."""
    return {
        "price_lookahead": args.lookahead,
        "charge_levels": args.charge_levels,
        "late_penalty_fixed": args.late_fixed,
        "late_penalty_per_hour": args.late_per_hour,
    }


def window_kwargs(args) -> dict:
    """Training-time window sampling (not needed to run the model later)."""
    if args.tw_prob <= 0:
        return {"time_windows": "off"}
    return {"time_windows": "random", "tw_prob": args.tw_prob}


def make_env(args, base_map, seed, visit_range, world_seed_pool=None):
    def _init():
        common = dict(visit_range=visit_range, world_seed_pool=world_seed_pool,
                      seed=seed, **env_kwargs(args), **window_kwargs(args))
        if base_map is not None:
            env = EVRPEnv(fixed_world=base_map, randomize_order=True, **common)
        else:
            env = EVRPEnv(num_customers=args.nc, num_stations=args.ns,
                          grid_size=args.grid, randomize_world=True, **common)
        return Monitor(env)
    return _init


def build_stages(args, n_total):
    """[(name, (lo, hi), steps), ...]"""
    if not args.curriculum:
        return [("all", (args.min_visit, args.max_visit), args.steps)]

    sizes = [int(x) for x in args.curriculum.split(",")]
    if sorted(sizes) != sizes or sizes[0] < 1:
        sys.exit("--curriculum sizes must be increasing, e.g. 5,15,30")
    if sizes[-1] > n_total:
        sys.exit(f"--curriculum asks for {sizes[-1]} customers but the map only has "
                 f"{n_total}. Use a bigger map (examples/large_map.json) or --nc.")
    if args.stage_steps:
        steps = [int(x) for x in args.stage_steps.split(",")]
        if len(steps) != len(sizes):
            sys.exit("--stage-steps needs one value per curriculum stage")
    else:
        steps = [int(args.steps * s / sum(sizes)) for s in sizes]
    return [(f"stage{i + 1}_{size}",
             (max(1, math.ceil(args.stage_min_frac * size)), size), st)
            for i, (size, st) in enumerate(zip(sizes, steps))]


def save_config(args, n_stations, stages, path):
    cfg = {**vars(args), "num_stations": n_stations,
           "env_kwargs": env_kwargs(args),
           "stages": [{"name": n, "visit_range": list(r), "steps": s}
                      for n, r, s in stages]}
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


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
                        help="Comma-separated route sizes trained in turn, e.g. 5,15,30")
    parser.add_argument("--stage-steps", type=str, default=None,
                        help="Comma-separated steps per curriculum stage "
                             "(default: --steps split in proportion to stage size)")
    parser.add_argument("--stage-min-frac", type=float, default=0.5,
                        help="Each stage samples routes of ceil(frac*size)..size customers")
    parser.add_argument("--lookahead", type=int, default=4, help="Hours of future prices observed")
    parser.add_argument("--charge-levels", type=str,
                        default=",".join(str(l) for l in DEFAULT_CHARGE_LEVELS),
                        help="Target battery fractions a charge can stop at; 0 = just "
                             "enough, must include 1 (full). Use '1' for full-only.")
    parser.add_argument("--tw-prob", type=float, default=0.0,
                        help="Fraction of training routes given random (always "
                             "achievable) customer time windows; 0 = none")
    parser.add_argument("--late-fixed", type=float, default=5.0,
                        help="Penalty ($) for each customer reached late")
    parser.add_argument("--late-per-hour", type=float, default=2.0,
                        help="Extra penalty ($) per hour late")
    parser.add_argument("--steps", type=int, default=500_000, help="Total training steps")
    parser.add_argument("--envs", type=int, default=4, help="Parallel environments")
    parser.add_argument("--subproc", action="store_true",
                        help="Run the parallel environments in separate processes")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument("--save-dir", type=str, default="models/saved_weights")
    args = parser.parse_args()
    args.charge_levels = [float(x) for x in args.charge_levels.split(",")]

    base_map = load_world(args.map) if args.map else None
    n_total = base_map.num_customers if base_map else args.nc
    n_stations = base_map.num_stations if base_map else args.ns
    if args.curriculum and base_map is None:
        args.nc = n_total = max(args.nc, max(int(x) for x in args.curriculum.split(",")))
    args.max_visit = min(args.max_visit or n_total, n_total)
    args.min_visit = max(1, min(args.min_visit or max(1, n_total // 2), args.max_visit))
    stages = build_stages(args, n_total)

    os.makedirs(args.save_dir, exist_ok=True)
    cfg_path = os.path.join(args.save_dir, "train_config.json")
    save_config(args, n_stations, stages, cfg_path)
    where = f"map {args.map}" if base_map else f"random maps ({args.nc} customers)"
    print(f"Training MaskablePPO on {where}, {n_stations} stations, "
          f"charge levels {args.charge_levels}, time windows on "
          f"{100 * max(args.tw_prob, 0):.0f}% of routes")
    for name, (lo, hi), st in stages:
        print(f"  {name}: routes of {lo}-{hi} customers, {st} steps")

    vec_cls = SubprocVecEnv if args.subproc else DummyVecEnv
    train_env = VecNormalize(
        vec_cls([make_env(args, base_map, args.seed + i, stages[0][1])
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
        verbose=1,
        seed=args.seed,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    )

    # Fixed eval episodes, disjoint from training and test seeds.
    eval_pool = list(range(EVAL_SEED_START, EVAL_SEED_START + args.eval_episodes))
    freq = max(10_000 // args.envs, 1)
    multi = len(stages) > 1

    for i, (name, vrange, st) in enumerate(stages):
        if st <= 0:
            continue
        stage_dir = os.path.join(args.save_dir, name) if multi else args.save_dir
        os.makedirs(stage_dir, exist_ok=True)
        train_env.env_method("set_visit_range", vrange)

        eval_env = VecNormalize(
            DummyVecEnv([make_env(args, base_map, 0, vrange, world_seed_pool=eval_pool)]),
            norm_obs=True, norm_reward=False, clip_obs=10.0, training=False,
        )
        eval_env.obs_rms = train_env.obs_rms   # share running observation stats

        callbacks = [
            CheckpointCallback(save_freq=freq, save_path=args.save_dir,
                               name_prefix="ppo_evrp", save_vecnormalize=True),
            MaskableEvalCallback(eval_env, best_model_save_path=stage_dir,
                                 log_path=stage_dir, eval_freq=freq,
                                 n_eval_episodes=len(eval_pool), deterministic=True),
        ]
        print(f"\n=== {name}: routes of {vrange[0]}-{vrange[1]} customers, {st} steps ===")
        model.learn(total_timesteps=st, callback=callbacks, progress_bar=True,
                    reset_num_timesteps=(i == 0))
        if multi:
            model.save(os.path.join(stage_dir, "model"))
            train_env.save(os.path.join(stage_dir, "vec_normalize.pkl"))

    final_path = os.path.join(args.save_dir, "ppo_evrp_final")
    model.save(final_path)
    train_env.save(os.path.join(args.save_dir, "vec_normalize.pkl"))
    if multi:
        # best checkpoint of the last (longest-route) stage
        last_dir = os.path.join(args.save_dir, stages[-1][0])
        for f in ("best_model.zip", "evaluations.npz"):
            if os.path.exists(os.path.join(last_dir, f)):
                shutil.copy(os.path.join(last_dir, f), os.path.join(args.save_dir, f))
    save_config(args, n_stations, stages, cfg_path)
    print(f"\nTraining complete. Model saved to {final_path}.zip")


if __name__ == "__main__":
    main()
