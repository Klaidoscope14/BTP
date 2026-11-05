"""
Train a MaskablePPO agent on the fixed-order EVRP environment.

Two training modes:

  1. Known map (the professor's setting): the graph is fixed and known, and
     every episode the agent gets a different user-style visit order — a
     random subset of the map's customers in a random order.

        python train.py --map examples/sample_map.json --min-visit 4 --max-visit 12

  2. Random maps: a new synthetic map every episode (tests generalisation).

        python train.py --nc 10 --ns 6 --steps 500000

The agent's observation size depends only on the number of stations, so a
model trained here can be run on any visit order (any length) over a map with
the same number of stations.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.evrp_env import EVAL_SEED_START, EVRPEnv
from utils.world import load_world


def make_env(args, base_map, seed, world_seed_pool=None):
    def _init():
        if base_map is not None:
            env = EVRPEnv(
                fixed_world=base_map,
                randomize_order=True,
                visit_range=(args.min_visit, args.max_visit),
                price_lookahead=args.lookahead,
                world_seed_pool=world_seed_pool,
                seed=seed,
            )
        else:
            env = EVRPEnv(
                num_customers=args.nc,
                num_stations=args.ns,
                grid_size=args.grid,
                visit_range=(args.min_visit, args.max_visit),
                price_lookahead=args.lookahead,
                randomize_world=True,
                world_seed_pool=world_seed_pool,
                seed=seed,
            )
        return Monitor(env)
    return _init


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
    parser.add_argument("--lookahead", type=int, default=4, help="Hours of future prices observed")
    parser.add_argument("--steps", type=int, default=500_000, help="Total training steps")
    parser.add_argument("--envs", type=int, default=4, help="Parallel environments")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument("--save-dir", type=str, default="models/saved_weights")
    args = parser.parse_args()

    base_map = load_world(args.map) if args.map else None
    n_total = base_map.num_customers if base_map else args.nc
    n_stations = base_map.num_stations if base_map else args.ns
    args.max_visit = min(args.max_visit or n_total, n_total)
    args.min_visit = max(1, min(args.min_visit or max(1, n_total // 2), args.max_visit))

    os.makedirs(args.save_dir, exist_ok=True)
    where = f"map {args.map}" if base_map else f"random maps ({args.nc} customers)"
    print(f"Training MaskablePPO on {where}, {n_stations} stations, "
          f"visiting {args.min_visit}-{args.max_visit} customers, {args.steps} steps")

    train_env = VecNormalize(
        DummyVecEnv([make_env(args, base_map, seed=args.seed + i)
                     for i in range(args.envs)]),
        norm_obs=True, norm_reward=True, clip_obs=10.0,
    )

    # Fixed set of eval episodes, disjoint from training and test seeds.
    eval_pool = list(range(EVAL_SEED_START, EVAL_SEED_START + 20))
    eval_env = VecNormalize(
        DummyVecEnv([make_env(args, base_map, seed=0, world_seed_pool=eval_pool)]),
        norm_obs=True, norm_reward=False, clip_obs=10.0, training=False,
    )
    # share running observation stats with the eval env
    eval_env.obs_rms = train_env.obs_rms

    checkpoint_cb = CheckpointCallback(
        save_freq=max(10_000 // args.envs, 1),
        save_path=args.save_dir,
        name_prefix="ppo_evrp",
        save_vecnormalize=True,
    )
    eval_cb = MaskableEvalCallback(
        eval_env,
        best_model_save_path=args.save_dir,
        log_path=args.save_dir,
        eval_freq=max(10_000 // args.envs, 1),
        n_eval_episodes=len(eval_pool),
        deterministic=True,
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

    print("Starting training...")
    model.learn(total_timesteps=args.steps, callback=[checkpoint_cb, eval_cb],
                progress_bar=True)

    final_path = os.path.join(args.save_dir, "ppo_evrp_final")
    model.save(final_path)
    train_env.save(os.path.join(args.save_dir, "vec_normalize.pkl"))
    with open(os.path.join(args.save_dir, "train_config.json"), "w") as f:
        json.dump({**vars(args), "num_stations": n_stations}, f, indent=2)
    print(f"\nTraining complete. Model saved to {final_path}.zip")


if __name__ == "__main__":
    main()
