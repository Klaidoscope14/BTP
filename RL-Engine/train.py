"""
Train a PPO agent on the fixed-order EVRP environment.

Usage:
    python train.py                          # default: 10 customers, 6 stations, 500K steps
    python train.py --steps 1000000 --nc 15  # 15 customers, 1M steps
"""

import argparse
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from envs.evrp_env import EVRPEnv


def make_env(num_customers, num_stations, num_visit, grid_size, seed=None):
    def _init():
        env = EVRPEnv(
            num_customers=num_customers,
            num_stations=num_stations,
            num_visit=num_visit,
            grid_size=grid_size,
            seed=seed,
            randomize_world=True,
        )
        return Monitor(env)
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train PPO on fixed-order EVRP")
    parser.add_argument("--nc", type=int, default=10, help="Number of customers")
    parser.add_argument("--ns", type=int, default=6, help="Number of stations")
    parser.add_argument("--nv", type=int, default=None, help="Customers to visit (default=all)")
    parser.add_argument("--grid", type=float, default=40.0, help="Grid half-size")
    parser.add_argument("--steps", type=int, default=500_000, help="Total training steps")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument("--save-dir", type=str, default="models/saved_weights",
                        help="Directory for checkpoints")
    args = parser.parse_args()

    num_visit = args.nv or args.nc
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Training PPO on EVRP: {args.nc} customers, {args.ns} stations, "
          f"visiting {num_visit}, {args.steps} steps")

    # vectorized training env
    train_env = DummyVecEnv([
        make_env(args.nc, args.ns, num_visit, args.grid, seed=args.seed + i)
        for i in range(4)  # 4 parallel envs
    ])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True,
                              clip_obs=10.0)

    # eval env (fixed seed for consistent comparison)
    eval_env = DummyVecEnv([
        make_env(args.nc, args.ns, num_visit, args.grid, seed=9999)
    ])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                             clip_obs=10.0, training=False)

    # callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=25_000,
        save_path=args.save_dir,
        name_prefix="ppo_evrp",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=args.save_dir,
        log_path=args.save_dir,
        eval_freq=10_000,
        n_eval_episodes=20,
        deterministic=True,
    )

    model = PPO(
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
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        ),
    )

    print("Starting training...")
    model.learn(
        total_timesteps=args.steps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    final_path = os.path.join(args.save_dir, "ppo_evrp_final")
    model.save(final_path)
    train_env.save(os.path.join(args.save_dir, "vec_normalize.pkl"))
    print(f"\nTraining complete. Model saved to {final_path}.zip")


if __name__ == "__main__":
    main()
