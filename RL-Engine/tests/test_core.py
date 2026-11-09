"""
Core checks for the environment, masks and exact solver.

    python -m pytest tests/        (or)        python tests/test_core.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from envs import route as R
from envs.evrp_env import EVRPEnv
from utils.optimal import brute_force, solve
from utils.world import load_world

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = load_world(os.path.join(ROOT, "examples", "sample_map.json"))
LARGE = load_world(os.path.join(ROOT, "examples", "large_map.json"))


def episode(world, seed, **kw):
    return EVRPEnv(fixed_world=world, randomize_order=True, world_seed_pool=[seed], **kw)


def run_random(env, seed):
    env.reset()
    rng = np.random.default_rng(seed)
    done, ret, info = False, 0.0, {}
    while not done:
        a = int(rng.choice(np.flatnonzero(env.action_masks())))
        _, r, done, _, info = env.step(a)
        ret += r
    return ret, info


def test_solver_matches_brute_force():
    for ep in range(15):
        for kw in (dict(charge_levels=0), dict(charge_levels=3),
                   dict(charge_levels=3, time_windows="generated", tw_slack=(0.0, 3.0))):
            env = episode(SAMPLE, ep, visit_range=(3, 4), **kw)
            assert abs(solve(env.ctx).cost - brute_force(env.ctx)) < 1e-6, (ep, kw)


def test_solver_route_replays_to_its_cost():
    for ep in range(10):
        env = episode(SAMPLE, ep, visit_range=(5, 8))
        res = solve(env.ctx)
        env.reset()
        info = {}
        for a in res.actions:
            assert env.action_masks()[a]
            _, _, done, _, info = env.step(a)
        assert done and abs(info["total_cost"] - res.cost) < 1e-6


def test_mask_never_strands_or_misses_deadlines():
    for ep in range(60):
        env = episode(LARGE, ep, visit_range=(5, 30), time_windows="generated")
        ret, info = run_random(env, ep)
        assert not info.get("dead")
        assert info["late_customers"] == 0
        # episode return is exactly completion bonus - total cost
        assert abs(ret - (env.completion_bonus - info["total_cost"])) < 1e-6


def test_partial_charging_never_wastes_energy():
    for ep in range(40):
        _, info = run_random(episode(SAMPLE, ep, visit_range=(4, 15), charge_levels=5), ep)
        assert info["energy_unused"] < 1e-3


def test_legacy_mode_always_fills_battery():
    env = EVRPEnv(fixed_world=SAMPLE, charge_levels=0)
    assert env.action_space.n == 1 + SAMPLE.num_stations
    run_random(env, 0)
    for _, _, _, kwh, _ in env.charges:
        assert kwh > 0
    # every charge ended at full capacity
    assert all(abs(R.charge_target(env.ctx, 0, k, 0) - SAMPLE.battery_capacity) < 1e-9
               for k in range(SAMPLE.num_stations))


def test_observation_in_bounds():
    for kw in (dict(), dict(time_windows="generated"), dict(charge_levels=0)):
        env = episode(LARGE, 1, visit_range=(30, 30), **kw)
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape
        assert obs.min() >= 0.0 and obs.max() <= 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
