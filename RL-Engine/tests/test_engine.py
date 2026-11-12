"""
Sanity tests for the RL engine (no training needed):

    python -m pytest tests -q
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from envs.evrp_env import EVRPEnv  # noqa: E402
from evaluate import BASELINES, make_optimal_policy, run_episode  # noqa: E402
from utils.optimal_solver import solve_optimal  # noqa: E402
from utils.world import load_world, world_from_dict, world_to_dict  # noqa: E402

MAP = load_world(os.path.join(ROOT, "examples", "sample_map.json"))


def make(ep, k=(4, 10), **kw):
    return EVRPEnv(fixed_world=MAP, randomize_order=True, visit_range=k,
                   world_seed_pool=[1000 + ep], **kw)


def brute_force(env):
    """Cheapest completion over every action the mask could allow."""
    best = [float("inf")]
    n = len(env.mandatory_sequence)

    def rec(node, b, t, leg, c):
        if leg >= n:
            best[0] = min(best[0], c)
            return
        for _, r in env.valid_legs(node, b, t, leg):
            rec(r.target, r.battery, r.time, leg + 1, c + r.objective)

    rec(env.current_node_idx, env.battery, env.time, 0, 0.0)
    return best[0]


@pytest.mark.parametrize("tw", ["off", "random"])
def test_masked_random_play_never_dies(tw):
    rng = np.random.default_rng(0)
    for ep in range(100):
        env = make(ep, (4, 15), time_windows=tw)
        env.reset()
        done = False
        while not done:
            obs, _, done, _, info = env.step(rng.choice(np.flatnonzero(env.action_masks())))
            assert env.observation_space.contains(obs)
        assert not info.get("dead")


def test_partial_charging_actions():
    env = make(0)
    assert env.action_space.n == 1 + MAP.num_stations * len(env.charge_levels)
    for a in range(1, env.action_space.n):
        s, l = env.decode(a)
        assert env.encode(s, l) == a


@pytest.mark.parametrize("levels,tw", [((0, 0.25, 0.5, 0.75, 1.0), "off"),
                                       ((1.0,), "off"),
                                       ((0, 0.25, 0.5, 0.75, 1.0), "random")])
def test_solver_matches_brute_force(levels, tw):
    for ep in range(10):
        env = make(ep, (2, 3), charge_levels=levels, time_windows=tw)
        res = solve_optimal(env)
        assert res.proven
        assert res.cost == pytest.approx(brute_force(env), abs=1e-6)
        info = {}
        for a in res.actions:                     # the plan replays in the env
            _, _, _, _, info = env.step(a)
        assert info["objective"] == pytest.approx(res.cost, abs=1e-6)


def test_optimal_beats_every_baseline():
    opt = make_optimal_policy(30)
    for ep in range(10):
        best = run_episode(make(ep, time_windows="random"), opt)["objective"]
        for policy in BASELINES.values():
            info = run_episode(make(ep, time_windows="random"), policy)
            assert info["objective"] >= best - 1e-6


def test_partial_charging_never_worse_than_full_only():
    for ep in range(10):
        part = solve_optimal(make(ep, (4, 8))).cost
        full = solve_optimal(make(ep, (4, 8), charge_levels=(1.0,))).cost
        assert part <= full + 1e-6


def test_random_windows_are_achievable():
    # with lateness made prohibitively expensive, the optimum is never late
    # (with the default $5 penalty being a little late can be the cheaper plan)
    for ep in range(20):
        env = make(ep, (4, 10), time_windows="random", late_penalty_fixed=1e4)
        info = run_episode(env, make_optimal_policy(30))
        assert info["late_stops"] == 0


def test_windows_json_roundtrip():
    path = os.path.join(ROOT, "examples", "sample_map_tw.json")
    w = load_world(path)
    assert w.has_windows
    w2 = world_from_dict(world_to_dict(w))
    assert np.allclose(w.windows, w2.windows)


def test_set_visit_range():
    env = EVRPEnv(fixed_world=load_world(os.path.join(ROOT, "examples", "large_map.json")),
                  randomize_order=True, visit_range=(3, 5), seed=0)
    env.set_visit_range((25, 30))
    for _ in range(5):
        env.reset()
        assert 25 <= len(env.world.visit_order) <= 30
