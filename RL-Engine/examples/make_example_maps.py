"""
Build the derived example maps from sample_map.json:

  large_map.json      the sample map grown to 40 customers. The depot, the 6
                      stations (positions and price schedules) and the first
                      15 customers are copied; 25 extra customers are placed
                      at random in the same area. A model trained on it also
                      fits the sample map (same stations), and it is big
                      enough for the 5 -> 15 -> 30 customer curriculum.

  sample_map_tw.json  the sample map and its visit order, with a delivery
                      time window on every customer of the route. Windows
                      are drawn the same way as random training windows, so
                      they can always be met.

    python examples/make_example_maps.py
"""

import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402

from envs.evrp_env import EVRPEnv  # noqa: E402
from utils.world import load_world, save_world, with_windows  # noqa: E402

EXTRA_CUSTOMERS = 25
SEED = 2024
HALF_SIZE = 40.0
TW_SEED = 7


def make_large_map():
    with open(os.path.join(HERE, "sample_map.json")) as f:
        spec = json.load(f)

    rng = random.Random(SEED)
    n0 = len(spec["customers"])
    for i in range(EXTRA_CUSTOMERS):
        spec["customers"].append({
            "name": f"C{n0 + i + 1:02d}",
            "x": round(rng.uniform(-HALF_SIZE, HALF_SIZE), 1),
            "y": round(rng.uniform(-HALF_SIZE, HALF_SIZE), 1),
        })
    names = [c["name"] for c in spec["customers"]]
    spec["visit_order"] = rng.sample(names, 30)

    out = os.path.join(HERE, "large_map.json")
    with open(out, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"Wrote {out}: {len(names)} customers, {len(spec['stations'])} stations, "
          f"visit order of {len(spec['visit_order'])}")


def make_tw_map():
    world = load_world(os.path.join(HERE, "sample_map.json"))
    env = EVRPEnv(fixed_world=world, time_windows="random", seed=TW_SEED)
    windows = np.array([[0.0, np.inf]] * world.num_customers)
    for j, ci in enumerate(world.visit_order):
        windows[ci] = [round(env.tw_earliest[j], 1), round(env.tw_latest[j], 1)]
    out = os.path.join(HERE, "sample_map_tw.json")
    save_world(with_windows(world, windows), out)
    print(f"Wrote {out}: windows on the {len(world.visit_order)} customers of the route")


if __name__ == "__main__":
    make_large_map()
    make_tw_map()
