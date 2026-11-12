# EVRP RL Engine — Fixed-Order Charging Optimizer

## Problem Reformulation

The customer visit order is **fixed** (provided by the user). The graph (all
node positions, distances, station prices) is **fully known**. The RL agent's
job is to decide the **charging strategy**: between each pair of consecutive
mandatory stops, should the vehicle detour to a charging station? If so,
which one, and **how much should it charge**? Customers may also have
**delivery time windows**.

### Why this is still hard
With a fixed customer order of length N, M stations and L charge levels, the
agent faces 1 + M·L choices at each of the N+1 legs. For N=20, M=6, L=5 that
is 31^21 ≈ 2×10^31 strategies. Greedy heuristics leave money on the table
because they don't reason about *future* prices, and with partial charging
and deadlines the interactions between legs get stronger still.

### How the professor's requirements map to the code
| Requirement | Implementation |
|---|---|
| Customers visited in a fixed order | The agent never chooses the next customer, only whether/where/how much to charge on each leg |
| Not all nodes are visited | `visit_order` is any subset of the map's customers |
| Nodes provided by the user | JSON map (`examples/sample_map.json`) with a `visit_order`, or `evaluate.py --order C05,C12,...` |
| Graph is known | Distance matrix and every station's 24-hour price schedule are fixed and given; training uses the same map with random visit orders |
| RL engine | MaskablePPO (sb3-contrib), `train.py` / `evaluate.py` |
| Partial charging | Action = (station, charge level); level 0 = "just enough" |
| Optimal comparison | Exact branch-and-bound solver, `utils/optimal_solver.py`, `evaluate.py --optimal` |
| Bigger routes step by step | `train.py --curriculum 5,15,30` on `examples/large_map.json` |
| Delivery time windows | `"window": [earliest, latest]` per customer; waiting + lateness penalty + time-aware mask |

## Architecture

```
RL-Engine/
├── envs/
│   └── evrp_env.py          # Gymnasium env: fixed-order EVRP (+ shared transition model)
├── utils/
│   ├── world.py             # Map building, JSON load/save, visit orders, windows
│   ├── optimal_solver.py    # Exact branch-and-bound solver for small routes
│   └── visualizer.py        # Route + price plotting
├── examples/
│   ├── sample_map.json      # Example user map + visit order (15 customers)
│   ├── sample_map_tw.json   # Same route with delivery time windows
│   ├── large_map.json       # 40-customer version (same stations) for the curriculum
│   └── make_example_maps.py # Rebuilds the two derived maps
├── models/
│   ├── saved_weights/       # Current PPO model, checkpoints, VecNormalize stats, config
│   └── archive/             # Older models (not compatible with the current env)
├── tests/test_engine.py     # Sanity tests (mask safety, solver exactness, windows)
├── train.py                 # MaskablePPO training loop (+ curriculum)
├── evaluate.py              # Run trained agent + baselines + optimum, compare
├── learning_curve.py        # Cost / gap to optimum vs training steps
└── PLAN.md                  # This file
```

## MDP Formulation

### Action A_t (Discrete, 1 + M·L)
- 0: proceed directly to the next mandatory stop
- 1 + k·L + l: detour to station k, charge to level l, then proceed

Charge levels (default 0, 0.25, 0.5, 0.75, 1.0) are target battery fractions
on leaving the station. A target is never below **"just enough"** = energy to
reach the next stop + `need` there (see the mask), so level 0 means "charge
the minimum" and 1.0 "charge to full". Detours that would charge nothing are
masked out. Charging takes `kWh / charge_rate` hours at the price of the hour
the vehicle arrives at the station.

### State S_t (flattened vector, every value in [0, 1])
| Component             | Shape  | Description                                       |
|-----------------------|--------|---------------------------------------------------|
| battery_soc           | (1,)   | Current SoC                                       |
| progress              | (1,)   | Fraction of legs completed                        |
| dist_to_next          | (1,)   | Distance to next mandatory stop                   |
| time_of_day           | (1,)   | Current hour mod 24                               |
| remaining_energy      | (1,)   | Energy to finish without detours, squashed x/(x+2·cap) |
| energy_shortfall      | (1,)   | max(0, remaining_energy − battery), squashed      |
| following_leg         | (1,)   | Length of the leg after the next one              |
| legs_left             | (1,)   | Legs still to drive, squashed x/(x+10)            |
| tw_active             | (1,)   | 1 if any remaining stop has a time window         |
| tw_slack              | (1,)   | Slack to the next stop's (tightened) deadline if driven direct, 0.5+0.5·tanh(h/12) |
| tw_wait               | (1,)   | How long the vehicle would wait at the next stop, squashed |
| current_prices        | (M,)   | Price at each station right now                   |
| arrival_prices        | (M,)   | Price at each station when we'd arrive there      |
| future_prices         | (M*K,) | Price at each station for the next K hours        |
| station_detour_dists  | (M,)   | Extra distance for each station detour            |
| min_charge            | (M,)   | kWh a "just enough" charge at each station adds (÷cap) |
| station_valid         | (M,)   | 1 if some charge level at station k is allowed    |

Total: 11 + 5M + M·K (M=6, K=4 → 65 dims). The size depends only on M and
K, so one trained model handles visit orders of any length. Squashing
(instead of clipping) keeps long routes distinguishable for the curriculum.

### Action masking
At reset the env computes, backwards from the depot, `need[j]` — the least
battery the vehicle must have on arriving at stop j to still finish the
route. Charging to full is the most a station can give, so partial charging
doesn't change this table. An action is valid only if it arrives at the next
stop with at least `need`; following the mask never runs the battery flat.

With time windows the env also computes `latest_arrival[j]`: the stop's own
deadline, tightened by the deadlines after it (driving direct, the fastest
way). Among the battery-safe actions, the mask keeps only those that reach
the next stop by `latest_arrival` whenever there are any; otherwise lateness
is unavoidable and all battery-safe actions stay allowed.

### Time windows
`[earliest, latest]` per customer, absolute hours (06:00 day 0 = 6.0). Early
arrival → wait until `earliest`. Late arrival → penalty
`late_penalty_fixed + late_penalty_per_hour · hours_late` ($5 + $2/h).
Windows come from the map (`time_windows="map"`), are drawn per episode
(`"random"`, used in training with `--tw-prob`), or are ignored (`"off"`).
Random windows are built around a feasible reference drive (fastest safe
action each leg), so an on-time plan always exists.

### Reward R_t
- **Step cost:** −(distance cost + charging cost + lateness penalty) of the leg, in $
- **Route completion:** +50
- **Battery death:** −(100 + worst-case cost of the remaining legs), terminal

So the episode return is exactly `50 − objective`, where objective = total
cost + lateness penalties (= total cost without windows). Maximising reward
is the same thing as minimising what `evaluate.py` reports.

### Episode
One episode = one full route execution (depot → customer_1 → ... → customer_N → depot).
The agent makes one decision per leg (N+1 decisions total).

## Exact solver (`utils/optimal_solver.py`)
Depth-first branch and bound over the env's own transition model
(`EVRPEnv.simulate`), so its costs are exactly what the env would charge and
its action set is exactly the agent's (battery-safe actions; it may choose to
be late if that is cheaper, so its optimum is a true lower bound on what the
agent can achieve). Pruning:
- lower bound = cost so far + direct distance cost of the rest of the route
  (detours only add distance) + energy still to buy × cheapest price on the map
- memo on (leg, battery, time); charge levels that collapse to the same
  target are merged
It is checked against plain enumeration in `tests/test_engine.py`. Routes of
up to 12 customers solve in well under a second; a time limit returns the
best route found (marked not proven) on longer ones.

Finding while building it: if detours that charge nothing were allowed, the
optimum would sometimes drive through a station just to arrive at a cheaper
hour — i.e. *waiting for cheap prices* has value. The mask forbids those
detours; an explicit "wait at the station" action is a possible extension.

## Baselines (for comparison)
All baselines follow the same action mask as the agent, so none of them ever
runs out of battery, and they skip charging when the rest of the route can
be finished on the current battery. They charge as full as the mask allows.
1. **Charge when forced:** go direct; charge at the nearest station only when the mask forbids going direct
2. **Greedy nearest:** charge at nearest valid station when battery < 30%
3. **Greedy cheapest:** charge at the valid station with the lowest price at arrival when battery < 30%
4. **Smart rule-based:** threshold rule + opportunistic charging when a station is < 65% of the average price, picking the station with the lowest detour + charge cost

## Training Plan
- Algorithm: MaskablePPO (sb3-contrib), MlpPolicy [256, 256], VecNormalize
- Known-map mode (`--map`): the graph is fixed; every episode samples a new
  visit order (random subset of customers, random order)
- Curriculum (`--curriculum 5,15,30`): stage i trains on routes of
  ⌈0.5·size⌉..size customers; the same network and normalisation statistics
  carry over between stages; steps split in proportion to stage size
- Time windows on a fraction of training routes (`--tw-prob 0.5`), so one
  model handles routes with and without windows
- Random-map mode: a new synthetic map every episode
- Seeds are split so training never sees evaluation or test episodes:
  training maps use seeds 0–999,999, the EvalCallback uses 2,000,000+, and
  `evaluate.py` / `learning_curve.py` use 3,000,000+

## Results (model in models/saved_weights/)
Trained with `train.py --map examples/large_map.json --curriculum 5,15,30
--steps 2000000 --tw-prob 0.5 --envs 8 --subproc` (≈10 min on 8 cores).
All numbers are on held-out test routes; "gap" = mean extra cost over the
exact optimum of each route.

Short routes (sample map, 4–12 customers, 100 routes):

| Method | Cost, no windows | Gap | Cost, windows | Gap | Late |
|---|---|---|---|---|---|
| Optimal (exact) | $22.80 | 0% | $26.00 | 0% | 0.1% |
| **PPO agent** | **$29.07** | **27.2%** | **$35.25** | **33.4%** | 1.0% |
| Smart rule-based | $35.33 | 54.8% | $45.63 | 88.5% | 1.0% |
| Greedy cheapest | $40.74 | 83.5% | $49.86 | 98.4% | 1.0% |
| Charge when forced | $54.00 | 136.9% | $55.79 | 118.3% | 0.5% |

Long routes (large map, 25–30 customers, 50 routes; too long to solve exactly):

| Method | Cost, no windows | Cost, windows |
|---|---|---|
| **PPO agent** | **$111.64** | **$124.43** |
| Smart rule-based | $128.75 | $141.42 |
| Greedy cheapest | $148.55 | $161.43 |

Old full-charge model (v1, 300k steps) on its own benchmark (30 routes,
7–15 customers): $42.37; the new model on the same routes: $40.07 (optimum
$30.65).

**Longer training** (`learning_curve.py`): on short routes the gap falls from
54% at 200k steps to ≈29% by 400k and then stays flat (28–36%) up to 2M —
the last stage trains only on 15–30-customer routes, so short-route cost
drifts slightly. On 25–30-customer routes cost falls from $159 (200k) to
$109 (800k) and then barely moves without windows ($109 at 2M), while with
windows it keeps improving slowly ($124 → $120). So 2M steps buys little over
~1M; the remaining ~30% gap needs something other than more steps (a bigger
network / longer look-ahead, reward shaping, or the "wait" action).

**Curriculum vs none** (control: same 2M steps, routes 3–30 from the start,
`models/experiments/no_curriculum/`): small but consistent edge for the
curriculum — short-route gap 27.2% vs 30.7% (no windows), 33.4% vs 34.9%
(windows); 30-customer routes $109.8 vs $111.4 (no windows), $122.5 vs $122.8
(windows).

## Not done yet
- An explicit "wait for a cheaper hour" action (see the solver finding)
- Stochastic / forecast prices (the proposal's noisy price model)
- Service time at customers
