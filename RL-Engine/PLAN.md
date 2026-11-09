# EVRP RL Engine — Fixed-Order Charging Optimizer

## Problem Reformulation

The customer visit order is **fixed** (provided by the user). The graph (all
node positions, distances, station prices) is **fully known**. The RL agent's
sole job is to decide the **charging strategy**: between each pair of
consecutive mandatory stops, should the vehicle detour to a charging station?
If so, which one?

### Why this is still hard
Even with a fixed customer order of length N and M stations, the agent faces
M+1 choices at each of the N legs — that's (M+1)^N possible charging
strategies. For N=20, M=8 this is 9^20 ≈ 1.2×10^19. Brute force is
impossible; greedy heuristics leave money on the table because they don't
reason about *future* prices. An RL agent can learn to anticipate cheap
windows and delay/advance charging accordingly.

### How the professor's requirements map to the code
| Requirement | Implementation |
|---|---|
| Customers visited in a fixed order | The agent never chooses the next customer, only whether/where to charge on each leg |
| Not all nodes are visited | `visit_order` is any subset of the map's customers |
| Nodes provided by the user | JSON map (`examples/sample_map.json`) with a `visit_order`, or `evaluate.py --order C05,C12,...` |
| Graph is known | Distance matrix and every station's 24-hour price schedule are fixed and given; training uses the same map with random visit orders |
| RL engine | MaskablePPO (sb3-contrib), `train.py` / `evaluate.py` |

## Architecture

```
RL-Engine/
├── envs/
│   ├── route.py             # Leg dynamics shared by env, baselines and solver:
│   │                        #   partial-charge targets, waiting/lateness,
│   │                        #   exact battery mask, deadline table
│   └── evrp_env.py          # Gymnasium env: fixed-order EVRP
├── utils/
│   ├── world.py             # Map building, JSON load/save, visit orders, time windows
│   ├── optimal.py           # Exact branch-and-bound solver (small routes)
│   └── visualizer.py        # Route + price plotting
├── examples/
│   ├── sample_map.json      # 15 customers, 6 stations + a visit order
│   ├── sample_map_tw.json   # same, with delivery time windows
│   └── large_map.json       # 30 customers (curriculum up to 30)
├── models/
│   ├── saved_weights/       # default model (partial charging, 2M steps)
│   └── experiments/         # one folder per experiment (see EXPERIMENTS.md)
├── tests/test_core.py       # solver = brute force, masks are safe, ...
├── train.py                 # MaskablePPO training (+ curriculum, cost curve)
├── evaluate.py              # agent vs baselines vs optimum
├── learning_curve.py        # plot / summarise cost_curve.csv
├── run_experiments.sh       # reproduces EXPERIMENTS.md
├── README.md
├── EXPERIMENTS.md
└── PLAN.md                  # This file
```

## MDP Formulation

### State S_t (flattened vector, every value in [0, 1])
| Component (scalars)   | Description                                              |
|-----------------------|----------------------------------------------------------|
| battery_soc           | Current SoC                                              |
| progress              | Fraction of legs completed                               |
| dist_to_next          | Distance to next mandatory stop                          |
| time_of_day           | Current hour mod 24                                      |
| remaining_energy      | Energy to finish without detours, E/(E+capacity)         |
| energy_shortfall      | max(0, remaining_energy − battery) ÷ 3·cap               |
| following_leg         | Length of the leg after the next one                     |
| legs_remaining        | n/(n+5) — absolute route length, for 5- to 30-stop routes|
| need_next             | Least battery needed on arrival at the next stop         |
| time_slack_next       | Hours of slack at the next stop going direct (tanh-scaled) |
| windows_active        | 1 if the route has time windows                          |

| Component (per station, M each) | Description                                |
|-----------------------|----------------------------------------------------------|
| current_price         | Price right now                                          |
| arrival_price         | Price when we'd arrive there                             |
| future_prices (K)     | Price for each of the next K hours                       |
| detour_dist           | Extra distance of the detour                             |
| station_valid         | 1 if any charge level at this station is allowed         |
| min_kwh               | Least energy to buy there to stay feasible               |
| max_kwh               | Most energy the rest of the route can use                |
| time_slack_via        | Slack at the next stop if detouring here                 |

Total: 11 + M·(7 + K)  (M=6, K=4 → 77 dims). The size depends only on M and
K, so one trained model handles visit orders of any length.

### Action A_t (Discrete, 1 + M·L)
- 0: proceed directly to the next mandatory customer
- 1 + k·L + l: detour to station k and charge to level l, where for the leg
  towards stop y and battery b_s on arrival at the station

      lo = e(s,y) + need[y]                 (least that keeps the route feasible)
      hi = min(capacity, e(s,y) + E_direct(y))   (enough to finish the route)
      target(l) = lo + l/(L−1) · (hi − lo)

  Level 0 buys the minimum (top up again later, somewhere cheaper); level L−1
  fills to full or to exactly what the rest of the route needs. No level ever
  buys energy the route cannot use; a level below b_s is masked.
- `--charge-levels 0` keeps the original behaviour: M+1 actions, always fill to 100 %.

### Action masking
- **Battery (exact).** At reset the env computes, backwards from the depot,
  `need[j]` — the least battery on arriving at stop j that still lets the route
  be finished. An action is valid only if it arrives at the next stop with at
  least `need`. Following the mask never runs the battery flat.
- **Time windows (hard).** At reset a backward table `T[j][b]` is computed: the
  latest arrival time at stop j with battery b that still meets every later
  deadline, charging time included (1 %-of-capacity battery grid, rounded down,
  so it is conservative). An action is valid only if it arrives at the next stop
  at time t with battery b and t ≤ T[next][b]. Only when no action can meet the
  windows is this check dropped and lateness paid for.

### Reward R_t
- **Step cost:** −(distance cost + charging cost + lateness cost) of the leg, in $
  (lateness = `late_penalty_fixed` + `late_penalty_per_hour` × hours late)
- **Route completion:** +50
- **Battery death:** −(100 + worst-case cost of the remaining legs), terminal

So the episode return is exactly `50 − total_cost`.

### Episode
One episode = one full route (depot → customer_1 → ... → customer_N → depot),
one decision per leg (N+1 decisions). Arriving early at a customer means
waiting until its window opens.

## Baselines (for comparison)
All baselines follow the same action mask as the agent (so they never run out
of battery or miss a deadline the mask protects), skip charging when the rest
of the route can be finished on the current battery, and charge to the top
level (full, or just enough to finish).
1. **Charge when forced:** go direct; charge at the nearest station only when the mask forbids going direct
2. **Greedy nearest:** charge at nearest valid station when battery < 30%
3. **Greedy cheapest:** charge at the valid station with the lowest price at arrival when battery < 30%
4. **Smart rule-based:** threshold rule + opportunistic charging when a station is < 65% of the average price, picking the station with the lowest detour + charge cost

## Exact optimum (utils/optimal.py)
Depth-first branch and bound over the same actions, mask and costs as the
agent. Lower bound = direct distance cost of the rest + energy shortfall ×
cheapest price anywhere + cheapest single detour if charging is still needed
+ lateness if driven directly; identical states reached at a higher cost are
cut. Verified against plain enumeration in `tests/test_core.py`. Proves
optimality in ~0.01 s for 5 customers, ~0.3 s for 10, a few seconds for 15.

## Training Plan
- Algorithm: MaskablePPO (sb3-contrib), MlpPolicy [256, 256], VecNormalize
- Known-map mode (`--map`): the graph is fixed; every episode samples a new
  visit order (random subset of customers, random order)
- Random-map mode: a new synthetic map every episode
- Curriculum (`--curriculum 5,15,30`): stage k samples route lengths in
  [⌈n_k/3⌉, n_k] and continues training the same network
- Seeds are split so training never sees evaluation or test episodes:
  training uses seeds 0–999,999, in-training evaluation 2,000,000+, and
  `evaluate.py` 3,000,000+
- Held-out cost logged every 25K steps to `cost_curve.csv`

## Done (future tasks from the mid-term review)
1. Partial charging — `--charge-levels`
2. Exact optimal solver for small routes — `utils/optimal.py`, `evaluate.py --optimal`
3. Curriculum 5 → 15 → 30 customers — `train.py --curriculum`, `examples/large_map.json`
4. Delivery time windows — JSON `window`, `--time-windows`, hard deadline mask
5. Longer training (1–2M steps) with learning curves — `cost_curve.csv`, `learning_curve.py`

Results: [EXPERIMENTS.md](EXPERIMENTS.md).
