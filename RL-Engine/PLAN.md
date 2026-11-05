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
│   ├── __init__.py
│   └── evrp_env.py          # Gymnasium env: fixed-order EVRP
├── utils/
│   ├── __init__.py
│   ├── world.py              # Map building, JSON load/save, visit orders
│   └── visualizer.py         # Route + price plotting
├── examples/
│   └── sample_map.json       # Example user map + visit order
├── models/
│   └── saved_weights/        # PPO checkpoints, VecNormalize stats, config
├── train.py                  # MaskablePPO training loop
├── evaluate.py               # Run trained agent + baselines, compare
├── requirements.txt
├── README.md                 # How to run
└── PLAN.md                   # This file
```

## MDP Formulation

### State S_t (flattened vector, every value in [0, 1])
| Component             | Shape  | Description                                       |
|-----------------------|--------|---------------------------------------------------|
| battery_soc           | (1,)   | Current SoC                                       |
| progress              | (1,)   | Fraction of legs completed                        |
| dist_to_next          | (1,)   | Distance to next mandatory stop                   |
| time_of_day           | (1,)   | Current hour mod 24                               |
| remaining_energy      | (1,)   | Energy to finish the route without detours (÷3·cap)|
| energy_shortfall      | (1,)   | max(0, remaining_energy − battery) ÷ cap          |
| following_leg         | (1,)   | Length of the leg after the next one              |
| current_prices        | (M,)   | Price at each station right now                   |
| arrival_prices        | (M,)   | Price at each station when we'd arrive there      |
| future_prices         | (M*K,) | Price at each station for the next K hours        |
| station_detour_dists  | (M,)   | Extra distance for each station detour            |
| station_valid         | (M,)   | 1 if detouring via station k is allowed (mask)    |

Total: 7 + 4M + M*K  (e.g., M=6, K=4 → 55 dims). The size depends only on
M and K, so one trained model handles visit orders of any length.

### Action A_t (Discrete, M+1)
- 0: proceed directly to the next mandatory customer
- 1..M: detour to station k, charge to full, then proceed to next customer

### Action masking (exact)
At reset the env computes, backwards from the depot, `need[j]` — the least
battery the vehicle must have on arriving at stop j to still finish the
route (stations always charge to full, so this is a simple recursion). An
action is valid only if it arrives at the next stop with at least `need`.
Following the mask therefore never runs the battery flat; MaskablePPO only
ever samples valid actions.

### Reward R_t
- **Step cost:** −(distance cost + charging cost) of the leg, in $
- **Route completion:** +50
- **Battery death:** −(100 + worst-case cost of the remaining legs), terminal

So the episode return is exactly `50 − total_cost` — maximising reward is the
same thing as minimising the cost reported by `evaluate.py`. (The earlier
"arbitrage bonus" was removed: it paid the agent for charging whenever the
price was below a third of the average, which rewarded unnecessary detours.)

### Episode
One episode = one full route execution (depot → customer_1 → ... → customer_N → depot).
The agent makes one decision per leg (N+1 decisions total: N legs + return to depot).

## Baselines (for comparison)
All baselines follow the same action mask as the agent, so none of them ever
runs out of battery, and they skip charging when the rest of the route can
be finished on the current battery.
1. **Charge when forced:** go direct; charge at the nearest station only when the mask forbids going direct
2. **Greedy nearest:** charge at nearest valid station when battery < 30%
3. **Greedy cheapest:** charge at the valid station with the lowest price at arrival when battery < 30%
4. **Smart rule-based:** threshold rule + opportunistic charging when a station is < 65% of the average price, picking the station with the lowest detour + charge cost

## Training Plan
- Algorithm: MaskablePPO (sb3-contrib), MlpPolicy [256, 256], VecNormalize
- Known-map mode (`--map`): the graph is fixed; every episode samples a new
  visit order (random subset of 4–12 customers, random order)
- Random-map mode: a new synthetic map every episode
- Seeds are split so training never sees evaluation or test episodes:
  training maps use seeds 0–999,999, the EvalCallback uses 2,000,000+, and
  `evaluate.py` uses 3,000,000+
- Steps: 500K–2M depending on convergence

## Not done yet
- Curriculum (5 → 15 → 30 customers)
- Partial charging (the vehicle always charges to full, so it can pay for
  energy it never uses near the end of the route)
- An exact optimal solver for small routes, to measure how close the agent is to optimal
- Customer time windows
