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

## Architecture

```
RL-Engine/
├── envs/
│   ├── __init__.py
│   └── evrp_env.py          # Gymnasium env: fixed-order EVRP
├── utils/
│   ├── __init__.py
│   ├── world.py              # World generation (nodes, distances, prices)
│   └── visualizer.py         # Route + price plotting
├── models/
│   └── saved_weights/        # PPO checkpoints
├── train.py                  # PPO training loop
├── evaluate.py               # Run trained agent + baselines, compare
├── requirements.txt
└── PLAN.md                   # This file
```

## MDP Formulation

### State S_t (flattened vector)
| Component             | Shape  | Description                                |
|-----------------------|--------|--------------------------------------------|
| battery_soc           | (1,)   | Current SoC, normalized [0, 1]             |
| progress              | (1,)   | Fraction of route completed [0, 1]         |
| dist_to_next          | (1,)   | Distance to next mandatory stop, normalized|
| time_of_day           | (1,)   | Current hour mod 24, normalized [0, 1]     |
| current_prices        | (M,)   | Price at each station right now             |
| future_prices         | (M*K,) | Price at each station for next K hours      |
| station_detour_dists  | (M,)   | Extra distance for each station detour      |
| station_reachable     | (M,)   | Binary: can battery reach station k?        |

Total: 4 + 3M + M*K  (e.g., M=8, K=4 → 40 dims)

### Action A_t (Discrete, M+1)
- 0: proceed directly to the next mandatory customer
- 1..M: detour to station k, charge to full, then proceed to next customer

Invalid actions (unreachable stations) are masked.

### Reward R_t
- **Step cost:** -(distance_cost + charging_cost) normalized
- **Arbitrage bonus:** if charged at price < average, bonus proportional to savings
- **Battery death:** -100 (terminal)
- **Route completion:** +50

### Episode
One episode = one full route execution (depot → customer_1 → ... → customer_N → depot).
The agent makes one decision per leg (N+1 decisions total: N legs + return to depot).

## Baselines (for comparison)
1. **Greedy nearest:** charge at nearest station when battery < threshold
2. **Greedy cheapest-now:** charge at cheapest reachable station when battery < threshold
3. **Rule-based smart:** the two-phase solver from Manual-Simulation
4. **Always direct:** never charge voluntarily (only when forced)

## Training Plan
- Algorithm: PPO (Stable-Baselines3)
- Policy: MlpPolicy (flat state vector)
- Randomization: each episode uses a different world (random customer positions,
  random fixed order, random price schedules) to prevent memorization
- Curriculum: start with 5 customers, scale to 15, then 30
- Steps: 500K–2M depending on convergence
