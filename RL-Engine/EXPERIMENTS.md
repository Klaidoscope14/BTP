# Experiments — future tasks from the mid-term review

All numbers are on **held-out test routes** (seeds the training never saw),
map `examples/sample_map.json` (15 customers, 6 stations) unless stated.
"Gap" is the mean % above the exact optimum from `utils/optimal.py`, computed
over the same actions, mask and costs as the agent. Raw outputs are in
[results/](results/); `./run_experiments.sh` retrains every model and
`results/run_task3.sh` reruns the curriculum evaluation.

| Run (`models/experiments/`) | Setting | Steps |
|---|---|---|
| `full_charge`   | always charge to 100 % (old behaviour), routes of 4–12 | 1M |
| `partial_2m`    | partial charging (5 levels), routes of 4–12 — **default model** | 2M |
| `curriculum`    | `large_map.json`, curriculum 5 → 15 → 30 (500K each) | 1.5M |
| `no_curriculum` | `large_map.json`, routes of 10–30 from the start | 1.5M |
| `time_windows`  | partial charging + generated delivery windows | 1M |

## 1. Partial charging

Equal training budget (1M steps each), same 100 test routes of 4–12 customers:

| | Agent cost | Agent gap | kWh bought but unused | Optimum | Best baseline |
|---|---|---|---|---|---|
| Always charge to 100 % | $31.48 | 24.9 % | 26.1 per route | $24.88 | $35.33 |
| Partial charging       | **$26.51** | **16.8 %** | **0.0** | $22.28 | $30.61 |

- The agent's cost drops **15.8 %** and it no longer pays for energy it never
  uses (26 kWh per route before). Even the *optimal* full-charge plan wasted
  15.5 kWh per route, so this was a limit of the action set, not of the agent.
- The cheapest possible cost also drops (optimum $24.88 → $22.28, −10.4 %),
  and the agent gets closer to it.
- How: each station offers 5 charge targets between "least that keeps the
  route feasible" and "full, or exactly enough to finish" (see PLAN.md).
  The agent uses the low levels to top up at expensive stations and the high
  levels at cheap ones: it makes slightly more stops (3.7 vs 3.5) but pays
  $0.100/kWh against $0.114 for the best rule-based baseline.

## 2. Distance from the optimum (small routes)

Default model (`partial_2m`), 100 test routes per size:

| Customers | Optimum | PPO agent | Gap | Routes solved exactly | Smart rule-based | Charge when forced |
|---|---|---|---|---|---|---|
| 5  | $12.23 | $13.51 | **10.2 %** | 18 / 100 | 22.0 % | 61.4 % |
| 10 | $28.57 | $34.61 | 20.8 % | 1 / 100 | 41.0 % | 104.2 % |
| 15 | $44.86 | $55.70 | 24.1 % | 0 / 100 | 47.9 % | 114.5 % |
| 4–12 mixed | $22.28 | $26.30 | 16.0 % | 8 / 100 | 34.7 % | 85.5 % |

- The agent is about **half as far from the optimum as the best heuristic**
  at every size, but the gap grows with route length.
- The solver is a depth-first branch and bound with an admissible lower bound
  (see `utils/optimal.py`); it matches plain enumeration on every test route
  and proves optimality in ~0.01 s at 5 customers, ~0.3 s at 10 and a few
  seconds at 15. At 30 customers it does not finish in 2 minutes, so 30-customer
  results below have no optimum.

## 3. Curriculum 5 → 15 → 30

`examples/large_map.json` (the sample map + 15 more customers), 50 test
routes per size:

| Test route size | Curriculum | No curriculum | Optimum | Smart rule-based |
|---|---|---|---|---|
| 5  | $15.41 (gap 18.1 %) | $15.35 (gap 19.0 %) | $12.81 | $15.96 |
| 15 | $63.62 (gap 30.9 %) | $64.89 (gap 33.8 %) | $48.65 | $72.85 |
| 30 | $132.47 | $132.53 | — | $155.02 |

- **Both reach the same final quality**; the curriculum is marginally better
  at 15 customers (−2 %) and equal at 5 and 30. Both beat every baseline at
  every size, including 30-customer routes (−15 % vs smart rule-based).
- The reason is that the observation does not depend on route length (it
  describes the next leg, the remaining energy and the stations), so what is
  learned on short routes already transfers: the curriculum model's **first**
  evaluation on 10–30-customer routes, after training only on ≤15, was
  $76.32 — already the converged level (no-curriculum plateaued at ~$76 too).
- Learning curves: [results/task3_learning_curves.png](results/task3_learning_curves.png).
  Every stage was flat by its second half.
- Conclusion: the curriculum works and is kept as an option
  (`--curriculum 5,15,30`), but with this state design it is not needed;
  training directly on mixed lengths is as good.

## 4. Delivery time windows

Each customer gets a window [earliest, latest] (hours after departure).
Deadlines are hard: a backward table of "latest arrival time per battery
level" masks every action after which some later deadline could no longer be
met (charging time included). Training draws new windows each episode from a
feasible plan + 2–24 h slack. 100 test routes of 4–12 customers:

| | Cost | Gap | Late customers | $/kWh |
|---|---|---|---|---|
| Agent trained **with** windows    | **$30.93** | **25.3 %** | 0 | 0.137 |
| Agent trained **without** windows | $34.78 | 41.8 % | 0 | 0.165 |
| Optimum                            | $23.82 | 0 % | 0 | 0.092 |
| Smart rule-based                   | $35.17 | 41.8 % | 0 | 0.169 |

- **No customer was ever late** — for any method, because of the mask
  (also checked on 200 random-policy routes of 5–30 customers in the tests).
- Windows make cheap charging harder (fewer detours fit), so costs rise from
  $26.30 to $30.93. Training with windows matters: the model trained without
  them falls back to roughly the smart-heuristic level.
- On the user route with windows (`examples/sample_map_tw.json`): agent
  $26.29, optimum $19.49, every baseline $39.27.

## 5. Longer training (1–2M steps)

Mean held-out cost (50 fixed eval routes, logged every 25K steps) by quarter
of training — [results/task5_learning_curve.png](results/task5_learning_curve.png):

| Run | Q1 | Q2 | Q3 | Q4 | Q4 vs Q3 |
|---|---|---|---|---|---|
| Partial charging, 2M | $24.25 | $22.15 | $22.02 | $22.04 | −0.1 % (flat) |
| Always full, 1M      | $29.67 | $26.98 | $25.50 | $25.17 | +1.3 % (still improving slowly) |
| Time windows, 1M     | $27.07 | $25.50 | $24.09 | $23.99 | +0.4 % (flat) |

- **Costs do not drop further after ~700K steps.** On the 100 test routes
  the partial-charging agent costs $26.51 at 1M steps and $26.30 at 2M
  (−0.8 %); the gap to the optimum moves from 16.8 % to 16.0 %.
- So the remaining 10–24 % gap is not a training-length problem. Likely
  levers, in order: (a) more information in the state — prices beyond the
  4-hour lookahead and the stations near the *following* legs, since good
  charging decisions depend on where cheap energy will be later; (b) a
  learning-rate / entropy schedule for the late phase; (c) a larger network.

## Reproduce
```bash
./run_experiments.sh                 # ~1 h on 4 cores, one run per core
python learning_curve.py models/experiments/full_charge models/experiments/partial_2m \
    --labels "always full (1M),partial (2M)" --out results/task5_learning_curve.png
python evaluate.py --map examples/sample_map.json --random-orders \
    --min-visit 5 --max-visit 5 --episodes 100 --optimal
results/run_task3.sh
python tests/test_core.py
```
