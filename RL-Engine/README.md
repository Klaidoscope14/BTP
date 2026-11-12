# EVRP RL Engine — fixed-order charging optimizer

The vehicle must visit a user-given list of customers **in that exact order**
(not necessarily every customer on the map), on a map whose distances and
station prices are fully known. A reinforcement-learning agent (MaskablePPO)
decides, for each leg, whether to drive straight to the next customer or
detour to a charging station first — and **how much to charge** there — so
that total distance and charging cost are as low as possible, optionally
while meeting each customer's **delivery time window**. An **exact solver**
gives the true optimum on small routes, to measure how close the agent gets.
See [PLAN.md](PLAN.md) for the full MDP.

## Setup
```bash
pip install -r requirements.txt
python -m pytest tests -q        # sanity tests, no training needed
```
The scripts can be run from any directory.

## 1. Describe the map and route
A map is a JSON file — see [examples/sample_map.json](examples/sample_map.json):

```json
{
  "depot":     {"name": "Depot", "x": 0, "y": 0},
  "customers": [{"name": "C01", "x": 12.5, "y": -3.0, "window": [8.0, 20.5]}, ...],
  "stations":  [{"name": "S1", "x": 20, "y": 5, "prices": [24 hourly $/kWh]}, ...],
  "visit_order": ["C02", "C03", "C05", "C13"],
  "params": {"battery_capacity": 100, "consumption_per_dist": 0.6}
}
```
- `visit_order` is the fixed order of customers to visit (any subset).
- `window` is optional: `[earliest, latest]` in hours since 00:00 on day 0
  (the vehicle leaves at 06:00). Early arrivals wait; late ones are penalised
  ($5 + $2 per hour late by default). See [examples/sample_map_tw.json](examples/sample_map_tw.json).
- `prices` is optional; stations without it get a random schedule from `price_seed`.
- `params` is optional; see `WORLD_PARAMS` in [utils/world.py](utils/world.py).

[examples/large_map.json](examples/large_map.json) is the sample map grown to
40 customers (same stations and prices) for the 5 → 15 → 30 curriculum. Both
derived maps are rebuilt with `python examples/make_example_maps.py`.

## 2. Train on the known map
```bash
# the shipped model: curriculum 5 -> 15 -> 30 customers, windows on half the routes
python train.py --map examples/large_map.json --curriculum 5,15,30 --steps 2000000 \
                --tw-prob 0.5 --envs 8 --subproc

# simpler: one stage, no windows
python train.py --map examples/sample_map.json --min-visit 4 --max-visit 12 --steps 500000
```
Each training episode uses the same map with a random visit order, so the
agent learns to handle whatever order a user gives it later. Useful options:
`--charge-levels 0,0.25,0.5,0.75,1` (charge targets; `1` = full-only),
`--stage-steps` (steps per curriculum stage), `--late-fixed` / `--late-per-hour`.
Checkpoints, `vec_normalize.pkl` and `train_config.json` go to
`models/saved_weights/` (each curriculum stage also gets a `stage<i>_<size>/` folder).

## 3. Run the user's route
```bash
python evaluate.py --map examples/sample_map.json --optimal
python evaluate.py --map examples/sample_map_tw.json --optimal      # with time windows
python evaluate.py --map examples/sample_map.json --order C05,C12,C03,C08 --plot outputs/
```
Prints the cost and the leg-by-leg log (charges, kWh, waits, late arrivals)
for the agent, every baseline and, with `--optimal`, the exact optimum and
each method's gap to it. `--plot` saves route maps and the price heatmap.
The env settings (look-ahead, charge levels, penalties) are read from the
model's `train_config.json`.

## 4. Benchmark
```bash
python evaluate.py --map examples/sample_map.json --random-orders --episodes 100 --optimal \
                   --min-visit 4 --max-visit 12
python evaluate.py --map examples/sample_map.json --random-orders --episodes 100 --tw-prob 1
python evaluate.py --map examples/large_map.json --random-orders --min-visit 25 --max-visit 30
python learning_curve.py --plot outputs/learning_curve.png     # cost vs training steps
```
Test episodes use seeds the training never sees. The exact solver is fast up
to ~12 customers per route; longer routes may hit its time limit
(`--optimal-time`), in which case the best route found is reported as such.

A model only works on maps with the same number of stations (and the same
`--lookahead` and `--charge-levels`) it was trained with; any number of
customers is fine. The pre-partial-charging model is kept in
`models/archive/v1_full_charge/` for reference; it does not run on the current env.
