# EVRP RL Engine — fixed-order charging optimizer

The vehicle must visit a user-given list of customers **in that exact order**
(not necessarily every customer on the map), on a map whose distances and
station prices are fully known. A reinforcement-learning agent (MaskablePPO)
decides, for each leg, whether to drive straight to the next customer or to
detour to a charging station, **and how much to charge there**, so that the
total of distance cost, charging cost and late-delivery cost is as low as
possible. See [PLAN.md](PLAN.md) for the full MDP and
[EXPERIMENTS.md](EXPERIMENTS.md) for results.

## Setup
```bash
pip install -r requirements.txt
```
The scripts can be run from any directory.

## 1. Describe the map and route
A map is a JSON file — see [examples/sample_map.json](examples/sample_map.json):

```json
{
  "depot":     {"name": "Depot", "x": 0, "y": 0},
  "customers": [{"name": "C01", "x": 12.5, "y": -3.0, "window": [0, 40]}, ...],
  "stations":  [{"name": "S1", "x": 20, "y": 5, "prices": [24 hourly $/kWh]}, ...],
  "visit_order": ["C02", "C03", "C05", "C13"],
  "params": {"battery_capacity": 100, "consumption_per_dist": 0.6}
}
```
- `visit_order` is the fixed order of customers to visit (any subset).
- `window` (optional) is the delivery time window `[earliest, latest]` in hours
  after leaving the depot — see [examples/sample_map_tw.json](examples/sample_map_tw.json).
- `prices` is optional; stations without it get a random schedule from `price_seed`.
- `params` is optional; see `WORLD_PARAMS` in [utils/world.py](utils/world.py)
  (includes `late_penalty_per_hour` and `late_penalty_fixed`).
- [examples/large_map.json](examples/large_map.json) is the sample map with 30
  customers, for curriculum training up to 30-customer routes.

## 2. Train on the known map
```bash
python train.py --map examples/sample_map.json --min-visit 4 --max-visit 12 --steps 500000
```
Each training episode uses the same map with a random visit order, so the
agent learns to handle whatever order a user gives it later.

| Option | What it does |
|---|---|
| `--charge-levels 5` | Partial charging: 5 charge levels per station (default). `0` = always charge to 100 % (old behaviour) |
| `--time-windows generated` | Every episode gets random delivery windows that the route can meet |
| `--curriculum 5,15,30` | Train on routes of ≤5 customers, then ≤15, then ≤30 (needs a map with ≥30 customers) |
| `--steps 2000000` | Long runs; the held-out cost every `--eval-freq` steps goes to `cost_curve.csv` |

The model, `vec_normalize.pkl`, the best model (`best_model.zip` +
`vec_normalize_best.pkl`), `cost_curve.csv` and `train_config.json` go to
`--save-dir` (default `models/saved_weights/`).

## 3. Run the user's route
```bash
python evaluate.py --map examples/sample_map.json
python evaluate.py --map examples/sample_map.json --order C05,C12,C03,C08 --plot outputs/
python evaluate.py --map examples/sample_map_tw.json --optimal      # with time windows
```
Prints the cost and the leg-by-leg log for the agent, every baseline and
(with `--optimal`) the exact optimum. `--plot` saves route maps and the price
heatmap. The env settings the model was trained with (charge levels, time
windows, lookahead) are read from `train_config.json` next to the model.

## 4. Benchmark
```bash
python evaluate.py --map examples/sample_map.json --random-orders --episodes 100
python evaluate.py --map examples/sample_map.json --random-orders \
    --min-visit 5 --max-visit 5 --episodes 100 --optimal          # optimality gap
python evaluate.py --nc 10 --ns 6 --episodes 100     # random maps (train without --map first)
```
Test episodes use seeds the training never sees. Columns: average cost, $/kWh
paid, charging stops, kWh bought but never used, late customers (with time
windows), and with `--optimal` the average gap to the optimum.

## 5. Learning curves
```bash
python learning_curve.py models/saved_weights --out outputs/curve.png
```
Prints the mean held-out cost per quarter of training and whether it was
still dropping at the end.

## 6. Reproduce all experiments and tests
```bash
./run_experiments.sh          # trains the runs in models/experiments/ (≈1 h on 4 cores)
python tests/test_core.py
```

A model only works on maps with the same number of stations, `--lookahead`
and `--charge-levels` it was trained with; any number of customers is fine.
