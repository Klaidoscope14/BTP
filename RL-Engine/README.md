# EVRP RL Engine — fixed-order charging optimizer

The vehicle must visit a user-given list of customers **in that exact order**
(not necessarily every customer on the map), on a map whose distances and
station prices are fully known. A reinforcement-learning agent (MaskablePPO)
decides, for each leg, whether to drive straight to the next customer or
detour to a charging station first, so that total distance and charging cost
are as low as possible. See [PLAN.md](PLAN.md) for the full MDP.

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
  "customers": [{"name": "C01", "x": 12.5, "y": -3.0}, ...],
  "stations":  [{"name": "S1", "x": 20, "y": 5, "prices": [24 hourly $/kWh]}, ...],
  "visit_order": ["C02", "C03", "C05", "C13"],
  "params": {"battery_capacity": 100, "consumption_per_dist": 0.6}
}
```
- `visit_order` is the fixed order of customers to visit (any subset).
- `prices` is optional; stations without it get a random schedule from `price_seed`.
- `params` is optional; see `WORLD_PARAMS` in [utils/world.py](utils/world.py).

## 2. Train on the known map
```bash
python train.py --map examples/sample_map.json --min-visit 4 --max-visit 12 --steps 500000
```
Each training episode uses the same map with a random visit order, so the
agent learns to handle whatever order a user gives it later. Checkpoints,
`vec_normalize.pkl` and `train_config.json` go to `models/saved_weights/`.

## 3. Run the user's route
```bash
python evaluate.py --map examples/sample_map.json
python evaluate.py --map examples/sample_map.json --order C05,C12,C03,C08 --plot outputs/
```
Prints the cost and the leg-by-leg log for the agent and every baseline;
`--plot` saves route maps and the price heatmap.

## 4. Benchmark
```bash
python evaluate.py --map examples/sample_map.json --random-orders --episodes 100
python evaluate.py --nc 10 --ns 6 --episodes 100     # random maps (train without --map first)
```
Test episodes use seeds the training never sees.

A model only works on maps with the same number of stations (and the same
`--lookahead`) it was trained with; any number of customers is fine.
