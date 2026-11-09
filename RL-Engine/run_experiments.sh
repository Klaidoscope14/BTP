#!/usr/bin/env bash
# Reproduces the experiments for the five future tasks (see EXPERIMENTS.md).
# Runs in parallel, one CPU core each; logs go to logs/.
set -e
cd "$(dirname "$0")"
mkdir -p logs models/experiments
E=models/experiments

# Task 1 control: legacy "always charge to 100 %"
python train.py --map examples/sample_map.json --min-visit 4 --max-visit 12 \
  --charge-levels 0 --steps 1000000 --save-dir $E/full_charge > logs/full_charge.log 2>&1 &
# Tasks 1, 2, 5: partial charging, long run (2M steps) -> default model
python train.py --map examples/sample_map.json --min-visit 4 --max-visit 12 \
  --charge-levels 5 --steps 2000000 --save-dir $E/partial_2m > logs/partial_2m.log 2>&1 &
# Task 3: curriculum 5 -> 15 -> 30 customers, and a no-curriculum control
python train.py --map examples/large_map.json --curriculum 5,15,30 \
  --steps 1500000 --save-dir $E/curriculum > logs/curriculum.log 2>&1 &
python train.py --map examples/large_map.json --min-visit 10 --max-visit 30 \
  --steps 1500000 --save-dir $E/no_curriculum > logs/no_curriculum.log 2>&1 &
# Task 4: delivery time windows
python train.py --map examples/sample_map.json --min-visit 4 --max-visit 12 \
  --time-windows generated --steps 1000000 --save-dir $E/time_windows > logs/time_windows.log 2>&1 &
wait
echo "all experiments finished"
