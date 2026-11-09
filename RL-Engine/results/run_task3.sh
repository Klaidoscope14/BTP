#!/usr/bin/env bash
# Task 3: curriculum vs no curriculum on held-out routes of 5, 15 and 30 customers
cd "$(dirname "$0")/.."
for run in curriculum no_curriculum; do
  E=models/experiments/$run
  for n in 5 15 30; do
    opt=""; [ $n -le 15 ] && opt="--optimal"
    echo "### $run, $n customers"
    python evaluate.py --map examples/large_map.json --random-orders --min-visit $n --max-visit $n \
      --episodes 50 $opt --model $E/ppo_evrp_final.zip --norm $E/vec_normalize.pkl 2>&1 | sed -n '7,15p'
  done
done
