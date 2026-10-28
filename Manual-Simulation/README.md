# Small-World EVRP — Brute-Force Sanity Check

A deliberately tiny version of the full BTP problem, small enough to check
by hand, built to prove the "charge smart, not just when low" idea before
scaling up to reinforcement learning.

## World (fixed, in the script)
- **1 depot** used (`Warehouse-1`); a second warehouse exists in the world but
  is unused here — multi-depot assignment is out of scope for this check.
- **5 customers**, each must be visited exactly once, in any order.
- **4 charging stations**, each visited **at most once**, visiting is optional.
- Each station has its own **time-of-use price curve** (a sine wave — cheaper
  at some hours, pricier at others, exactly like the main project). Different
  stations peak/dip at different hours on purpose, so *when* you arrive
  somewhere changes what it costs — that's the arbitrage opportunity.

## What's deliberately left out (for now)
- **Hard delivery time windows** — kept out purely to keep the number of
  paths small enough to enumerate exhaustively in a few seconds and to
  check by hand. Everything else that matters for the arbitrage idea
  (battery limits + dynamic pricing) is kept in.
- **Partial charging** — every station visit tops the battery to 100%.
  Deciding *how much* to charge is a refinement for later, not needed to
  prove the core idea.
- Distances/speed/battery are abstract units, not real km or kWh — rescale
  the constants at the top of the script if you want realistic numbers.

## How the brute force works
For every possible **subset** of the 4 charging stations (none, one, two,
three, or all four) and every possible **order** of visiting (all 5
customers + that subset), the script actually drives the route
step-by-step, tracks battery and elapsed time, prices the charging events
at the moment of arrival, and throws out any route that would run the
battery below zero. It keeps the cheapest route that's actually feasible.
No shortcuts, no pruning heuristics — every candidate is genuinely evaluated.

With 5 customers + 4 optional stations that's **557,400 candidate paths**,
checked in about 5 seconds.

## How to check it by hand
1. Look at the printed coordinates for the nodes you care about and compute
   a couple of distances yourself with `sqrt((x2-x1)^2 + (y2-y1)^2)`.
2. Pick a short alternative route (e.g. skip a charging stop) and manually
   check whether the battery would actually survive it — if it goes
   negative, the script would have correctly rejected it too.
3. Compute the charging cost for one station visit by hand: look up its
   `base`, `amplitude`, `phase` in `STATION_PRICING`, work out
   `price = base + amplitude * sin(2*pi*(hour - phase)/24)` at the arrival
   time printed in the log, multiply by the kWh added.
4. Compare your hand-computed cost for a couple of paths against the
   script's printed optimum — they should match.

## Run it
```bash
pip install matplotlib   # only external dependency
python3 evrp_toy_bruteforce.py
```

Output includes: the optimal path, a full step-by-step log (distance,
battery, arrival time, price paid at each charge), a comparison against a
simple greedy "charge when battery is low" baseline, and a saved
`world_map.png` plotting both routes over the node layout.

## Sample result (seeded, deterministic)
The brute-force optimum came out **~40% cheaper** than the greedy baseline
in this layout — the optimal route deliberately visits stations at
different points in the day to catch each one's cheap window, while the
greedy baseline charges reactively and eats peak prices.

## Scaling this up
Growing to 6 customers + 5 stations is a ~100x jump in candidate paths
(roughly 62 million) — still solvable by brute force, but no longer a
few-second, by-hand-checkable exercise. That jump is itself worth
mentioning to your prof: it's a small, concrete illustration of *why*
brute force stops being viable at real-world scale and why the BTP
proposes learning a policy (DRL) instead of exhaustively searching.
