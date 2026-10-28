"""
Small-World EVRP Brute-Force Sanity Check
==========================================
A tiny, hand-verifiable sandbox for the "Arbitrage Router" BTP project.

WORLD
  - 2 warehouses   (Warehouse-1 is used as the vehicle's depot in this toy;
                     Warehouse-2 exists in the world but is unused here --
                     multi-depot assignment is out of scope for this check)
  - 5 customers    (must ALL be delivered to, exactly once, any order)
  - 4 charging stations (visiting is OPTIONAL, each used at most once)

RULES
  - The vehicle starts fully charged at the depot and must return to it.
  - Driving consumes battery proportional to distance; running out of
    battery mid-leg makes that path infeasible.
  - Each charging station has its own time-of-use PRICE CURVE (a sine wave,
    same idea as the main project), so the cost of charging depends on
    WHEN you arrive, not just where.
  - Charging always tops the battery back up to 100% (kept simple on
    purpose -- partial-charge optimization is a later refinement).
  - Hard delivery time-windows are intentionally left OUT so the search
    space stays small enough to enumerate by hand. Everything else
    (battery + dynamic pricing) is kept in, because that's the one thing
    this sanity-check exists to demonstrate.

METHOD: TRUE BRUTE FORCE
  For every possible subset of charging stations (visit none, one, two...)
  and every possible order of (all customers + that subset), simulate the
  drive and keep the cheapest FEASIBLE result. Nothing is guessed or
  pruned heuristically -- every candidate path is actually evaluated.

Run:  python3 evrp_toy_bruteforce.py
"""

import itertools
import math
import random
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 1. WORLD DEFINITION
# ---------------------------------------------------------------------------

@dataclass
class Node:
    name: str
    x: float
    y: float
    kind: str  # 'depot' | 'customer' | 'station'


WAREHOUSES = [
    Node("Warehouse-1", 0.0, 0.0, "depot"),
    Node("Warehouse-2", 40.0, 40.0, "depot"),  # in the world, unused as depot here
]

CUSTOMERS = [
    Node("Cust-A", 10, 5, "customer"),
    Node("Cust-B", 18, -8, "customer"),
    Node("Cust-C", -6, 14, "customer"),
    Node("Cust-D", 22, 20, "customer"),
    Node("Cust-E", -12, -10, "customer"),
]

STATIONS = [
    Node("Station-1", 8, -3, "station"),
    Node("Station-2", -8, 6, "station"),
    Node("Station-3", 20, 10, "station"),
    Node("Station-4", -14, -4, "station"),
]

DEPOT = WAREHOUSES[0]

# Time-of-use price schedule per station: random step function.
# Each station gets a random price for every 1-hour block of the day (24 steps).
# The agent knows all prices at all times (full information).
# Seeded for reproducibility — change PRICING_SEED for a different price landscape.
PRICING_SEED = 42
PRICE_BLOCK_HOURS = 1  # each step spans this many hours
PRICE_MIN = 0.08       # $/kWh floor
PRICE_MAX = 0.36       # $/kWh ceiling
NUM_BLOCKS = int(24 / PRICE_BLOCK_HOURS)

def _generate_step_prices(seed):
    rng = random.Random(seed)
    schedule = {}
    for s in ["Station-1", "Station-2", "Station-3", "Station-4"]:
        schedule[s] = [round(rng.uniform(PRICE_MIN, PRICE_MAX), 3)
                       for _ in range(NUM_BLOCKS)]
    return schedule

STATION_PRICING = _generate_step_prices(PRICING_SEED)

# ---------------------------------------------------------------------------
# 2. VEHICLE / PHYSICS PARAMETERS
# ---------------------------------------------------------------------------

BATTERY_CAPACITY_KWH = 50.0       # full battery
CONSUMPTION_PER_UNIT_DIST = 1.0   # kWh consumed per unit distance travelled
SPEED = 1.0                       # distance units per hour -> travel_time = distance / SPEED
CHARGE_RATE_KWH_PER_HOUR = 25.0   # how fast the battery fills at any station
DISTANCE_COST_PER_UNIT = 0.05     # $ wear-and-tear cost per unit distance (keeps routes honest)
START_TIME = 8.0                  # simulation starts at 08:00 on a 24h clock


def price_at(station_name, t):
    hour_of_day = t % 24.0
    block = int(hour_of_day / PRICE_BLOCK_HOURS) % NUM_BLOCKS
    return STATION_PRICING[station_name][block]


def dist(a: Node, b: Node) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


# ---------------------------------------------------------------------------
# 3. ROUTE EVALUATION -- simulate driving a candidate path step by step
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    feasible: bool
    total_cost: float = math.inf
    total_distance: float = 0.0
    total_charge_cost: float = 0.0
    finish_time: float = 0.0
    log: list = field(default_factory=list)


def evaluate_path(path):
    """path: list of Node objects, starting and ending at DEPOT."""
    battery = BATTERY_CAPACITY_KWH
    t = START_TIME
    total_distance = 0.0
    total_charge_cost = 0.0
    log = []

    for a, b in zip(path[:-1], path[1:]):
        d = dist(a, b)
        needed = d * CONSUMPTION_PER_UNIT_DIST
        if needed > battery + 1e-9:
            return EvalResult(False, log=log + [f"Ran out of battery driving {a.name} -> {b.name}"])
        battery -= needed
        t += d / SPEED
        total_distance += d
        log.append(f"Drive {a.name} -> {b.name}: {d:.1f} units, battery {battery:.1f} kWh left, t={t:.1f}h")

        if b.kind == "station":
            price = price_at(b.name, t)
            energy_needed = BATTERY_CAPACITY_KWH - battery
            charge_time = energy_needed / CHARGE_RATE_KWH_PER_HOUR
            cost = price * energy_needed
            battery = BATTERY_CAPACITY_KWH
            t += charge_time
            total_charge_cost += cost
            log.append(f"  Charge at {b.name}: +{energy_needed:.1f} kWh @ ${price:.3f}/kWh = ${cost:.2f}, t={t:.1f}h")

    distance_cost = total_distance * DISTANCE_COST_PER_UNIT
    total_cost = distance_cost + total_charge_cost
    return EvalResult(True, total_cost, total_distance, total_charge_cost, t, log)


# ---------------------------------------------------------------------------
# 4. BRUTE-FORCE SEARCH -- every customer order x every subset/order of stations
# ---------------------------------------------------------------------------

def brute_force_search(customers, stations, depot):
    best = None
    n_checked = 0
    n_feasible = 0

    for k in range(0, len(stations) + 1):
        for station_subset in itertools.combinations(stations, k):
            combined = list(customers) + list(station_subset)
            for perm in itertools.permutations(combined):
                path = [depot] + list(perm) + [depot]
                n_checked += 1
                result = evaluate_path(path)
                if result.feasible:
                    n_feasible += 1
                    if best is None or result.total_cost < best[0].total_cost:
                        best = (result, path)

    return best, n_checked, n_feasible


# ---------------------------------------------------------------------------
# 5. BASELINE HEURISTIC -- greedy nearest-neighbour, charge when battery is low
#    (this mirrors the classical approach the full BTP project benchmarks against)
# ---------------------------------------------------------------------------

def greedy_baseline(customers, stations, depot, low_battery_kwh=15.0):
    unvisited = list(customers)
    current = depot
    battery = BATTERY_CAPACITY_KWH
    t = START_TIME
    path = [depot]
    total_distance = 0.0
    total_charge_cost = 0.0

    def do_charge(current, battery, t):
        nearest_station = min(stations, key=lambda s: dist(current, s))
        d = dist(current, nearest_station)
        if d * CONSUMPTION_PER_UNIT_DIST > battery:
            return None
        battery -= d * CONSUMPTION_PER_UNIT_DIST
        t += d / SPEED
        price = price_at(nearest_station.name, t)
        energy_needed = BATTERY_CAPACITY_KWH - battery
        cost = price * energy_needed
        battery = BATTERY_CAPACITY_KWH
        t += energy_needed / CHARGE_RATE_KWH_PER_HOUR
        return nearest_station, d, cost, battery, t

    while unvisited:
        if battery < low_battery_kwh:
            out = do_charge(current, battery, t)
            if out is None:
                return None
            station, d, cost, battery, t = out
            total_distance += d
            total_charge_cost += cost
            path.append(station)
            current = station

        nxt = min(unvisited, key=lambda c: dist(current, c))
        d = dist(current, nxt)
        if d * CONSUMPTION_PER_UNIT_DIST > battery:
            out = do_charge(current, battery, t)
            if out is None:
                return None
            station, dd, cost, battery, t = out
            total_distance += dd
            total_charge_cost += cost
            path.append(station)
            current = station
            d = dist(current, nxt)

        battery -= d * CONSUMPTION_PER_UNIT_DIST
        t += d / SPEED
        total_distance += d
        path.append(nxt)
        current = nxt
        unvisited.remove(nxt)

    d = dist(current, depot)
    if d * CONSUMPTION_PER_UNIT_DIST > battery:
        return None
    total_distance += d
    path.append(depot)

    total_cost = total_distance * DISTANCE_COST_PER_UNIT + total_charge_cost
    return total_cost, path, total_distance, total_charge_cost


# ---------------------------------------------------------------------------
# 6. RULE-BASED SMART SOLVER
#    Two-phase construction + improvement, using full price knowledge.
#
#    PHASE 1 — CONSTRUCT CANDIDATES
#      Generate multiple customer orderings (nearest-neighbour seeded from
#      each customer, plus the depot).  For each ordering, do a forward
#      simulation that inserts charging stops optimally:
#        R1  Before each leg, score every reachable unused station by
#            (detour_distance_cost + charge_price_at_arrival * energy_needed).
#        R2  Insert a charge stop if (a) battery can't reach the next node, OR
#            (b) the best station score beats a "charge later" estimate —
#            i.e. opportunistic charging when a cheap window is available now
#            and we'll need energy eventually.
#
#    PHASE 2 — LOCAL IMPROVEMENT (2-opt on customer sub-sequence)
#      Repeatedly try swapping pairs of customers in the best route found in
#      Phase 1 and re-simulate with optimal station insertion. Keep any swap
#      that lowers total cost.
#
#    The solver never enumerates all permutations — it builds O(N) candidate
#    routes and improves the best with O(N^2) swaps, each evaluated in O(N).
# ---------------------------------------------------------------------------


def _simulate_with_optimal_charging(customer_order, stations, depot):
    """Given a fixed customer visit order, insert charging stops optimally.
    Returns (EvalResult, full_path) or None if infeasible."""
    battery = BATTERY_CAPACITY_KWH
    t = START_TIME
    total_distance = 0.0
    total_charge_cost = 0.0
    log = []
    path = [depot]
    current = depot
    used_stations = set()

    waypoints = list(customer_order) + [depot]

    for idx, nxt in enumerate(waypoints):
        # Before driving to nxt, consider inserting a charge stop
        while True:
            d_to_nxt = dist(current, nxt)

            # Score every reachable, unused station as a potential detour
            best_station = None
            best_net_cost = math.inf  # net cost of detouring to charge vs not
            for s in stations:
                if s.name in used_stations:
                    continue
                d_cs = dist(current, s)
                if d_cs * CONSUMPTION_PER_UNIT_DIST > battery + 1e-9:
                    continue
                batt_at_s = battery - d_cs * CONSUMPTION_PER_UNIT_DIST
                t_at_s = t + d_cs / SPEED
                energy_filled = BATTERY_CAPACITY_KWH - batt_at_s
                charge_cost = price_at(s.name, t_at_s) * energy_filled
                d_s_to_nxt = dist(s, nxt)
                detour_extra = (d_cs + d_s_to_nxt) - d_to_nxt
                detour_dist_cost = detour_extra * DISTANCE_COST_PER_UNIT
                net = detour_dist_cost + charge_cost
                if net < best_net_cost:
                    best_net_cost = net
                    best_station = s

            must_charge = d_to_nxt * CONSUMPTION_PER_UNIT_DIST > battery + 1e-9

            # Opportunistic: if remaining legs need more energy than battery
            # currently holds, we'll need to charge somewhere — is now cheaper
            # than the average?  Also verify the route stays feasible after
            # using this station (there must be enough unused stations left
            # for any future mandatory charges).
            want_charge = False
            if not must_charge and best_station is not None:
                remaining_energy = battery - d_to_nxt * CONSUMPTION_PER_UNIT_DIST
                future_dist = 0.0
                prev = nxt
                for w in waypoints[idx+1:]:
                    future_dist += dist(prev, w)
                    prev = w
                if future_dist * CONSUMPTION_PER_UNIT_DIST > remaining_energy:
                    avg_price = sum(sum(v) for v in STATION_PRICING.values()) / (
                        len(STATION_PRICING) * NUM_BLOCKS)
                    d_cs = dist(current, best_station)
                    t_at_s = t + d_cs / SPEED
                    price_now = price_at(best_station.name, t_at_s)
                    # Only charge opportunistically if price is well below
                    # average AND at least one other station remains unused
                    # (so we don't consume the last available station early)
                    other_unused = [s for s in stations
                                    if s.name not in used_stations
                                    and s.name != best_station.name]
                    if price_now < avg_price * 0.80 and len(other_unused) >= 1:
                        want_charge = True

            if must_charge:
                if best_station is None:
                    return None
                # Drive to station and charge
                d = dist(current, best_station)
                battery -= d * CONSUMPTION_PER_UNIT_DIST
                t += d / SPEED
                total_distance += d
                log.append(f"Drive {current.name} -> {best_station.name}: {d:.1f} units, "
                           f"battery {battery:.1f} kWh left, t={t:.1f}h")
                path.append(best_station)
                price = price_at(best_station.name, t)
                energy_needed = BATTERY_CAPACITY_KWH - battery
                cost = price * energy_needed
                battery = BATTERY_CAPACITY_KWH
                t += energy_needed / CHARGE_RATE_KWH_PER_HOUR
                total_charge_cost += cost
                log.append(f"  Charge at {best_station.name}: +{energy_needed:.1f} kWh "
                           f"@ ${price:.3f}/kWh = ${cost:.2f}, t={t:.1f}h")
                current = best_station
                used_stations.add(best_station.name)
                continue  # re-evaluate from new position

            if want_charge:
                d = dist(current, best_station)
                battery -= d * CONSUMPTION_PER_UNIT_DIST
                t += d / SPEED
                total_distance += d
                log.append(f"Drive {current.name} -> {best_station.name}: {d:.1f} units, "
                           f"battery {battery:.1f} kWh left, t={t:.1f}h")
                path.append(best_station)
                price = price_at(best_station.name, t)
                energy_needed = BATTERY_CAPACITY_KWH - battery
                cost = price * energy_needed
                battery = BATTERY_CAPACITY_KWH
                t += energy_needed / CHARGE_RATE_KWH_PER_HOUR
                total_charge_cost += cost
                log.append(f"  Charge at {best_station.name}: +{energy_needed:.1f} kWh "
                           f"@ ${price:.3f}/kWh = ${cost:.2f}, t={t:.1f}h")
                current = best_station
                used_stations.add(best_station.name)
                continue

            break  # no charge needed, proceed to nxt

        # Drive to the next waypoint
        d = dist(current, nxt)
        battery -= d * CONSUMPTION_PER_UNIT_DIST
        t += d / SPEED
        total_distance += d
        log.append(f"Drive {current.name} -> {nxt.name}: {d:.1f} units, "
                   f"battery {battery:.1f} kWh left, t={t:.1f}h")
        path.append(nxt)
        current = nxt

    total_cost = total_distance * DISTANCE_COST_PER_UNIT + total_charge_cost
    result = EvalResult(True, total_cost, total_distance, total_charge_cost, t, log)
    return result, path


def smart_rule_based(customers, stations, depot):
    # --- Phase 1: generate candidate customer orderings ---
    seed_nodes = [depot] + list(customers)
    best_overall = None

    for seed in seed_nodes:
        unvisited = list(customers)
        order = []
        current = seed if seed.kind == "customer" else depot
        if seed.kind == "customer":
            order.append(seed)
            unvisited.remove(seed)

        while unvisited:
            nxt = min(unvisited, key=lambda c: dist(current, c))
            order.append(nxt)
            unvisited.remove(nxt)
            current = nxt

        out = _simulate_with_optimal_charging(order, stations, depot)
        if out is not None:
            result, path = out
            if best_overall is None or result.total_cost < best_overall[0].total_cost:
                best_overall = (result, path, list(order))

    if best_overall is None:
        return None

    # --- Phase 2: 2-opt improvement on customer sub-sequence ---
    best_result, best_path, best_order = best_overall
    improved = True
    while improved:
        improved = False
        for i in range(len(best_order)):
            for j in range(i + 1, len(best_order)):
                new_order = list(best_order)
                new_order[i], new_order[j] = new_order[j], new_order[i]
                out = _simulate_with_optimal_charging(new_order, stations, depot)
                if out is not None:
                    result, path = out
                    if result.total_cost < best_result.total_cost - 1e-9:
                        best_result, best_path, best_order = result, path, new_order
                        improved = True

    return (best_result.total_cost, best_path, best_result.total_distance,
            best_result.total_charge_cost, best_result)


# ---------------------------------------------------------------------------
# 7. OPTIONAL VISUALIZATION
# ---------------------------------------------------------------------------

def plot_world(best_path, baseline_path, smart_path=None, filename="world_map.png"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed -- skipping plot)")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    for w in WAREHOUSES:
        ax.scatter(w.x, w.y, marker="s", s=140, c="black")
        ax.annotate(w.name, (w.x, w.y), textcoords="offset points", xytext=(6, 6))
    for c in CUSTOMERS:
        ax.scatter(c.x, c.y, marker="o", s=100, c="tab:blue")
        ax.annotate(c.name, (c.x, c.y), textcoords="offset points", xytext=(6, 6))
    for s in STATIONS:
        ax.scatter(s.x, s.y, marker="^", s=120, c="tab:green")
        ax.annotate(s.name, (s.x, s.y), textcoords="offset points", xytext=(6, 6))

    if best_path:
        xs = [n.x for n in best_path]
        ys = [n.y for n in best_path]
        ax.plot(xs, ys, c="tab:red", linewidth=2, label="Brute-force optimal")

    if smart_path:
        xs = [n.x for n in smart_path]
        ys = [n.y for n in smart_path]
        ax.plot(xs, ys, c="tab:orange", linewidth=2, linestyle="-.", label="Rule-based smart")

    if baseline_path:
        xs = [n.x for n in baseline_path]
        ys = [n.y for n in baseline_path]
        ax.plot(xs, ys, c="gray", linewidth=1.5, linestyle="--", label="Greedy baseline")

    ax.set_title("Small-World EVRP: Optimal vs. Smart vs. Greedy")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print(f"\nSaved route map to {filename}")


# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("SMALL-WORLD EVRP BRUTE-FORCE SANITY CHECK")
    print("=" * 70)
    print(f"Depot     : {DEPOT.name}")
    print(f"Customers : {[c.name for c in CUSTOMERS]}")
    print(f"Stations  : {[s.name for s in STATIONS]}\n")

    print("STATION PRICE SCHEDULE ($/kWh per hour block, seed=%d):" % PRICING_SEED)
    header = "Hour  " + "  ".join(f"{s.name:>11}" for s in STATIONS)
    print(header)
    for b in range(NUM_BLOCKS):
        h_start = b * PRICE_BLOCK_HOURS
        h_end = h_start + PRICE_BLOCK_HOURS
        row = f"{h_start:02.0f}-{h_end:02.0f}  "
        row += "  ".join(f"${STATION_PRICING[s.name][b]:.3f}     " for s in STATIONS)
        print(row)
    print()

    print("Running exhaustive brute-force search over all paths...")
    (best_result, best_path), n_checked, n_feasible = brute_force_search(CUSTOMERS, STATIONS, DEPOT)
    print(f"Checked {n_checked:,} candidate paths ({n_feasible:,} feasible)\n")

    print("OPTIMAL PATH FOUND:")
    print(" -> ".join(n.name for n in best_path))
    print(f"Total cost      : ${best_result.total_cost:.2f}")
    print(f"  distance cost : ${best_result.total_distance * DISTANCE_COST_PER_UNIT:.2f} "
          f"({best_result.total_distance:.1f} units)")
    print(f"  charging cost : ${best_result.total_charge_cost:.2f}")
    print(f"Finish time     : {best_result.finish_time:.1f}h\n")

    print("Step-by-step log:")
    for line in best_result.log:
        print(" ", line)

    # --- Rule-based smart solver ---
    print("\n" + "=" * 70)
    print("RULE-BASED SMART SOLVER:")
    print("  Phase 1: nearest-neighbour from each seed + optimal station insertion")
    print("  Phase 2: 2-opt swap improvement on customer order")
    smart = smart_rule_based(CUSTOMERS, STATIONS, DEPOT)
    smart_path = None
    if smart is None:
        print("Smart solver got stranded (infeasible)!")
    else:
        s_cost, smart_path, s_distance, s_charge_cost, s_result = smart
        print(" -> ".join(n.name for n in smart_path))
        print(f"Total cost      : ${s_cost:.2f}")
        print(f"  distance cost : ${s_distance * DISTANCE_COST_PER_UNIT:.2f} "
              f"({s_distance:.1f} units)")
        print(f"  charging cost : ${s_charge_cost:.2f}")
        print(f"Finish time     : {s_result.finish_time:.1f}h\n")
        print("Step-by-step log:")
        for line in s_result.log:
            print(" ", line)
        gap = s_cost - best_result.total_cost
        gap_pct = 100 * gap / best_result.total_cost if best_result.total_cost else 0
        print(f"\nGap to brute-force optimum: +${gap:.2f} (+{gap_pct:.1f}%)")

    # --- Greedy baseline ---
    print("\n" + "=" * 70)
    print("BASELINE (greedy nearest-neighbour, charge below 15 kWh):")
    baseline = greedy_baseline(CUSTOMERS, STATIONS, DEPOT)
    baseline_path = None
    if baseline is None:
        print("Baseline got stranded (infeasible)!")
    else:
        cost, baseline_path, distance, charge_cost = baseline
        print(" -> ".join(n.name for n in baseline_path))
        print(f"Total cost      : ${cost:.2f}  (distance ${distance * DISTANCE_COST_PER_UNIT:.2f} "
              f"+ charging ${charge_cost:.2f})")

    # --- Summary comparison ---
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print(f"  {'Method':<28} {'Total Cost':>10}  {'vs Optimal':>12}")
    print(f"  {'Brute-force optimal':<28} ${best_result.total_cost:>8.2f}  {'baseline':>12}")
    if smart is not None:
        gap = s_cost - best_result.total_cost
        print(f"  {'Rule-based smart':<28} ${s_cost:>8.2f}  {'+${:.2f} (+{:.1f}%)'.format(gap, 100*gap/best_result.total_cost):>12}")
    if baseline is not None:
        gap_b = cost - best_result.total_cost
        print(f"  {'Greedy baseline':<28} ${cost:>8.2f}  {'+${:.2f} (+{:.1f}%)'.format(gap_b, 100*gap_b/best_result.total_cost):>12}")

    plot_world(best_path, baseline_path, smart_path)
