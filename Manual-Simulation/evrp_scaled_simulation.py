"""
Scaled EVRP Simulation
======================
A larger scenario (15 customers, 8 charging stations) where brute force is
impossible (~10^17 candidate paths).  Only the rule-based smart solver and
the greedy baseline run here — the brute-force optimal is not computable,
so we compare the two heuristics head-to-head.

Run:  python3 evrp_scaled_simulation.py
"""

import math
import random
import time
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 1. WORLD DEFINITION — randomly generated, seeded for reproducibility
# ---------------------------------------------------------------------------

@dataclass
class Node:
    name: str
    x: float
    y: float
    kind: str  # 'depot' | 'customer' | 'station'


def _generate_world(seed=123, num_customers=15, num_stations=8, grid_size=40.0):
    rng = random.Random(seed)
    depot = Node("Depot", 0.0, 0.0, "depot")
    customers = []
    for i in range(num_customers):
        x = rng.uniform(-grid_size, grid_size)
        y = rng.uniform(-grid_size, grid_size)
        customers.append(Node(f"C{i+1:02d}", round(x, 1), round(y, 1), "customer"))
    stations = []
    for i in range(num_stations):
        angle = 2 * math.pi * i / num_stations + rng.uniform(-0.3, 0.3)
        r = rng.uniform(grid_size * 0.3, grid_size * 0.85)
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        stations.append(Node(f"S{i+1}", round(x, 1), round(y, 1), "station"))
    return depot, customers, stations


WORLD_SEED = 123
DEPOT, CUSTOMERS, STATIONS = _generate_world(WORLD_SEED)

# ---------------------------------------------------------------------------
# 2. PRICING — random step function per station
# ---------------------------------------------------------------------------

PRICING_SEED = 99
PRICE_BLOCK_HOURS = 1
PRICE_MIN = 0.06
PRICE_MAX = 0.40
NUM_BLOCKS = 24

def _generate_step_prices(seed, station_names):
    rng = random.Random(seed)
    return {s: [round(rng.uniform(PRICE_MIN, PRICE_MAX), 3)
                for _ in range(NUM_BLOCKS)] for s in station_names}

STATION_PRICING = _generate_step_prices(PRICING_SEED,
                                         [s.name for s in STATIONS])

def price_at(station_name, t):
    hour_of_day = t % 24.0
    block = int(hour_of_day / PRICE_BLOCK_HOURS) % NUM_BLOCKS
    return STATION_PRICING[station_name][block]

# ---------------------------------------------------------------------------
# 3. VEHICLE PARAMETERS
# ---------------------------------------------------------------------------

BATTERY_CAPACITY_KWH = 75.0
CONSUMPTION_PER_UNIT_DIST = 0.8   # kWh per distance unit
SPEED = 1.5                       # distance units per hour
CHARGE_RATE_KWH_PER_HOUR = 30.0
DISTANCE_COST_PER_UNIT = 0.03     # $/unit wear-and-tear (lower so charging cost dominates)
START_TIME = 6.0                  # 06:00

def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

# ---------------------------------------------------------------------------
# 4. ROUTE EVALUATION
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    feasible: bool
    total_cost: float = math.inf
    total_distance: float = 0.0
    total_charge_cost: float = 0.0
    num_charges: int = 0
    finish_time: float = 0.0
    log: list = field(default_factory=list)


def evaluate_path(path, depot):
    battery = BATTERY_CAPACITY_KWH
    t = START_TIME
    total_distance = 0.0
    total_charge_cost = 0.0
    num_charges = 0
    log = []

    for a, b in zip(path[:-1], path[1:]):
        d = dist(a, b)
        needed = d * CONSUMPTION_PER_UNIT_DIST
        if needed > battery + 1e-9:
            return EvalResult(False,
                              log=log + [f"Ran out of battery {a.name}->{b.name}"])
        battery -= needed
        t += d / SPEED
        total_distance += d
        log.append(f"{a.name}->{b.name}: {d:.1f}u, bat={battery:.1f}, t={t:.1f}h")

        if b.kind == "station":
            price = price_at(b.name, t)
            energy = BATTERY_CAPACITY_KWH - battery
            charge_time = energy / CHARGE_RATE_KWH_PER_HOUR
            cost = price * energy
            battery = BATTERY_CAPACITY_KWH
            t += charge_time
            total_charge_cost += cost
            num_charges += 1
            log.append(f"  CHARGE {b.name}: +{energy:.1f}kWh @${price:.3f} "
                       f"= ${cost:.2f}, t={t:.1f}h")

    distance_cost = total_distance * DISTANCE_COST_PER_UNIT
    total_cost = distance_cost + total_charge_cost
    return EvalResult(True, total_cost, total_distance, total_charge_cost,
                      num_charges, t, log)

# ---------------------------------------------------------------------------
# 5. GREEDY BASELINE — nearest-neighbour, charge when low
# ---------------------------------------------------------------------------

def greedy_baseline(customers, stations, depot, low_battery_kwh=25.0):
    unvisited = list(customers)
    current = depot
    battery = BATTERY_CAPACITY_KWH
    t = START_TIME
    path = [depot]
    total_distance = 0.0
    total_charge_cost = 0.0
    num_charges = 0

    def do_charge(cur, bat, t_now):
        nearest = min(stations, key=lambda s: dist(cur, s))
        d = dist(cur, nearest)
        if d * CONSUMPTION_PER_UNIT_DIST > bat + 1e-9:
            return None
        bat -= d * CONSUMPTION_PER_UNIT_DIST
        t_now += d / SPEED
        p = price_at(nearest.name, t_now)
        energy = BATTERY_CAPACITY_KWH - bat
        cost = p * energy
        bat = BATTERY_CAPACITY_KWH
        t_now += energy / CHARGE_RATE_KWH_PER_HOUR
        return nearest, d, cost, bat, t_now

    while unvisited:
        if battery < low_battery_kwh:
            out = do_charge(current, battery, t)
            if out is None:
                return None
            station, d, cost, battery, t = out
            total_distance += d
            total_charge_cost += cost
            num_charges += 1
            path.append(station)
            current = station

        nxt = min(unvisited, key=lambda c: dist(current, c))
        d = dist(current, nxt)
        if d * CONSUMPTION_PER_UNIT_DIST > battery + 1e-9:
            out = do_charge(current, battery, t)
            if out is None:
                return None
            station, dd, cost, battery, t = out
            total_distance += dd
            total_charge_cost += cost
            num_charges += 1
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
    if d * CONSUMPTION_PER_UNIT_DIST > battery + 1e-9:
        out = do_charge(current, battery, t)
        if out is None:
            return None
        station, dd, cost, battery, t = out
        total_distance += dd
        total_charge_cost += cost
        num_charges += 1
        path.append(station)
        current = station

    d_home = dist(current, depot)
    total_distance += d_home
    path.append(depot)

    total_cost = total_distance * DISTANCE_COST_PER_UNIT + total_charge_cost
    result = EvalResult(True, total_cost, total_distance, total_charge_cost,
                        num_charges, t + d_home / SPEED, [])
    return total_cost, path, result

# ---------------------------------------------------------------------------
# 6. RULE-BASED SMART SOLVER (same two-phase design as the toy version)
# ---------------------------------------------------------------------------

def _simulate_with_optimal_charging(customer_order, stations, depot):
    battery = BATTERY_CAPACITY_KWH
    t = START_TIME
    total_distance = 0.0
    total_charge_cost = 0.0
    num_charges = 0
    log = []
    path = [depot]
    current = depot
    waypoints = list(customer_order) + [depot]

    for idx, nxt in enumerate(waypoints):
        while True:
            d_to_nxt = dist(current, nxt)

            best_station = None
            best_net_cost = math.inf
            for s in stations:
                d_cs = dist(current, s)
                if d_cs * CONSUMPTION_PER_UNIT_DIST > battery + 1e-9:
                    continue
                batt_at_s = battery - d_cs * CONSUMPTION_PER_UNIT_DIST
                t_at_s = t + d_cs / SPEED
                energy = BATTERY_CAPACITY_KWH - batt_at_s
                charge_cost = price_at(s.name, t_at_s) * energy
                d_s_nxt = dist(s, nxt)
                detour = (d_cs + d_s_nxt) - d_to_nxt
                net = max(detour, 0.0) * DISTANCE_COST_PER_UNIT + charge_cost
                if net < best_net_cost:
                    best_net_cost = net
                    best_station = s

            must_charge = d_to_nxt * CONSUMPTION_PER_UNIT_DIST > battery + 1e-9

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
                    other_unused = [s for s in stations
                                    if s.name != best_station.name]
                    # Only charge opportunistically if:
                    # 1) price is well below average
                    # 2) at least one other station remains
                    # 3) the detour is modest relative to direct path
                    d_s_nxt = dist(best_station, nxt)
                    detour = (d_cs + d_s_nxt) - d_to_nxt
                    detour_cost = detour * DISTANCE_COST_PER_UNIT
                    energy_at_s = BATTERY_CAPACITY_KWH - (battery - d_cs * CONSUMPTION_PER_UNIT_DIST)
                    savings_vs_avg = (avg_price - price_now) * energy_at_s
                    if (price_now < avg_price * 0.65
                            and len(other_unused) >= 1
                            and savings_vs_avg > detour_cost * 2):
                        want_charge = True

            if must_charge or want_charge:
                if best_station is None:
                    return None
                d = dist(current, best_station)
                battery -= d * CONSUMPTION_PER_UNIT_DIST
                t += d / SPEED
                total_distance += d
                log.append(f"{current.name}->{best_station.name}: {d:.1f}u, "
                           f"bat={battery:.1f}")
                path.append(best_station)
                price = price_at(best_station.name, t)
                energy = BATTERY_CAPACITY_KWH - battery
                cost = price * energy
                battery = BATTERY_CAPACITY_KWH
                t += energy / CHARGE_RATE_KWH_PER_HOUR
                total_charge_cost += cost
                num_charges += 1
                log.append(f"  CHARGE {best_station.name}: +{energy:.1f}kWh "
                           f"@${price:.3f} = ${cost:.2f}, t={t:.1f}h")
                current = best_station
                continue

            break

        d = dist(current, nxt)
        battery -= d * CONSUMPTION_PER_UNIT_DIST
        t += d / SPEED
        total_distance += d
        log.append(f"{current.name}->{nxt.name}: {d:.1f}u, bat={battery:.1f}, "
                   f"t={t:.1f}h")
        path.append(nxt)
        current = nxt

    total_cost = total_distance * DISTANCE_COST_PER_UNIT + total_charge_cost
    result = EvalResult(True, total_cost, total_distance, total_charge_cost,
                        num_charges, t, log)
    return result, path


def smart_rule_based(customers, stations, depot):
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

    best_result, best_path, best_order = best_overall
    improved = True
    rounds = 0
    while improved:
        improved = False
        rounds += 1
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

    return (best_result.total_cost, best_path, best_result, rounds)

# ---------------------------------------------------------------------------
# 7. VISUALIZATION
# ---------------------------------------------------------------------------

def plot_world(smart_path, baseline_path, filename="world_map_scaled.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed — skipping plot)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, path, title, color in [
        (axes[0], smart_path, "Rule-Based Smart Route", "tab:orange"),
        (axes[1], baseline_path, "Greedy Baseline Route", "gray"),
    ]:
        ax.scatter(DEPOT.x, DEPOT.y, marker="s", s=160, c="black", zorder=5)
        ax.annotate("Depot", (DEPOT.x, DEPOT.y), fontsize=7,
                    textcoords="offset points", xytext=(5, 5))
        for c in CUSTOMERS:
            ax.scatter(c.x, c.y, marker="o", s=60, c="tab:blue", zorder=4)
            ax.annotate(c.name, (c.x, c.y), fontsize=6,
                        textcoords="offset points", xytext=(4, 4))
        for s in STATIONS:
            ax.scatter(s.x, s.y, marker="^", s=80, c="tab:green", zorder=4)
            ax.annotate(s.name, (s.x, s.y), fontsize=6,
                        textcoords="offset points", xytext=(4, 4))
        if path:
            xs = [n.x for n in path]
            ys = [n.y for n in path]
            ax.plot(xs, ys, c=color, linewidth=1.5, alpha=0.8)
            for k, (n1, n2) in enumerate(zip(path[:-1], path[1:])):
                mx, my = (n1.x + n2.x) / 2, (n1.y + n2.y) / 2
                if k % 3 == 0:
                    ax.annotate(str(k+1), (mx, my), fontsize=5, color=color,
                                ha="center")
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Scaled EVRP: {len(CUSTOMERS)} Customers, "
                 f"{len(STATIONS)} Stations", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print(f"\nSaved route map to {filename}")


def plot_price_heatmap(filename="price_heatmap.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    station_names = [s.name for s in STATIONS]
    data = np.array([STATION_PRICING[s] for s in station_names])

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r",
                   vmin=PRICE_MIN, vmax=PRICE_MAX)
    ax.set_yticks(range(len(station_names)))
    ax.set_yticklabels(station_names, fontsize=8)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=7)
    ax.set_xlabel("Hour of Day")
    ax.set_title("Station Price Schedule ($/kWh) — Green = Cheap, Red = Expensive")
    fig.colorbar(im, ax=ax, label="$/kWh", shrink=0.8)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print(f"Saved price heatmap to {filename}")


# ---------------------------------------------------------------------------
# 8. MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("SCALED EVRP SIMULATION")
    print(f"  {len(CUSTOMERS)} customers, {len(STATIONS)} charging stations")
    print(f"  Battery: {BATTERY_CAPACITY_KWH} kWh, "
          f"Consumption: {CONSUMPTION_PER_UNIT_DIST} kWh/unit")
    print(f"  World seed: {WORLD_SEED}, Price seed: {PRICING_SEED}")
    print("=" * 70)

    # --- Smart solver ---
    print("\nRunning rule-based smart solver...")
    t0 = time.time()
    smart = smart_rule_based(CUSTOMERS, STATIONS, DEPOT)
    t_smart = time.time() - t0

    smart_path = None
    if smart is None:
        print("Smart solver: INFEASIBLE")
    else:
        s_cost, smart_path, s_result, s_rounds = smart
        print(f"Smart solver completed in {t_smart:.2f}s "
              f"({s_rounds} improvement rounds)")
        print(f"  Route: {' -> '.join(n.name for n in smart_path)}")
        print(f"  Total cost      : ${s_cost:.2f}")
        print(f"    distance cost : ${s_result.total_distance * DISTANCE_COST_PER_UNIT:.2f} "
              f"({s_result.total_distance:.1f} units)")
        print(f"    charging cost : ${s_result.total_charge_cost:.2f} "
              f"({s_result.num_charges} charges)")
        print(f"  Finish time     : {s_result.finish_time:.1f}h")

    # --- Greedy baseline ---
    print("\nRunning greedy baseline...")
    t0 = time.time()
    baseline = greedy_baseline(CUSTOMERS, STATIONS, DEPOT)
    t_greedy = time.time() - t0

    baseline_path = None
    if baseline is None:
        print("Greedy baseline: INFEASIBLE")
    else:
        b_cost, baseline_path, b_result = baseline
        print(f"Greedy baseline completed in {t_greedy:.4f}s")
        print(f"  Route: {' -> '.join(n.name for n in baseline_path)}")
        print(f"  Total cost      : ${b_cost:.2f}")
        print(f"    distance cost : ${b_result.total_distance * DISTANCE_COST_PER_UNIT:.2f} "
              f"({b_result.total_distance:.1f} units)")
        print(f"    charging cost : ${b_result.total_charge_cost:.2f} "
              f"({b_result.num_charges} charges)")
        print(f"  Finish time     : {b_result.finish_time:.1f}h")

    # --- Comparison ---
    if smart is not None and baseline is not None:
        savings = b_cost - s_cost
        pct = 100 * savings / b_cost
        print(f"\n{'='*70}")
        print("COMPARISON")
        print(f"  {'Method':<24} {'Cost':>8}  {'Distance':>10}  "
              f"{'Charges':>8}  {'Time':>8}")
        print(f"  {'Rule-based smart':<24} ${s_cost:>7.2f}  "
              f"{s_result.total_distance:>9.1f}u  "
              f"{s_result.num_charges:>7d}   {s_result.finish_time:>6.1f}h")
        print(f"  {'Greedy baseline':<24} ${b_cost:>7.2f}  "
              f"{b_result.total_distance:>9.1f}u  "
              f"{b_result.num_charges:>7d}   {b_result.finish_time:>6.1f}h")
        print(f"\n  Smart solver saves ${savings:.2f} ({pct:.1f}%) "
              f"over greedy baseline")

    plot_world(smart_path, baseline_path)
    plot_price_heatmap()
