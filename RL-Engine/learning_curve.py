"""
Plot and summarise learning curves written by train.py (cost_curve.csv).

    python learning_curve.py models/experiments/partial_2m
    python learning_curve.py models/experiments/full_charge models/experiments/partial_2m \
        --labels "always full,partial" --out outputs/curves.png

For each run it prints the mean held-out cost around a few milestones and
whether the cost still dropped over the last part of training — the check
asked for in "train longer and see whether costs drop further". Curriculum
stages are drawn as separate segments (their costs are on routes of
different lengths, so they are not comparable with each other).
"""

import argparse
import csv
import os

import numpy as np


def read_curve(path):
    if os.path.isdir(path):
        path = os.path.join(path, "cost_curve.csv")
    rows = list(csv.DictReader(open(path)))
    return {
        "steps": np.array([int(r["timesteps"]) for r in rows]),
        "stage": [r["stage"] for r in rows],
        "cost": np.array([float(r["mean_cost"]) for r in rows]),
        "deaths": np.array([int(r["deaths"]) for r in rows]),
        "late": np.array([float(r["mean_late"]) for r in rows]),
    }


def window_mean(c, lo, hi):
    sel = (c["steps"] > lo) & (c["steps"] <= hi)
    return float(np.mean(c["cost"][sel])) if sel.any() else float("nan")


def summarise(name, c):
    print(f"\n{name}")
    stages = list(dict.fromkeys(c["stage"]))
    for st in stages:
        idx = [i for i, s in enumerate(c["stage"]) if s == st]
        sub = {k: (v[idx] if isinstance(v, np.ndarray) else [v[i] for i in idx])
               for k, v in c.items()}
        steps = sub["steps"]
        start, end = int(steps.min()), int(steps.max())
        label = f"  stage {st}" if len(stages) > 1 else " "
        best = int(np.argmin(sub["cost"]))
        print(f"{label} steps {start:,}-{end:,}: first eval ${sub['cost'][0]:.2f}, "
              f"best ${sub['cost'][best]:.2f} at {steps[best]:,}, "
              f"last ${sub['cost'][-1]:.2f}")
        span = end - start
        if span <= 0:
            continue
        # mean cost over consecutive quarters of the stage
        qs = [window_mean(sub, start + span * q / 4 - 1, start + span * (q + 1) / 4)
              for q in range(4)]
        print("   mean cost by quarter: " + "  ".join(f"${q:.2f}" for q in qs))
        if all(np.isfinite(qs[2:])):
            drop = (qs[2] - qs[3]) / qs[2] * 100
            verdict = ("still improving" if drop > 1.0 else
                       "flat (converged)" if drop > -1.0 else "got worse")
            print(f"   last quarter vs third quarter: {drop:+.1f}% -> {verdict}")
        if sub["deaths"].sum():
            print(f"   evaluations with battery deaths: {(sub['deaths'] > 0).sum()}")


def main():
    p = argparse.ArgumentParser(description="Learning curves from cost_curve.csv")
    p.add_argument("runs", nargs="+", help="Run directories or cost_curve.csv files")
    p.add_argument("--labels", type=str, default=None, help="Comma-separated legend labels")
    p.add_argument("--out", type=str, default=None, help="Save the plot here (PNG)")
    p.add_argument("--smooth", type=int, default=3, help="Moving-average window (evals)")
    args = p.parse_args()

    labels = (args.labels.split(",") if args.labels
              else [os.path.basename(os.path.normpath(r)) for r in args.runs])
    curves = [read_curve(r) for r in args.runs]
    for name, c in zip(labels, curves):
        summarise(name, c)

    if not args.out:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name, c in zip(labels, curves):
        stages = list(dict.fromkeys(c["stage"]))
        color = None
        for st in stages:
            idx = np.array([i for i, s in enumerate(c["stage"]) if s == st])
            y = c["cost"][idx]
            if args.smooth > 1 and len(y) >= args.smooth:
                k = np.ones(args.smooth) / args.smooth
                y = np.concatenate([y[:args.smooth - 1], np.convolve(y, k, "valid")])
            lab = name if len(stages) == 1 else f"{name} (≤{st} customers)"
            line, = ax.plot(c["steps"][idx], y, label=lab, color=color, linewidth=1.8)
            ax.scatter(c["steps"][idx], c["cost"][idx], s=6, alpha=0.3,
                       color=line.get_color())
            if len(stages) > 1:
                ax.axvline(c["steps"][idx].min(), color="grey", alpha=0.3, linestyle=":")
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Mean held-out route cost ($)")
    ax.set_title("Learning curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
