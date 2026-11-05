"""Visualization utilities for the fixed-order EVRP."""

from __future__ import annotations
from typing import List, Optional

import numpy as np

from utils.world import World, Node


def plot_route(
    world: World,
    path_nodes: List[int],
    title: str = "Route",
    filename: Optional[str] = None,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot")
        return

    fig, ax = plt.subplots(figsize=(9, 9))
    w = world

    # depot
    ax.scatter(w.depot.x, w.depot.y, marker="s", s=160, c="black", zorder=5)
    ax.annotate("Depot", (w.depot.x, w.depot.y), fontsize=7,
                textcoords="offset points", xytext=(5, 5))

    # all customers (light)
    for c in w.customers:
        ax.scatter(c.x, c.y, marker="o", s=40, c="lightblue", zorder=3)

    # visited customers (bold, numbered)
    for seq_i, ci in enumerate(w.visit_order):
        c = w.customers[ci]
        ax.scatter(c.x, c.y, marker="o", s=80, c="tab:blue", zorder=4)
        ax.annotate(f"{seq_i+1}:{c.name}", (c.x, c.y), fontsize=6,
                    textcoords="offset points", xytext=(5, 5), fontweight="bold")

    # stations
    for s in w.stations:
        ax.scatter(s.x, s.y, marker="^", s=90, c="tab:green", zorder=4)
        ax.annotate(s.name, (s.x, s.y), fontsize=6,
                    textcoords="offset points", xytext=(4, 4))

    # route path
    if path_nodes:
        xs = [w.all_nodes[i].x for i in path_nodes]
        ys = [w.all_nodes[i].y for i in path_nodes]
        ax.plot(xs, ys, c="tab:orange", linewidth=1.8, alpha=0.8, zorder=2)
        for k in range(len(path_nodes) - 1):
            mx = (xs[k] + xs[k+1]) / 2
            my = (ys[k] + ys[k+1]) / 2
            ax.annotate(str(k+1), (mx, my), fontsize=5, color="tab:orange",
                        ha="center", zorder=6)

    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if filename:
        fig.savefig(filename, dpi=150)
        print(f"Saved {filename}")
    else:
        plt.show()
    plt.close(fig)


def plot_price_heatmap(world: World, filename: Optional[str] = None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    data = world.price_schedule
    fig, ax = plt.subplots(figsize=(14, max(3, world.num_stations * 0.5)))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r",
                   vmin=world.price_min, vmax=world.price_max)
    ax.set_yticks(range(world.num_stations))
    ax.set_yticklabels([s.name for s in world.stations], fontsize=8)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=7)
    ax.set_xlabel("Hour of Day")
    ax.set_title("Station Prices ($/kWh)")
    fig.colorbar(im, ax=ax, label="$/kWh", shrink=0.8)
    fig.tight_layout()

    if filename:
        fig.savefig(filename, dpi=150)
        print(f"Saved {filename}")
    else:
        plt.show()
    plt.close(fig)
