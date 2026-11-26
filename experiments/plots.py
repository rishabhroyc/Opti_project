"""Generate the four result figures from CSV / JSON data.

Figures (saved to results/):
    fig1_price_vs_demand.png   – ED λ vs demand, kinks at generator limits
    fig2_cost_spread_vs_rating.png – cost and LMP spread vs rating scale
    fig3_lmp_per_bus.png       – grouped bars of LMP at each bus for 3 scales
    fig4_active_set_history.png – objective per iteration (active-set solver)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.common import RESULTS


def _savefig(fig: plt.Figure, name: str) -> None:
    """Save a figure to results/ with standard settings."""
    path = RESULTS / name
    fig.savefig(str(path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def fig1_price_vs_demand() -> None:
    """Figure 1: ED price (λ) vs total demand (MW)."""
    csv_path = RESULTS / "load_sweep.csv"
    if not csv_path.exists():
        print("  SKIP fig1: load_sweep.csv not found")
        return

    df = pd.read_csv(csv_path)
    df = df[df["status"] == "optimal"].copy()
    df["demand_mw"] = df["demand_mw"].astype(float)
    df["lambda"] = df["lambda"].astype(float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["demand_mw"], df["lambda"], "o-", markersize=3, linewidth=1.2)
    ax.set_xlabel("Total Demand (MW)")
    ax.set_ylabel("System Marginal Price λ ($/MWh)")
    ax.set_title("Figure 1: ED Price vs Demand")
    ax.grid(True, alpha=0.3)
    _savefig(fig, "fig1_price_vs_demand.png")


def fig2_cost_spread_vs_rating() -> None:
    """Figure 2: cost and LMP spread vs rating scale, infeasible region shaded."""
    csv_path = RESULTS / "congestion.csv"
    if not csv_path.exists():
        print("  SKIP fig2: congestion.csv not found")
        return

    df = pd.read_csv(csv_path)
    opt = df[df["status"] == "optimal"].copy()
    opt["scale"] = opt["scale"].astype(float)
    opt["cost"] = opt["cost"].astype(float)
    opt["lmp_spread"] = opt["lmp_spread"].astype(float)

    infeasible = df[df["status"] != "optimal"]["scale"].astype(float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    # Top panel: cost
    ax1.plot(opt["scale"], opt["cost"], "o-", markersize=3, linewidth=1.2, color="tab:blue")
    ax1.set_ylabel("Total Cost ($/h)")
    ax1.set_title("Figure 2: Cost and LMP Spread vs Rating Scale")
    ax1.grid(True, alpha=0.3)

    # Bottom panel: LMP spread
    ax2.plot(opt["scale"], opt["lmp_spread"], "s-", markersize=3, linewidth=1.2, color="tab:orange")
    ax2.set_xlabel("Line Rating Scale Factor")
    ax2.set_ylabel("LMP Spread ($/MWh)")
    ax2.grid(True, alpha=0.3)

    # Shade infeasible region on both panels
    if len(infeasible) > 0:
        x_min = float(infeasible.min())
        x_max = float(infeasible.max())
        for ax in (ax1, ax2):
            ax.axvspan(x_min - 0.0025, x_max + 0.0025,
                       alpha=0.2, color="red", label="Infeasible")
        ax2.legend(loc="upper left")

    fig.tight_layout()
    _savefig(fig, "fig2_cost_spread_vs_rating.png")


def fig3_lmp_per_bus() -> None:
    """Figure 3: LMP at each bus for rating scales 1.00, 0.75, 0.72."""
    csv_path = RESULTS / "congestion.csv"
    if not csv_path.exists():
        print("  SKIP fig3: congestion.csv not found")
        return

    df = pd.read_csv(csv_path)
    scales = [1.0, 0.75, 0.72]
    n_bus = 30

    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.25
    x = np.arange(1, n_bus + 1)

    for i, s in enumerate(scales):
        row = df[np.isclose(df["scale"].astype(float), s, atol=1e-4)]
        if row.empty or row.iloc[0]["status"] != "optimal":
            print(f"  SKIP scale {s} in fig3: not optimal or missing")
            continue
        lmps = [float(row.iloc[0][f"lmp_{k+1}"]) for k in range(n_bus)]
        ax.bar(x + (i - 1) * width, lmps, width, label=f"×{s:.2f}")

    ax.set_xlabel("Bus Number")
    ax.set_ylabel("LMP ($/MWh)")
    ax.set_title("Figure 3: Locational Marginal Prices by Bus")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    _savefig(fig, "fig3_lmp_per_bus.png")


def fig4_active_set_history() -> None:
    """Figure 4: Active-set solver objective per iteration."""
    json_path = RESULTS / "active_set_history.json"
    if not json_path.exists():
        print("  SKIP fig4: active_set_history.json not found (Varshith's solver history needed)")
        return

    with open(json_path) as f:
        data = json.load(f)

    if not data:
        print("  SKIP fig4: no history data recorded")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, hist in data.items():
        ax.plot(range(1, len(hist) + 1), hist, "o-", markersize=4, label=label)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective Value ($/h)")
    ax.set_title("Figure 4: Active-Set Solver Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _savefig(fig, "fig4_active_set_history.png")


def run() -> None:
    """Generate all four figures."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    print("Generating figures...")
    fig1_price_vs_demand()
    fig2_cost_spread_vs_rating()
    fig3_lmp_per_bus()
    fig4_active_set_history()
    print("Done.")


if __name__ == "__main__":
    run()
