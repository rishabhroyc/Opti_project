"""ED load-sweep experiment: price (λ) and dispatch vs demand scaling.

Sweeps load_scale from 0.500 to 1.750 (step 0.025, 51 points).
Saves results/load_sweep.csv.
"""

import csv
from pathlib import Path

import numpy as np

from experiments.common import RESULTS, get_case, build_ed, solve


def run() -> Path:
    """Execute the load sweep and write results/load_sweep.csv."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    outpath = RESULTS / "load_sweep.csv"

    case = get_case()
    n_gen = len(case.gen_bus)
    base_demand = float(np.sum(case.bus_d))

    header = ["scale", "demand_mw", "status", "cost", "lambda"] + [
        f"P{i+1}" for i in range(n_gen)
    ]

    rows: list[list[str]] = []
    for k in range(51):
        s = round(0.5 + 0.025 * k, 3)
        demand = base_demand * s
        qp = build_ed(case, load_scale=s)
        result = solve(qp)

        if result.status == "optimal":
            lam_val = float(result.lam[0]) if len(result.lam) > 0 else float("nan")
            row = [
                f"{s:.3f}",
                f"{demand:.2f}",
                result.status,
                f"{result.obj:.6f}",
                f"{lam_val:.6f}",
            ] + [f"{result.x[i]:.6f}" for i in range(n_gen)]
        else:
            row = [
                f"{s:.3f}",
                f"{demand:.2f}",
                result.status,
            ] + ["NaN"] * (2 + n_gen)

        rows.append(row)
        print(f"  load_scale={s:.3f}  demand={demand:.1f} MW  status={result.status}"
              + (f"  λ={lam_val:.4f}" if result.status == "optimal" else ""))

    with open(outpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    print(f"Saved {outpath}")
    return outpath


if __name__ == "__main__":
    run()
