"""Rating-sweep experiment: cost, LMPs and binding lines vs line-rating scale.

Sweeps rate_scale from 1.000 down to 0.700 (step 0.005, 61 points).
Saves results/congestion.csv.
"""

import csv
import math
from pathlib import Path

import numpy as np

from experiments.common import (
    RESULTS,
    get_case,
    build_dcopf,
    solve,
    branch_ends,
    binding_lines,
)


def run() -> Path:
    """Execute the congestion sweep and write results/congestion.csv."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    outpath = RESULTS / "congestion.csv"

    case = get_case()
    ends = branch_ends(case)
    n_bus = 30

    header = [
        "scale", "status", "cost",
        "lmp_min", "lmp_max", "lmp_spread",
        "n_binding", "binding_lines",
    ] + [f"lmp_{k+1}" for k in range(n_bus)]

    rows: list[list[str]] = []
    for k in range(61):
        s = round(1.0 - 0.005 * k, 3)
        qp = build_dcopf(case, rate_scale=s)
        result = solve(qp)

        if result.status == "optimal":
            lam = result.lam[:n_bus]
            bl = binding_lines(result.mu)
            bl_str = ";".join(f"{ends[i][0]}-{ends[i][1]}" for i in bl)
            row = [
                f"{s:.3f}",
                result.status,
                f"{result.obj:.6f}",
                f"{lam.min():.6f}",
                f"{lam.max():.6f}",
                f"{lam.max() - lam.min():.6f}",
                str(len(bl)),
                bl_str,
            ] + [f"{lam[j]:.6f}" for j in range(n_bus)]
        else:
            row = [
                f"{s:.3f}",
                result.status,
            ] + ["NaN"] * (6 + n_bus)

        rows.append(row)
        print(f"  rate_scale={s:.3f}  status={result.status}"
              + (f"  cost={result.obj:.2f}" if result.status == "optimal" else ""))

    with open(outpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    print(f"Saved {outpath}")
    return outpath


if __name__ == "__main__":
    run()
