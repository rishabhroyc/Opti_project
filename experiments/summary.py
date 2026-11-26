"""Print a short summary of the main results.

Reads the files written by the other experiment scripts in results/, so the
summary always matches the saved data. Run it after run_all.py, or on its own:

    python -m experiments.summary

The text uses plain ASCII only, so it also works when the output is
redirected to a file on Windows.
"""

import csv
import json
from pathlib import Path

import numpy as np

from experiments.common import RESULTS, get_case

LINE = "=" * 60


def _read_csv(name: str) -> list[dict[str, str]]:
    """Read results/<name> into a list of row dictionaries."""
    with open(RESULTS / name, newline="") as f:
        return list(csv.DictReader(f))


def _row_at(rows: list[dict[str, str]], scale: float) -> dict[str, str] | None:
    """Return the sweep row whose 'scale' equals the given value."""
    for row in rows:
        if abs(float(row["scale"]) - scale) < 1e-9:
            return row
    return None


def _generator_kinks(case) -> list[tuple[int, float]]:
    """Demand (MW) at which each generator reaches its upper limit, sorted.

    For generator j the price at its limit is lam_j = b_j + 2 c_j Pmax_j;
    the demand at that price is the sum of every generator's output there.
    """
    kinks = []
    for j in range(len(case.b)):
        lam_j = case.b[j] + 2.0 * case.c[j] * case.p_max[j]
        output = np.clip((lam_j - case.b) / (2.0 * case.c), case.p_min, case.p_max)
        kinks.append((j + 1, float(np.sum(output))))
    return sorted(kinks, key=lambda item: item[1])


def build_summary() -> str:
    """Build the summary text from the files in results/."""
    case = get_case()
    lines = ["", LINE, "SUMMARY OF RESULTS", LINE]

    # 1. Base case economic dispatch (load scale 1.0 in the load sweep)
    base = _row_at(_read_csv("load_sweep.csv"), 1.0)
    if base is not None:
        dispatch = ", ".join(
            f"G{i + 1} {float(base[f'P{i + 1}']):.2f}" for i in range(len(case.b))
        )
        lines += [
            f"Base case ({float(base['demand_mw']):.1f} MW demand, no congestion):",
            f"  Minimum cost  : {float(base['cost']):.2f} $/h",
            f"  Price (lambda): {float(base['lambda']):.4f} $/MWh, the same at every bus",
            f"  Dispatch (MW) : {dispatch}",
            "",
        ]

    # 2. Our solvers against the CVXPY reference
    validation = _read_csv("validation.csv")
    names = {"active_set": "Active set", "lambda_iter": "Lambda iteration"}
    scenarios = {"base": "DC-OPF base", "rate_0.75": "DC-OPF 75%", "ed_base": "ED"}
    lines += [
        "Our solvers vs the CVXPY + Clarabel reference:",
        f"  {'Solver':<17}{'Problem':<12}{'Cost':>8}{'max|dx|':>10}{'KKT':>10}  Iterations",
    ]
    for row in validation:
        if row["solver"] not in names:
            continue
        iters = row["iters"]
        if row["iters_nonopt"] not in ("", "---"):
            iters += f" ({row['iters_nonopt']} from high-cost start)"
        lines.append(
            f"  {names[row['solver']]:<17}{scenarios.get(row['scenario'], row['scenario']):<12}"
            f"{float(row['cost']):>8.2f}{float(row['max_dx']):>10.1e}"
            f"{float(row['kkt_res']):>10.1e}  {iters}"
        )
    lines.append("")

    # 3. Prices really are marginal costs (+1 MW check)
    plus1 = _read_csv("plus1mw.csv")
    if plus1:
        row = plus1[0]
        lines += [
            "Price check (add 1 MW of demand at one bus):",
            f"  Bus {row['bus']}: cost rose by {float(row['delta_cost']):.4f} $/h; "
            f"price there was {float(row['lmp']):.4f} $/MWh",
            f"  Difference {float(row['diff']):.4f} = the quadratic-cost term, as expected",
            "",
        ]

    # 4. Congestion sweep
    congestion = _read_csv("congestion.csv")
    feasible = [r for r in congestion if r["status"] == "optimal"]
    uncongested = [r for r in feasible if not r["binding_lines"]]
    congested = [r for r in feasible if r["binding_lines"]]
    infeasible = [r for r in congestion if r["status"] == "infeasible"]
    lines.append("Line congestion (line ratings scaled down from 100%):")
    if uncongested:
        lowest = min(uncongested, key=lambda r: float(r["scale"]))
        lines.append(f"  No congestion down to {float(lowest['scale']) * 100:.1f}% ratings "
                     "(one price everywhere)")
    if congested:
        first = max(congested, key=lambda r: float(r["scale"]))
        lines.append(f"  First congested line: {first['binding_lines']} "
                     f"(at {float(first['scale']) * 100:.1f}% ratings)")
    for scale in (0.75, 0.72):
        row = _row_at(congestion, scale)
        if row is not None and row["status"] == "optimal":
            lines.append(
                f"  At {scale * 100:.0f}%: cost {float(row['cost']):.2f} $/h, "
                f"prices {float(row['lmp_min']):.2f} to {float(row['lmp_max']):.2f} $/MWh"
            )
            lines.append(f"           binding lines {row['binding_lines'].replace(';', ', ')}")
    if infeasible:
        highest = max(infeasible, key=lambda r: float(r["scale"]))
        lines.append(f"  Infeasible from {float(highest['scale']) * 100:.1f}% ratings down "
                     "(no dispatch can meet demand)")
    lines.append("")

    # 5. Generator limits in the demand sweep (kinks in Figure 1)
    demand_max = max(float(r["demand_mw"]) for r in _read_csv("load_sweep.csv"))
    kinks = [f"G{g} {d:.2f} MW" for g, d in _generator_kinks(case) if d <= demand_max]
    lines += [
        "Demand at which each generator reaches its limit (kinks in Figure 1):",
        "  " + ", ".join(kinks),
        "",
    ]

    # 6. Convergence from a high-cost start (Figure 4)
    history_path = RESULTS / "active_set_history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
        labels = {"base": "DC-OPF base", "rate_0.75": "DC-OPF 75%", "ed_base": "ED"}
        lines.append("Active-set convergence from a high-cost start (Figure 4):")
        for key, values in history.items():
            if values:
                lines.append(f"  {labels.get(key, key):<12}{values[0]:.2f} -> {values[-1]:.2f} $/h "
                             f"in {len(values)} iterations")
        lines.append("")

    lines += [f"Tables, data and figures: {RESULTS}", LINE]
    return "\n".join(lines)


def run() -> None:
    """Print the summary."""
    print(build_summary())


if __name__ == "__main__":
    run()
