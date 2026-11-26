"""Validation experiment: compare solvers against the reference on key scenarios.

Produces:
    results/validation.csv       – summary table
    results/validation_table.tex – LaTeX booktabs tabular for the report
    results/plus1mw.csv          – +1 MW sensitivity check
    results/active_set_history.json – iteration history for Figure 4
"""

import csv
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common import (
    RESULTS,
    get_case,
    build_ed,
    build_dcopf,
    kkt_residual,
    QP,
    QPResult,
)


# ---------------------------------------------------------------------------
# Solver registry: (name, import_path, is_ed_only)
# ---------------------------------------------------------------------------
_SOLVERS: list[tuple[str, str, bool]] = [
    ("reference", "opf.reference", False),
    ("active_set", "opf.active_set", False),
    ("lambda_iter", "opf.lambda_iteration", True),
]


def _try_import(module_path: str):
    """Try to import a solver module; return its solve function or None."""
    try:
        mod = __import__(module_path, fromlist=["solve"])
        return mod.solve
    except (ImportError, ModuleNotFoundError):
        return None


def run() -> None:
    """Run the full validation suite."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    case = get_case()

    # ----- Reference solutions for comparison -----
    ref_solve = _try_import("opf.reference")
    if ref_solve is None:
        print("ERROR: reference solver not importable. Cannot validate.")
        return

    scenarios: list[tuple[str, QP]] = [
        ("base", build_dcopf(case, rate_scale=1.0)),
        ("rate_0.75", build_dcopf(case, rate_scale=0.75)),
    ]
    ed_qp = build_ed(case, load_scale=1.0)

    ref_results: dict[str, QPResult] = {}
    for label, qp in scenarios:
        ref_results[label] = ref_solve(qp)
    ref_results["ed_base"] = ref_solve(ed_qp)

    # ----- Validation rows -----
    val_rows: list[dict[str, Any]] = []

    # Compute non-optimal feasible starts for Figure 4 and iteration counts
    from scipy.optimize import linprog
    as_solve_fn = _try_import("opf.active_set")
    non_opt_iters: dict[str, int] = {}
    history_data: dict[str, list[float]] = {}

    all_cases = [("base", scenarios[0][1]), ("rate_0.75", scenarios[1][1]), ("ed_base", ed_qp)]
    if as_solve_fn is not None:
        for label, qp in all_cases:
            lp_res = linprog(
                c=-qp.g,
                A_ub=qp.A_in,
                b_ub=qp.b_in,
                A_eq=qp.A_eq,
                b_eq=qp.b_eq,
                bounds=(None, None),
                method="highs",
            )
            if lp_res.success:
                res_non_opt = as_solve_fn(qp, x0=lp_res.x)
                if res_non_opt.status == "optimal":
                    history_data[label] = res_non_opt.history
                    non_opt_iters[label] = res_non_opt.iters

    for solver_name, mod_path, is_ed_only in _SOLVERS:
        solve_fn = _try_import(mod_path)
        if solve_fn is None:
            print(f"  SKIP {solver_name}: not importable yet")
            continue

        if is_ed_only:
            # Lambda iteration only works on ED
            test_cases = [("ed_base", ed_qp)]
        else:
            test_cases = list(scenarios) + [("ed_base", ed_qp)]

        for label, qp in test_cases:
            ref = ref_results.get(label if not is_ed_only else "ed_base")
            if ref is None or ref.status != "optimal":
                continue

            result = solve_fn(qp)
            it_no = non_opt_iters.get(label, "---") if solver_name == "active_set" else "---"
            if result.status != "optimal":
                val_rows.append({
                    "solver": solver_name,
                    "scenario": label,
                    "cost": float("nan"),
                    "max_dx": float("nan"),
                    "max_dlam": float("nan"),
                    "kkt_res": float("nan"),
                    "iters": result.iters,
                    "iters_nonopt": it_no,
                    "status": result.status,
                })
                continue

            # Compare primal and dual against reference
            if len(result.x) == len(ref.x):
                max_dx = float(np.max(np.abs(result.x - ref.x)))
            else:
                max_dx = float("nan")

            n_lam_cmp = min(len(result.lam), len(ref.lam))
            if n_lam_cmp > 0:
                max_dlam = float(np.max(np.abs(result.lam[:n_lam_cmp] - ref.lam[:n_lam_cmp])))
            else:
                max_dlam = float("nan")

            kkt = kkt_residual(qp, result)

            val_rows.append({
                "solver": solver_name,
                "scenario": label,
                "cost": result.obj,
                "max_dx": max_dx,
                "max_dlam": max_dlam,
                "kkt_res": kkt,
                "iters": result.iters,
                "iters_nonopt": it_no,
                "status": result.status,
            })

            print(f"  {solver_name:15s} {label:12s}  cost={result.obj:.4f}"
                  f"  Δx={max_dx:.2e}  Δλ={max_dlam:.2e}  KKT={kkt:.2e}"
                  f"  iters={result.iters} (non-opt iters={it_no})")

    # ----- Write validation.csv -----
    csv_path = RESULTS / "validation.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["solver", "scenario", "cost", "max_dx", "max_dlam",
                       "kkt_res", "iters", "iters_nonopt", "status"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(val_rows)
    print(f"Saved {csv_path}")

    # ----- Write validation_table.tex -----
    tex_path = RESULTS / "validation_table.tex"
    _write_validation_tex(val_rows, tex_path)
    print(f"Saved {tex_path}")

    # ----- +1 MW sensitivity check -----
    _plus1mw_check(case)

    # ----- Active-set history JSON -----
    json_path = RESULTS / "active_set_history.json"
    with open(json_path, "w") as f:
        json.dump(history_data, f, indent=2)
    print(f"Saved {json_path}")


def _write_validation_tex(rows: list[dict[str, Any]], path: Path) -> None:
    """Write a LaTeX booktabs tabular for the validation table."""
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Solver & Scenario & Cost (\$/h) & $\max|\Delta x|$ & "
        r"$\max|\Delta\lambda|$ & KKT res. & Iters & Iters (non-opt) \\",
        r"\midrule",
    ]
    for r in rows:
        cost_s = f"{r['cost']:.2f}" if np.isfinite(r["cost"]) else "---"
        dx_s = f"{r['max_dx']:.2e}" if np.isfinite(r["max_dx"]) else "---"
        dl_s = f"{r['max_dlam']:.2e}" if np.isfinite(r["max_dlam"]) else "---"
        kkt_s = f"{r['kkt_res']:.2e}" if np.isfinite(r["kkt_res"]) else "---"
        solver_tex = r["solver"].replace("_", r"\_")
        scenario_tex = r["scenario"].replace("_", r"\_")
        it_no_s = str(r.get("iters_nonopt", "---"))
        lines.append(
            f"{solver_tex} & {scenario_tex} & {cost_s} & {dx_s} & "
            f"{dl_s} & {kkt_s} & {r['iters']} & {it_no_s} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines), encoding="utf-8")


def _plus1mw_check(case) -> None:
    """Add +1 MW at buses 8 and 30, compare Δcost to LMP."""
    from experiments.common import solve as as_solve
    csv_path = RESULTS / "plus1mw.csv"
    qp_base = build_dcopf(case, rate_scale=1.0)
    res_base = as_solve(qp_base)
    if res_base.status != "optimal":
        print("  SKIP +1MW check: base case not optimal")
        return

    rows = []
    for bus in [8, 30]:
        idx = bus - 1
        qp_pert = dataclasses.replace(
            qp_base,
            b_eq=qp_base.b_eq.copy(),
        )
        qp_pert.b_eq[idx] += 1.0
        res_pert = as_solve(qp_pert)
        if res_pert.status != "optimal":
            print(f"  SKIP +1MW at bus {bus}: perturbed case not optimal")
            continue

        delta_cost = res_pert.obj - res_base.obj
        lmp = res_base.lam[idx]
        rows.append({
            "bus": bus,
            "lmp": f"{lmp:.6f}",
            "delta_cost": f"{delta_cost:.6f}",
            "diff": f"{delta_cost - lmp:.6f}",
        })
        print(f"  +1MW bus {bus}: Δcost={delta_cost:.6f}  LMP={lmp:.6f}"
              f"  diff={delta_cost - lmp:.6f}")

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bus", "lmp", "delta_cost", "diff"])
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    run()
