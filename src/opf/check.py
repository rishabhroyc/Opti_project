"""Solution verification, KKT residual checks, and solver comparison tools.

Author: Rishabh Roy Choudhury (Albert and Einstein)
Uses NumPy only.
"""

from dataclasses import dataclass
from typing import Any, Callable
import numpy as np

from src.opf.models import Case
from src.opf.qp import QP, QPResult


@dataclass
class KKTResiduals:
    """Four components of the KKT residual plus the overall maximum.

    Attributes:
        stationarity: ||Hx + g - A_eq^T lam + A_in^T mu||_inf
        primal: max(||A_eq x - b_eq||_inf, max(max(A_in x - b_in), 0))
        dual: max(max(-mu), 0) (dual feasibility violation)
        complementarity: max_i |mu_i * (A_in x - b_in)_i|
        max_residual: max of all four components
    """

    stationarity: float
    primal: float
    dual: float
    complementarity: float
    max_residual: float


def kkt_residuals(qp: QP, res: QPResult) -> KKTResiduals:
    """Compute the individual components and overall maximum KKT residual.

    Sign convention (CONTEXT.md section 4):
        Lagrangian: L = 1/2 x^T H x + g^T x - lam^T (A_eq x - b_eq) + mu^T (A_in x - b_in)
        Stationarity: Hx + g - A_eq^T lam + A_in^T mu = 0

    Args:
        qp: QP problem definition.
        res: QPResult from a solver.

    Returns:
        KKTResiduals dataclass with stationarity, primal, dual,
        complementarity, and max_residual.
    """
    x = res.x
    lam = res.lam
    mu = res.mu

    # 1. Stationarity: r_stat = Hx + g - A_eq^T lam + A_in^T mu
    stat = qp.H @ x + qp.g
    if qp.A_eq is not None and qp.A_eq.shape[0] > 0 and len(lam) > 0:
        stat = stat - qp.A_eq.T @ lam
    if qp.A_in is not None and qp.A_in.shape[0] > 0 and len(mu) > 0:
        stat = stat + qp.A_in.T @ mu
    stationarity = float(np.max(np.abs(stat))) if len(stat) > 0 else 0.0

    # 2. Primal feasibility:
    # A_eq x = b_eq
    if qp.A_eq is not None and qp.A_eq.shape[0] > 0:
        eq_res = float(np.max(np.abs(qp.A_eq @ x - qp.b_eq)))
    else:
        eq_res = 0.0

    # A_in x <= b_in
    if qp.A_in is not None and qp.A_in.shape[0] > 0:
        in_viol = float(np.max(np.maximum(qp.A_in @ x - qp.b_in, 0.0)))
    else:
        in_viol = 0.0
    primal = max(eq_res, in_viol)

    # 3. Dual feasibility: mu >= 0 (violation is positive part of -mu)
    if len(mu) > 0:
        dual = float(np.max(np.maximum(-mu, 0.0)))
    else:
        dual = 0.0

    # 4. Complementarity: mu_i * (A_in x - b_in)_i = 0
    if qp.A_in is not None and qp.A_in.shape[0] > 0 and len(mu) > 0:
        slacks = qp.A_in @ x - qp.b_in
        comp = np.abs(mu * slacks)
        complementarity = float(np.max(comp))
    else:
        complementarity = 0.0

    max_res = max(stationarity, primal, dual, complementarity)

    return KKTResiduals(
        stationarity=stationarity,
        primal=primal,
        dual=dual,
        complementarity=complementarity,
        max_residual=max_res,
    )


def kkt_residual(qp: QP, res: QPResult) -> float:
    """Compute the maximum KKT residual for a solution.

    Returns the largest of stationarity, primal feasibility,
    dual feasibility, and complementarity.

    Args:
        qp: QP problem specification.
        res: Solver QPResult.

    Returns:
        Scalar maximum KKT residual.
    """
    return kkt_residuals(qp, res).max_residual


def comparison_table(
    results: list[tuple[str, QPResult]],
    ref: QPResult,
    qp: QP | None = None,
) -> list[dict[str, Any]]:
    """Compare a list of solver results against a reference QPResult.

    Metrics computed:
        cost: Objective value.
        max_dx: max |x - x_ref|.
        max_dlambda: max |lam - lam_ref|.
        kkt_residual: Maximum KKT residual (computed if qp is provided).
        iters: Number of iterations.

    Args:
        results: List of (solver_name, QPResult) pairs.
        ref: Reference QPResult benchmark.
        qp: Optional QP problem used to calculate KKT residuals.

    Returns:
        List of dictionaries with comparison metrics per solver.
    """
    rows: list[dict[str, Any]] = []

    for name, res in results:
        cost = float(res.obj)

        if len(res.x) > 0 and len(ref.x) > 0 and len(res.x) == len(ref.x):
            max_dx = float(np.max(np.abs(res.x - ref.x)))
        else:
            max_dx = float("nan")

        if len(res.lam) > 0 and len(ref.lam) > 0 and len(res.lam) == len(ref.lam):
            max_dlam = float(np.max(np.abs(res.lam - ref.lam)))
        else:
            max_dlam = float("nan")

        if qp is not None and len(res.x) > 0:
            kkt_res = kkt_residual(qp, res)
        else:
            kkt_res = float("nan")

        rows.append({
            "name": name,
            "cost": cost,
            "max_dx": max_dx,
            "max_dlambda": max_dlam,
            "kkt_residual": kkt_res,
            "iters": int(res.iters),
            "status": res.status,
        })

    return rows


def print_comparison_table(table_data: list[dict[str, Any]]) -> str:
    """Format and print comparison table to console.

    Args:
        table_data: List of comparison dictionaries from comparison_table().

    Returns:
        Formatted ASCII string representation of the table.
    """
    header = f"{'Solver':<18} {'Cost':<12} {'Max |dx|':<12} {'Max |dlam|':<12} {'KKT Res':<12} {'Iters':<6} {'Status':<10}"
    separator = "-" * len(header)
    lines = [separator, header, separator]

    for r in table_data:
        cost_str = f"{r['cost']:.4f}" if np.isfinite(r['cost']) else "inf"
        dx_str = f"{r['max_dx']:.2e}" if np.isfinite(r['max_dx']) else "-"
        dlam_str = f"{r['max_dlambda']:.2e}" if np.isfinite(r['max_dlambda']) else "-"
        kkt_str = f"{r['kkt_residual']:.2e}" if np.isfinite(r['kkt_residual']) else "-"
        line = f"{r['name']:<18} {cost_str:<12} {dx_str:<12} {dlam_str:<12} {kkt_str:<12} {r['iters']:<6} {r['status']:<10}"
        lines.append(line)

    lines.append(separator)
    table_text = "\n".join(lines)
    print(table_text)
    return table_text


def marginal_cost_check(
    qp: QP,
    bus_idx: int = 0,
    solver_fn: Callable[[QP], QPResult] | None = None,
) -> dict[str, float]:
    """Check LMP sensitivity by adding 1 MW demand at a chosen bus.

    Solves the original problem, then adds 1.0 to bus_idx entry in b_eq,
    re-solves, and compares the cost increase to the bus's LMP (lambda).
    Because a 1 MW step is finite, they match to first order.

    Args:
        qp: Base DC-OPF QP problem.
        bus_idx: 0-based bus index where 1 MW demand is added (default 0 for bus 1).
        solver_fn: Solver callable taking QP and returning QPResult.
                   Defaults to src.opf.reference.solve.

    Returns:
        Dictionary with:
        - base_cost: Original cost.
        - perturbed_cost: Cost with +1 MW at bus_idx.
        - delta_cost: Cost increase.
        - bus_lambda: Lagrange multiplier (LMP) at bus_idx from base solve.
        - gap: |delta_cost - bus_lambda|.
    """
    if solver_fn is None:
        from src.opf.reference import solve
        solver_fn = solve

    # 1. Base solution
    res_base = solver_fn(qp)
    base_cost = float(res_base.obj)
    bus_lambda = float(res_base.lam[bus_idx])

    # 2. Perturbed QP (+1 MW at bus_idx)
    # Clone QP and add 1.0 MW to b_eq[bus_idx]
    b_eq_pert = qp.b_eq.copy()
    b_eq_pert[bus_idx] += 1.0

    qp_pert = QP(
        H=qp.H,
        g=qp.g,
        A_eq=qp.A_eq,
        b_eq=b_eq_pert,
        A_in=qp.A_in,
        b_in=qp.b_in,
        names=qp.names,
    )

    res_pert = solver_fn(qp_pert)
    pert_cost = float(res_pert.obj)
    delta_cost = pert_cost - base_cost
    gap = abs(delta_cost - bus_lambda)

    return {
        "base_cost": base_cost,
        "perturbed_cost": pert_cost,
        "delta_cost": delta_cost,
        "bus_lambda": bus_lambda,
        "gap": gap,
    }


def marginal_costs(case: Case, result: QPResult) -> np.ndarray:
    """Compute generator marginal costs b_i + 2 * c_i * P_i for the 6 generators.

    Args:
        case: Case system data with cost parameters b and c.
        result: QPResult containing dispatch vector x.

    Returns:
        Array of shape (6,) with marginal generation costs.
    """
    P = result.x[:6]
    return case.b + 2.0 * case.c * P
