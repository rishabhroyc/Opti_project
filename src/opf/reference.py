"""Reference QP solver using CVXPY and Clarabel.

Author: Rishabh Roy Choudhury (Albert and Einstein)

Conforms to the shared QP interface in src/opf/qp.py.
Solves convex quadratic programs with cp.CLARABEL.
"""

from typing import Any
import cvxpy as cp
import numpy as np

from src.opf.qp import QP, QPResult


def solve(
    qp: QP,
    x0: np.ndarray | None = None,
    tol: float = 1e-8,
    max_iter: int = 500,
) -> QPResult:
    """Solve convex quadratic program using CVXPY with the Clarabel solver.

    Problem formulation:
        minimize    1/2 x^T H x + g^T x
        subject to  A_eq x = b_eq
                    A_in x <= b_in

    Sign convention (CONTEXT.md section 4):
        Lagrangian: L = f - lam^T (A_eq x - b_eq) + mu^T (A_in x - b_in)
        Stationarity: Hx + g - A_eq^T lam + A_in^T mu = 0
        - lam is positive (+3.7892 in the base case) and equals d(cost*)/d(b_eq).
        - Each balance-row lam is that bus's LMP.
        - CVXPY's dual for equality constraints has the opposite sign, so it is
          flipped here: lam = -eq_con.dual_value.
        - mu >= 0 (inequality duals).

    Args:
        qp: QP problem specification.
        x0: Optional initial point (ignored by Clarabel interior-point solver).
        tol: Convergence tolerance passed to Clarabel.
        max_iter: Maximum iterations.

    Returns:
        QPResult containing primal/dual solution, status, and stats.
    """
    n_vars = qp.H.shape[0]
    x = cp.Variable(n_vars)

    # Wrap H in psd_wrap because DC-OPF Hessian is PSD (zeros on theta)
    H_psd = cp.psd_wrap(qp.H)
    objective = cp.Minimize(0.5 * cp.quad_form(x, H_psd) + qp.g @ x)

    constraints = []
    has_eq = qp.A_eq is not None and qp.A_eq.shape[0] > 0
    has_in = qp.A_in is not None and qp.A_in.shape[0] > 0

    eq_con = None
    in_con = None

    if has_eq:
        eq_con = (qp.A_eq @ x == qp.b_eq)
        constraints.append(eq_con)

    if has_in:
        in_con = (qp.A_in @ x <= qp.b_in)
        constraints.append(in_con)

    prob = cp.Problem(objective, constraints)

    # Solve using Clarabel
    solver_opts: dict[str, Any] = {
        "max_iter": max_iter,
        "tol_gap_abs": tol,
        "tol_gap_rel": tol,
        "tol_feas": tol,
    }
    prob.solve(solver=cp.CLARABEL, **solver_opts)

    # Map CVXPY status to interface status ("optimal" | "infeasible" | "max_iter" | "singular")
    if prob.status == cp.OPTIMAL:
        status = "optimal"
    elif prob.status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE):
        status = "infeasible"
    elif prob.status in (cp.UNBOUNDED, cp.UNBOUNDED_INACCURATE):
        status = "infeasible"
    elif prob.status == cp.USER_LIMIT:
        status = "max_iter"
    elif prob.status == cp.SOLVER_ERROR:
        status = "singular"
    else:
        raise ValueError(f"Unexpected CVXPY status '{prob.status}' could not be mapped cleanly")

    iters = 0
    if prob.solver_stats is not None and prob.solver_stats.num_iters is not None:
        iters = int(prob.solver_stats.num_iters)

    if status == "optimal":
        x_val = np.array(x.value, dtype=np.float64)
        obj_val = float(prob.value)
        # Flip equality dual sign per convention: L = f - lam^T (A_eq x - b_eq) + mu^T (A_in x - b_in)
        if eq_con is not None and eq_con.dual_value is not None:
            lam = -np.array(eq_con.dual_value, dtype=np.float64)
        else:
            lam = np.array([], dtype=np.float64)

        if in_con is not None and in_con.dual_value is not None:
            # Ensure mu >= 0 numerically
            mu = np.maximum(np.array(in_con.dual_value, dtype=np.float64), 0.0)
        else:
            mu = np.array([], dtype=np.float64)

        history = [obj_val]
    else:
        x_val = np.array([], dtype=np.float64)
        obj_val = float("inf")
        lam = np.array([], dtype=np.float64)
        mu = np.array([], dtype=np.float64)
        history = []

    return QPResult(
        x=x_val,
        obj=obj_val,
        lam=lam,
        mu=mu,
        status=status,
        iters=iters,
        history=history,
    )
