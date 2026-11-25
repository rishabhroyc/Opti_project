"""Economic Dispatch (ED) solver using the Lambda-Iteration (bisection) method.

Author: Varshith Tadiparthi (Albert and Einstein)
Workstream: Solvers

Solves:
    min  sum_i (b_i * P_i + c_i * P_i^2)
    s.t. sum_i P_i = D
         P_min_i <= P_i <= P_max_i  forall i = 1, ..., n

Mathematical Formulation:
    Lagrangian (under project sign convention):
        L(P, lam, mu^+, mu^-) = sum_i (c_i * P_i^2 + b_i * P_i)
                                - lam * (sum_i P_i - D)
                                + sum_i mu_i^+ * (P_i - P_max_i)
                                + sum_i mu_i^- * (P_min_i - P_i)
    Stationarity:
        dL / dP_i = 2 * c_i * P_i + b_i - lam + mu_i^+ - mu_i^- = 0
    Decoupled dispatch for a given trial marginal price lam:
        P_i(lam) = clip((lam - b_i) / (2 * c_i), P_min_i, P_max_i)
    Monotonicity:
        Since c_i > 0, sum_i P_i(lam) is continuous and monotonically non-decreasing in lam.
        Bisection on lam finds the unique root where sum_i P_i(lam) = D.
"""

from typing import Tuple
import numpy as np

from src.opf.qp import QP, QPResult


def solve_lambda_iteration_raw(
    b: np.ndarray,
    c: np.ndarray,
    p_min: np.ndarray,
    p_max: np.ndarray,
    demand: float,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> Tuple[np.ndarray, float, float, np.ndarray, str, int, list[float]]:
    """Solve Economic Dispatch via bisection on marginal cost lambda using raw NumPy arrays.

    Args:
        b: Linear cost coefficients in $/MWh, shape (n_gen,).
        c: Quadratic cost coefficients in $/(MW^2 h), shape (n_gen,). Must be strictly positive.
        p_min: Minimum generator outputs in MW, shape (n_gen,).
        p_max: Maximum generator outputs in MW, shape (n_gen,).
        demand: Total real power demand in MW.
        tol: Convergence tolerance for power balance residual in MW (default 1e-8).
        max_iter: Maximum bisection iterations (default 100).

    Returns:
        tuple containing:
            - p: Optimal generator dispatch in MW, shape (n_gen,).
            - obj: Total generation cost in $/h.
            - lam: System marginal price (Lagrange multiplier) in $/MWh.
            - mu: Inequality multipliers [mu_upper, mu_lower], shape (2 * n_gen,).
            - status: "optimal" | "infeasible" | "max_iter"
            - iters: Number of iterations performed.
            - history: Objective cost per iteration.
    """
    # 1. Input validation
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    p_min = np.asarray(p_min, dtype=np.float64)
    p_max = np.asarray(p_max, dtype=np.float64)
    demand = float(demand)

    n_gen = len(b)
    if not (len(c) == n_gen and len(p_min) == n_gen and len(p_max) == n_gen):
        raise ValueError(f"Dimension mismatch in generator parameters: {n_gen=}, {len(c)=}, {len(p_min)=}, {len(p_max)=}")

    if not (np.all(np.isfinite(b)) and np.all(np.isfinite(c)) and np.all(np.isfinite(p_min)) and np.all(np.isfinite(p_max)) and np.isfinite(demand)):
        raise ValueError("All inputs must contain finite numeric values.")

    if np.any(c <= 0.0):
        raise ValueError("Quadratic cost coefficients c must be strictly positive for strict convexity.")

    if np.any(p_max < p_min):
        raise ValueError("Generator upper limits p_max must be greater than or equal to lower limits p_min.")

    # 2. Feasibility check on total demand
    total_p_min = float(np.sum(p_min))
    total_p_max = float(np.sum(p_max))

    if demand < total_p_min - tol or demand > total_p_max + tol:
        # Infeasible: demand cannot be served within generator capability bounds
        empty_p = np.zeros(n_gen, dtype=np.float64)
        empty_mu = np.zeros(2 * n_gen, dtype=np.float64)
        return empty_p, float("inf"), float("nan"), empty_mu, "infeasible", 0, []

    # 3. Handle boundary conditions
    marginal_cost_min = b + 2.0 * c * p_min
    marginal_cost_max = b + 2.0 * c * p_max

    # Exact lower boundary: demand equals total minimum generation
    if abs(demand - total_p_min) <= tol:
        p_opt = p_min.copy()
        lam_opt = float(np.min(marginal_cost_min))
        obj_opt = float(np.sum(b * p_opt + c * (p_opt ** 2)))
        # Lower bound active multipliers: mu_i^- = max(0, MC_i - lam)
        mu_upper = np.zeros(n_gen, dtype=np.float64)
        mu_lower = np.maximum(0.0, marginal_cost_min - lam_opt)
        mu_opt = np.concatenate([mu_upper, mu_lower])
        return p_opt, obj_opt, lam_opt, mu_opt, "optimal", 0, [obj_opt]

    # Exact upper boundary: demand equals total maximum generation
    if abs(demand - total_p_max) <= tol:
        p_opt = p_max.copy()
        lam_opt = float(np.max(marginal_cost_max))
        obj_opt = float(np.sum(b * p_opt + c * (p_opt ** 2)))
        # Upper bound active multipliers: mu_i^+ = max(0, lam - MC_i)
        mu_upper = np.maximum(0.0, lam_opt - marginal_cost_max)
        mu_lower = np.zeros(n_gen, dtype=np.float64)
        mu_opt = np.concatenate([mu_upper, mu_lower])
        return p_opt, obj_opt, lam_opt, mu_opt, "optimal", 0, [obj_opt]

    # 4. Bracket initialization
    # Since sum P_i(lam) is monotonically non-decreasing:
    # At lam_low = min(marginal_cost_min), sum P_i(lam_low) <= total_p_min <= demand
    # At lam_high = max(marginal_cost_max), sum P_i(lam_high) >= total_p_max >= demand
    lam_low = float(np.min(marginal_cost_min))
    lam_high = float(np.max(marginal_cost_max))

    history: list[float] = []
    p = np.clip((0.5 * (lam_low + lam_high) - b) / (2.0 * c), p_min, p_max)
    lam = 0.5 * (lam_low + lam_high)

    # 5. Bisection loop
    iters = 0
    status = "max_iter"
    for iters in range(1, max_iter + 1):
        lam = 0.5 * (lam_low + lam_high)
        # Dispatch formula derived from KKT stationarity
        p = np.clip((lam - b) / (2.0 * c), p_min, p_max)

        current_cost = float(np.sum(b * p + c * (p ** 2)))
        history.append(current_cost)

        mismatch = float(np.sum(p) - demand)

        # Check power balance residual convergence
        if abs(mismatch) <= tol:
            status = "optimal"
            break

        # Machine precision limit on interval width
        if (lam_high - lam_low) <= 1e-14 * (1.0 + abs(lam)):
            if abs(mismatch) <= 10.0 * tol:
                status = "optimal"
            break

        if mismatch > 0.0:
            # Over-generation: decrease lambda
            lam_high = lam
        else:
            # Under-generation: increase lambda
            lam_low = lam

    obj = float(np.sum(b * p + c * (p ** 2)))

    # 6. Multipliers calculation from KKT stationarity:
    # 2 * c_i * P_i + b_i - lam + mu_i^+ - mu_i^- = 0
    # MC_i = 2 * c_i * P_i + b_i
    # If at upper bound: mu_i^+ = max(0, lam - MC_i)
    # If at lower bound: mu_i^- = max(0, MC_i - lam)
    mc = b + 2.0 * c * p
    mu_upper = np.maximum(0.0, lam - mc)
    mu_lower = np.maximum(0.0, mc - lam)
    mu = np.concatenate([mu_upper, mu_lower])

    return p, obj, lam, mu, status, iters, history


def solve_lambda_iteration(
    qp: QP,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> QPResult:
    """Solve an Economic Dispatch quadratic program using Lambda-Iteration.

    Conforms to the shared QP interface in src/opf/qp.py.
    Extracts generator cost curves and bounds from the QP matrices.

    Args:
        qp: QP problem specification created by build_ed().
        tol: Power balance residual convergence tolerance (default 1e-8).
        max_iter: Maximum bisection iterations (default 100).

    Returns:
        QPResult instance containing optimal dispatch, objective cost,
        balance multiplier lambda, inequality multipliers mu, status, and stats.
    """
    n_gen = len(qp.names)

    # Validate QP dimensions for pure Economic Dispatch
    if qp.H.shape != (n_gen, n_gen) or len(qp.g) != n_gen:
        raise ValueError(f"Incompatible QP objective dimensions: H {qp.H.shape}, g {qp.g.shape}")

    if qp.A_eq is None or qp.A_eq.shape[0] != 1:
        raise ValueError(f"Economic dispatch requires exactly 1 equality constraint, got {qp.A_eq.shape if qp.A_eq is not None else None}")

    # Extract model parameters
    c = 0.5 * np.diag(qp.H)
    b = qp.g.copy()
    demand = float(qp.b_eq[0])

    # In build_ed(), b_in contains n upper bounds, followed by n negated lower bounds
    p_max = qp.b_in[:n_gen]
    p_min = -qp.b_in[n_gen:2 * n_gen]

    p, obj, lam, mu, status, iters, history = solve_lambda_iteration_raw(
        b=b,
        c=c,
        p_min=p_min,
        p_max=p_max,
        demand=demand,
        tol=tol,
        max_iter=max_iter,
    )

    lam_vec = np.array([lam], dtype=np.float64) if np.isfinite(lam) else np.array([], dtype=np.float64)

    return QPResult(
        x=p,
        obj=obj,
        lam=lam_vec,
        mu=mu,
        status=status,
        iters=iters,
        history=history,
    )
