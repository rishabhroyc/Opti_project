"""Primal Active-Set Quadratic Programming (QP) solver for general convex QPs.

Author: Varshith Tadiparthi (Albert and Einstein)
Workstream: Solvers

Implements:
    Nocedal and Wright, Algorithm 16.3 (Primal Active-Set Method for Convex QP).
    Uses pure NumPy (no external QP solvers, optimizers, or matrix perturbations).

Mathematical Formulation:
    minimize    q(x) = 1/2 * x^T @ H @ x + g^T @ x
    subject to  A_eq @ x = b_eq
                A_in @ x <= b_in

Sign Convention (CONTEXT.md section 4 & src/opf/qp.py):
    Lagrangian:
        L(x, lam, mu) = 1/2 * x^T @ H @ x + g^T @ x - lam^T @ (A_eq @ x - b_eq) + mu^T @ (A_in @ x - b_in)
    Stationarity:
        H @ x + g - A_eq^T @ lam + A_in^T @ mu = 0
    - Equality multipliers lam are positive (+3.7892 in the base case) and equal d(cost)/d(b_eq).
    - Inequality multipliers mu >= 0 at optimality.
    - Status values: "optimal" | "infeasible" | "max_iter" | "singular".
"""

from typing import Optional, Tuple
import numpy as np

from src.opf.qp import QP, QPResult
from src.opf.models import feasible_start


def _validate_qp(qp: QP) -> Tuple[np.ndarray, np.ndarray]:
    """Validate QP inputs for convexity, shapes, and symmetry.

    Validates:
        - H is square (n, n) and symmetric within numerical tolerance.
        - g has shape (n,).
        - H is positive semidefinite (min eigenvalue >= -1e-7).
        - Constraint dimensions match variable count n.

    Returns:
        Symmetrized Hessian H and gradient g.
    """
    if qp.H.ndim != 2 or qp.H.shape[0] != qp.H.shape[1]:
        raise ValueError(f"Hessian H must be a 2D square matrix, got shape {qp.H.shape}")

    n = qp.H.shape[0]
    if len(qp.g) != n:
        raise ValueError(f"Linear term g shape {qp.g.shape} does not match H shape ({n}, {n})")

    if not (np.all(np.isfinite(qp.H)) and np.all(np.isfinite(qp.g))):
        raise ValueError("H and g must contain finite numeric values.")

    # Symmetry check
    h_scale = max(1.0, float(np.max(np.abs(qp.H))))
    max_asymm = float(np.max(np.abs(qp.H - qp.H.T)))
    if max_asymm > 1e-7 * h_scale:
        raise ValueError(f"Hessian H is not symmetric: max |H - H^T| = {max_asymm:.2e}")

    # Symmetrize numerical noise without altering problem structure
    H = 0.5 * (qp.H + qp.H.T)

    # Positive semidefiniteness check (convexity requirement)
    min_eig = float(np.min(np.linalg.eigvalsh(H)))
    if min_eig < -1e-7 * h_scale:
        raise ValueError(f"Hessian H is not positive semidefinite: min eigenvalue = {min_eig:.2e}")

    # Validate constraint dimensions
    if qp.A_eq is not None and len(qp.A_eq) > 0:
        if qp.A_eq.ndim != 2 or qp.A_eq.shape[1] != n or len(qp.b_eq) != qp.A_eq.shape[0]:
            raise ValueError(f"Equality constraints dimension mismatch: A_eq {qp.A_eq.shape}, b_eq {qp.b_eq.shape}")

    if qp.A_in is not None and len(qp.A_in) > 0:
        if qp.A_in.ndim != 2 or qp.A_in.shape[1] != n or len(qp.b_in) != qp.A_in.shape[0]:
            raise ValueError(f"Inequality constraints dimension mismatch: A_in {qp.A_in.shape}, b_in {qp.b_in.shape}")

    return H, qp.g.copy()


def _init_working_set(
    x: np.ndarray,
    A_eq: Optional[np.ndarray],
    A_in: Optional[np.ndarray],
    b_in: Optional[np.ndarray],
    tol: float = 1e-6,
) -> list[int]:
    """Find active inequality constraints at x that maintain Linear Independence (LICQ).

    Adds active inequality rows one at a time only if they increase the rank of
    the combined constraint matrix [A_eq; A_W].

    Args:
        x: Feasible point, shape (n,).
        A_eq: Equality matrix, shape (m_eq, n) or None.
        A_in: Inequality matrix, shape (m_in, n) or None.
        b_in: Inequality bounds, shape (m_in,) or None.
        tol: Tolerance for considering a constraint active (|A_in[i] @ x - b_in[i]| <= tol).

    Returns:
        List of active inequality indices in the initial working set W.
    """
    if A_in is None or b_in is None or len(b_in) == 0:
        return []

    m_in = len(b_in)
    n = len(x)

    # Base constraint rows: start with equality constraints
    if A_eq is not None and len(A_eq) > 0:
        current_rows = [A_eq[i] for i in range(A_eq.shape[0])]
        current_rank = int(np.linalg.matrix_rank(np.array(current_rows, dtype=np.float64)))
    else:
        current_rows = []
        current_rank = 0

    working_set: list[int] = []

    for i in range(m_in):
        slack = float(b_in[i] - A_in[i] @ x)
        if abs(slack) <= tol:  # Constraint is numerically active
            candidate_rows = current_rows + [A_in[i]]
            candidate_mat = np.array(candidate_rows, dtype=np.float64)
            new_rank = int(np.linalg.matrix_rank(candidate_mat))
            if new_rank > current_rank:
                current_rows = candidate_rows
                current_rank = new_rank
                working_set.append(i)
                if current_rank == n:
                    break

    return working_set


def _solve_kkt_subproblem(
    H: np.ndarray,
    g_k: np.ndarray,
    A_eq: Optional[np.ndarray],
    A_in: Optional[np.ndarray],
    working_set: list[int],
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Solve the equality-constrained KKT subproblem for search direction p and multipliers.

    KKT System:
        [ H            -A_eq^T    A_W^T ] [   p   ]   [ -g_k ]
        [ A_eq            0         0   ] [  lam  ] = [   0  ]
        [ A_W             0         0   ] [  mu_W ]   [   0  ]

    Sign convention:
        - Equality column has -A_eq^T (yields positive lam equal to LMP).
        - Inequality column has +A_W^T (yields non-negative mu_W).

    Args:
        H: Symmetrized PSD Hessian matrix, shape (n, n).
        g_k: Current gradient H @ x_k + g, shape (n,).
        A_eq: Equality matrix, shape (m_eq, n) or None.
        A_in: Inequality matrix, shape (m_in, n) or None.
        working_set: Active inequality indices W.

    Returns:
        (p, lam, mu_W) or None if the KKT matrix is singular.
    """
    n = H.shape[0]
    m_eq = A_eq.shape[0] if (A_eq is not None and len(A_eq) > 0) else 0
    m_w = len(working_set)

    A_w = A_in[working_set, :] if (m_w > 0 and A_in is not None) else None

    dim = n + m_eq + m_w
    kkt_mat = np.zeros((dim, dim), dtype=np.float64)
    rhs = np.zeros(dim, dtype=np.float64)

    # 1. Hessian block
    kkt_mat[:n, :n] = H
    rhs[:n] = -g_k

    # 2. Equality constraint block
    if m_eq > 0 and A_eq is not None:
        kkt_mat[:n, n : n + m_eq] = -A_eq.T
        kkt_mat[n : n + m_eq, :n] = A_eq

    # 3. Active inequality constraint block
    if m_w > 0 and A_w is not None:
        kkt_mat[:n, n + m_eq :] = A_w.T
        kkt_mat[n + m_eq :, :n] = A_w

    try:
        sol = np.linalg.solve(kkt_mat, rhs)
    except np.linalg.LinAlgError:
        return None

    p = sol[:n]
    lam = sol[n : n + m_eq] if m_eq > 0 else np.array([], dtype=np.float64)
    mu_w = sol[n + m_eq :] if m_w > 0 else np.array([], dtype=np.float64)

    return p, lam, mu_w


def solve(
    qp: QP,
    x0: Optional[np.ndarray] = None,
    tol: float = 1e-8,
    max_iter: int = 500,
) -> QPResult:
    """Solve a convex quadratic program using the Primal Active-Set Method (Algorithm 16.3).

    Maintains primal feasibility x_k at every iteration while identifying the optimal
    active working set W*.

    Args:
        qp: QP problem specification.
        x0: Optional initial feasible point. If None, feasible_start(qp) is used.
        tol: Numerical tolerance for convergence, feasibility, and active sets (default 1e-8).
        max_iter: Maximum number of active-set iterations (default 500).

    Returns:
        QPResult containing primal solution, objective value, equality multipliers lam,
        full inequality multipliers mu (in original row order), status, and history.
    """
    # 1. Validate inputs and extract symmetrized problem matrices
    H, g = _validate_qp(qp)
    n = H.shape[0]
    m_eq = qp.A_eq.shape[0] if (qp.A_eq is not None and len(qp.A_eq) > 0) else 0
    m_in = qp.A_in.shape[0] if (qp.A_in is not None and len(qp.A_in) > 0) else 0

    # 2. Establish feasible starting point x0
    if x0 is None:
        x_init = feasible_start(qp, tol=1e-6)
        if x_init is None:
            # Check if origin x = 0 is trivially feasible
            origin = np.zeros(n, dtype=np.float64)
            eq_ok = (m_eq == 0) or np.max(np.abs(qp.A_eq @ origin - qp.b_eq)) <= tol
            in_ok = (m_in == 0) or np.all(qp.A_in @ origin <= qp.b_in + tol)
            if eq_ok and in_ok:
                x_init = origin
            else:
                return QPResult(
                    x=np.array([], dtype=np.float64),
                    obj=float("inf"),
                    lam=np.array([], dtype=np.float64),
                    mu=np.array([], dtype=np.float64),
                    status="infeasible",
                    iters=0,
                    history=[],
                )
    else:
        x_init = np.array(x0, dtype=np.float64).copy()
        if len(x_init) != n:
            raise ValueError(f"Initial point x0 length {len(x_init)} does not match problem dimension {n}")
        if m_eq > 0 and np.max(np.abs(qp.A_eq @ x_init - qp.b_eq)) > 1e-5:
            raise ValueError("Provided initial point x0 violates equality constraints.")
        if m_in > 0 and np.any(qp.A_in @ x_init > qp.b_in + 1e-5):
            raise ValueError("Provided initial point x0 violates inequality constraints.")

    x = x_init.copy()

    # 3. Initialize working set with linearly independent active constraints
    working_set = _init_working_set(x, qp.A_eq, qp.A_in, qp.b_in, tol=1e-6)

    history: list[float] = []

    # 4. Main Active-Set Iteration Loop
    for iters in range(1, max_iter + 1):
        current_obj = float(0.5 * x @ H @ x + g @ x)
        history.append(current_obj)

        g_k = H @ x + g

        # Solve KKT equality subproblem on the working manifold
        kkt_sol = _solve_kkt_subproblem(H, g_k, qp.A_eq, qp.A_in, working_set)
        if kkt_sol is None:
            return QPResult(
                x=x,
                obj=current_obj,
                lam=np.array([], dtype=np.float64),
                mu=np.array([], dtype=np.float64),
                status="singular",
                iters=iters,
                history=history,
            )

        p, lam, mu_w = kkt_sol

        # Check if step p is practically zero (subspace minimizer reached)
        if np.max(np.abs(p)) <= tol:
            # Check dual feasibility (multiplier non-negativity) for active inequalities
            if len(working_set) == 0 or np.all(mu_w >= -tol):
                # All KKT conditions satisfied: primal feasible, stationarity, dual feasible
                full_mu = np.zeros(m_in, dtype=np.float64)
                for idx, constr_i in enumerate(working_set):
                    full_mu[constr_i] = max(0.0, float(mu_w[idx]))

                return QPResult(
                    x=x,
                    obj=current_obj,
                    lam=lam,
                    mu=full_mu,
                    status="optimal",
                    iters=iters,
                    history=history,
                )
            else:
                # Suboptimal constraint active: remove the one with most negative multiplier
                # Break ties deterministically by lowest constraint index
                most_neg_val = 0.0
                drop_idx = -1
                for idx, constr_i in enumerate(working_set):
                    if mu_w[idx] < most_neg_val - 1e-14:
                        most_neg_val = mu_w[idx]
                        drop_idx = idx

                if drop_idx >= 0:
                    working_set.pop(drop_idx)
                # Keep x unchanged and re-solve on the relaxed manifold
                continue

        # Step p is nonzero: perform ratio test across all inactive constraints
        alpha = 1.0
        blocking_constraint = -1

        if m_in > 0 and qp.A_in is not None and qp.b_in is not None:
            inactive = [i for i in range(m_in) if i not in working_set]
            for i in inactive:
                a_i = qp.A_in[i]
                a_dot_p = float(a_i @ p)
                if a_dot_p > 1e-12:  # Directional derivative is strictly positive (moving toward boundary)
                    slack = max(0.0, float(qp.b_in[i] - a_i @ x))
                    alpha_i = slack / a_dot_p
                    # Deterministic tie-breaking by lowest index: inactive is sorted, so strict '<' retains first
                    if alpha_i < alpha - 1e-14:
                        alpha = alpha_i
                        blocking_constraint = i

        # Update iterate along search step
        x = x + alpha * p

        # Preserve numerical feasibility on active inequalities against floating-point drift
        if alpha < 1.0 and blocking_constraint >= 0:
            working_set.append(blocking_constraint)

    # Maximum iterations reached without convergence
    final_obj = float(0.5 * x @ H @ x + g @ x)
    full_mu = np.zeros(m_in, dtype=np.float64)
    if kkt_sol is not None:
        _, lam_final, mu_w_final = kkt_sol
        for idx, constr_i in enumerate(working_set):
            if idx < len(mu_w_final):
                full_mu[constr_i] = max(0.0, float(mu_w_final[idx]))
    else:
        lam_final = np.array([], dtype=np.float64)

    return QPResult(
        x=x,
        obj=final_obj,
        lam=lam_final,
        mu=full_mu,
        status="max_iter",
        iters=max_iter,
        history=history,
    )
