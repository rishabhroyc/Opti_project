"""Power system model builders and data structures for OPF.

Author: Rishabh Roy Choudhury (Albert and Einstein)
"""

from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
from scipy.optimize import linprog

from src.opf.qp import QP


@dataclass
class Case:
    """Power system parameters loaded from case data.

    Attributes:
        base_mva: System base power in MVA (typically 100.0).
        ref_bus: 1-based reference bus index (typically 1).
        bus_d: Real power demand d at each bus in MW, shape (n_bus,).
        gen_bus: 1-based bus location for each generator, shape (n_gen,).
        p_min: Minimum generator real power output in MW, shape (n_gen,).
        p_max: Maximum generator real power output in MW, shape (n_gen,).
        b: Linear cost coefficient in $/MWh, shape (n_gen,).
        c: Quadratic cost coefficient in $/(MW^2 h), shape (n_gen,).
        branch_from: 1-based from-bus index for each branch, shape (n_branch,).
        branch_to: 1-based to-bus index for each branch, shape (n_branch,).
        branch_x: Branch series reactance in per unit, shape (n_branch,).
        f_max: Branch long-term rating (RATE_A) in MVA/MW, shape (n_branch,).
    """

    base_mva: float
    ref_bus: int
    bus_d: np.ndarray
    gen_bus: np.ndarray
    p_min: np.ndarray
    p_max: np.ndarray
    b: np.ndarray
    c: np.ndarray
    branch_from: np.ndarray
    branch_to: np.ndarray
    branch_x: np.ndarray
    f_max: np.ndarray


def _parse_matpower_matrix(name: str, text: str) -> np.ndarray:
    """Extract a numeric matrix from MATPOWER .m file content."""
    pattern = rf"mpc\.{name}\s*=\s*\[(.*?)\];"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise ValueError(f"Matrix mpc.{name} not found in case file")

    rows = []
    for line in match.group(1).strip().split(";"):
        cleaned = re.sub(r"%.*", "", line).strip()
        if cleaned:
            rows.append([float(x) for x in cleaned.split()])
    return np.array(rows, dtype=np.float64)


def load_case(filepath: str | Path = "data/case30.m") -> Case:
    """Parse a MATPOWER .m case file into a Case dataclass.

    Only parses parameters needed for DC-OPF / ED:
    - baseMVA, bus demand, reference bus
    - generator limits and quadratic cost coefficients (gencost [2 0 0 3 c2 c1 c0])
    - branch from/to buses, reactance x, and rating rateA

    Args:
        filepath: Path to the MATPOWER .m file.

    Returns:
        Case instance containing system parameters.
    """
    path = Path(filepath)
    if not path.is_absolute() and not path.exists():
        # Fallback relative to repository root
        root = Path(__file__).resolve().parent.parent.parent
        candidate = root / path
        if candidate.exists():
            path = candidate

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Base MVA
    base_mva_match = re.search(r"mpc\.baseMVA\s*=\s*([\d\.]+);", text)
    if not base_mva_match:
        raise ValueError("mpc.baseMVA not found in case file")
    base_mva = float(base_mva_match.group(1))

    # Matrices
    bus = _parse_matpower_matrix("bus", text)
    gen = _parse_matpower_matrix("gen", text)
    branch = _parse_matpower_matrix("branch", text)
    gencost = _parse_matpower_matrix("gencost", text)

    # Reference bus (bus_type == 3)
    ref_buses = bus[bus[:, 1] == 3, 0]
    if len(ref_buses) == 0:
        raise ValueError("No reference bus (type 3) found in case file")
    ref_bus = int(ref_buses[0])

    # Demands (col 2 is Pd in MW)
    bus_d = bus[:, 2].copy()

    # Generator parameters (col 0: bus, col 8: Pmax, col 9: Pmin)
    gen_bus = gen[:, 0].astype(int)
    p_max = gen[:, 8].copy()
    p_min = gen[:, 9].copy()

    # Generator costs: gencost rows [2, startup, shutdown, n, c2, c1, c0]
    # For quadratic cost f(P) = c*P^2 + b*P + a:
    # col 4 is c2 (quadratic coeff c), col 5 is c1 (linear coeff b)
    c = gencost[:, 4].copy()
    b = gencost[:, 5].copy()

    # Branch parameters (col 0: fbus, col 1: tbus, col 3: x, col 5: rateA)
    branch_from = branch[:, 0].astype(int)
    branch_to = branch[:, 1].astype(int)
    branch_x = branch[:, 3].copy()
    f_max = branch[:, 5].copy()

    return Case(
        base_mva=base_mva,
        ref_bus=ref_bus,
        bus_d=bus_d,
        gen_bus=gen_bus,
        p_min=p_min,
        p_max=p_max,
        b=b,
        c=c,
        branch_from=branch_from,
        branch_to=branch_to,
        branch_x=branch_x,
        f_max=f_max,
    )


def build_ed(case: Case, load_scale: float = 1.0) -> QP:
    """Build the Economic Dispatch (ED) quadratic program.

    Formulation:
        minimize    1/2 x^T H x + g^T x = sum_i (b_i P_i + c_i P_i^2)
        subject to  sum_i P_i = D * load_scale          (1 equality)
                    P_i <= P_i^max                      (6 upper limits)
                   -P_i <= -P_i^min                     (6 lower limits)

    Variable ordering:
        x = [P1, P2, P3, P4, P5, P6] in MW (length 6).

    Args:
        case: System parameters from load_case().
        load_scale: Demand scaling factor (default 1.0).

    Returns:
        QP dataclass with H (6, 6), g (6,), A_eq (1, 6), b_eq (1,),
        A_in (12, 6), b_in (12,), and names (6,).
    """
    n_gen = len(case.gen_bus)
    H = np.diag(2.0 * case.c)
    g = case.b.copy()

    # Equality: sum_i P_i = total_demand * load_scale
    total_demand = float(np.sum(case.bus_d) * load_scale)
    A_eq = np.ones((1, n_gen), dtype=np.float64)
    b_eq = np.array([total_demand], dtype=np.float64)

    # Inequalities: 6 upper bounds P_i <= P_max, then 6 lower bounds -P_i <= -P_min
    A_in = np.vstack([
        np.eye(n_gen, dtype=np.float64),
        -np.eye(n_gen, dtype=np.float64),
    ])
    b_in = np.concatenate([case.p_max, -case.p_min])

    names = [f"P{i + 1}" for i in range(n_gen)]

    return QP(
        H=H,
        g=g,
        A_eq=A_eq,
        b_eq=b_eq,
        A_in=A_in,
        b_in=b_in,
        names=names,
    )


def build_dcopf(
    case: Case,
    load_scale: float = 1.0,
    rate_scale: float = 1.0,
) -> QP:
    """Build the DC Optimal Power Flow (DC-OPF) quadratic program.

    Formulation:
        minimize    1/2 x^T H x + g^T x = sum_i (b_i P_i + c_i P_i^2)
        subject to  C_g P - S_base * B_bus * theta = d * load_scale  (30 balance rows)
                    theta_ref = 0                                     (1 ref row)
                    +S_base * diag(1/x) * C * theta <= F_max * rate_scale (41 line upper)
                    -S_base * diag(1/x) * C * theta <= F_max * rate_scale (41 line lower)
                    P <= P_max                                            (6 gen upper)
                    -P <= -P_min                                          (6 gen lower)

    Variable ordering:
        x = [P1..P6, theta1..theta30] in MW and radians (length 36).

    Args:
        case: System parameters from load_case().
        load_scale: Demand scaling factor (default 1.0).
        rate_scale: Branch rating scaling factor (default 1.0).

    Returns:
        QP dataclass with H (36, 36), g (36,), A_eq (31, 36), b_eq (31,),
        A_in (94, 36), b_in (94,), and names (36,).
    """
    n_gen = len(case.gen_bus)
    n_bus = len(case.bus_d)
    n_branch = len(case.branch_x)
    n_vars = n_gen + n_bus  # 36
    S_base = case.base_mva  # 100.0

    # 1. Objective: H = diag(2c) on P, zero on theta; g = b on P, zero on theta
    H = np.zeros((n_vars, n_vars), dtype=np.float64)
    H[:n_gen, :n_gen] = np.diag(2.0 * case.c)

    g = np.zeros(n_vars, dtype=np.float64)
    g[:n_gen] = case.b

    # 2. Network matrices
    # Branch-bus incidence matrix C (n_branch x n_bus): +1 at from-bus, -1 at to-bus
    C = np.zeros((n_branch, n_bus), dtype=np.float64)
    for l in range(n_branch):
        C[l, case.branch_from[l] - 1] = 1.0
        C[l, case.branch_to[l] - 1] = -1.0

    # B_bus = C^T @ diag(1/x) @ C
    inv_x = 1.0 / case.branch_x
    B_bus = C.T @ np.diag(inv_x) @ C

    # Generator-to-bus incidence C_g (n_bus x n_gen)
    C_g = np.zeros((n_bus, n_gen), dtype=np.float64)
    for i in range(n_gen):
        C_g[case.gen_bus[i] - 1, i] = 1.0

    # 3. Equalities: 30 balance rows + 1 reference row (31 total)
    # Balance rows: [C_g, -S_base * B_bus] x = d * load_scale
    A_bal = np.hstack([C_g, -S_base * B_bus])
    b_bal = case.bus_d * load_scale

    # Reference angle row: theta_ref = 0
    ref_idx = case.ref_bus - 1
    A_ref = np.zeros((1, n_vars), dtype=np.float64)
    A_ref[0, n_gen + ref_idx] = 1.0
    b_ref = np.array([0.0], dtype=np.float64)

    A_eq = np.vstack([A_bal, A_ref])
    b_eq = np.concatenate([b_bal, b_ref])

    # 4. Inequalities: 41 line upper, 41 line lower, 6 gen upper, 6 gen lower (94 total)
    # Flow matrix: f = S_base * diag(1/x) * C * theta
    flow_theta = S_base * (inv_x[:, np.newaxis] * C)
    flow_block = np.hstack([np.zeros((n_branch, n_gen), dtype=np.float64), flow_theta])

    f_max_scaled = case.f_max * rate_scale

    # 41 line upper: flow <= F_max * rate_scale
    A_line_up = flow_block
    b_line_up = f_max_scaled

    # 41 line lower: -flow <= F_max * rate_scale
    A_line_low = -flow_block
    b_line_low = f_max_scaled

    # 6 gen upper: P <= P_max
    A_gen_up = np.hstack([np.eye(n_gen, dtype=np.float64), np.zeros((n_gen, n_bus), dtype=np.float64)])
    b_gen_up = case.p_max

    # 6 gen lower: -P <= -P_min
    A_gen_low = np.hstack([-np.eye(n_gen, dtype=np.float64), np.zeros((n_gen, n_bus), dtype=np.float64)])
    b_gen_low = -case.p_min

    A_in = np.vstack([A_line_up, A_line_low, A_gen_up, A_gen_low])
    b_in = np.concatenate([b_line_up, b_line_low, b_gen_up, b_gen_low])

    # Names: P1..P6, theta1..theta30
    names = [f"P{i + 1}" for i in range(n_gen)] + [f"theta{k + 1}" for k in range(n_bus)]

    return QP(
        H=H,
        g=g,
        A_eq=A_eq,
        b_eq=b_eq,
        A_in=A_in,
        b_in=b_in,
        names=names,
    )


def _solve_ed_bisection(
    b: np.ndarray,
    c: np.ndarray,
    p_min: np.ndarray,
    p_max: np.ndarray,
    demand: float,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> np.ndarray:
    """Solve economic dispatch via bisection on marginal cost lambda."""
    lam_low = float(np.min(b))
    lam_high = float(np.max(b + 2.0 * c * p_max) + 10.0)
    for _ in range(max_iter):
        lam = 0.5 * (lam_low + lam_high)
        p = np.clip((lam - b) / (2.0 * c), p_min, p_max)
        diff = float(np.sum(p) - demand)
        if abs(diff) < tol:
            break
        if diff > 0.0:
            lam_high = lam
        else:
            lam_low = lam
    return p


def feasible_start(qp: QP, tol: float = 1e-6) -> np.ndarray | None:
    """Find a feasible starting point x0 for the QP.

    Algorithm (CONTEXT.md section 5):
    1. Try the Economic Dispatch (ED) candidate:
       - Uses a private bisection helper on generator marginal costs.
       - Solves bus angles theta from balance rows with theta_ref = 0.
       - If x_cand satisfies all QP constraints A_in @ x <= b_in + tol, returns x_cand.
    2. Fallback to scipy.optimize.linprog (HiGHS):
       - Solves Phase-1 feasibility LP: min 0 s.t. A_eq x = b_eq, A_in x <= b_in.
       - Note: scipy.optimize.linprog is used strictly for initialization.
       - If feasible, returns x0.
       - If infeasible, returns None.

    Args:
        qp: QP problem instance (ED or DC-OPF).
        tol: Feasibility tolerance for constraint checks (default 1e-6).

    Returns:
        Feasible vector x0 of shape (n_vars,), or None if the problem is infeasible.
    """
    n_vars = len(qp.names)

    # 1. Attempt ED candidate
    if n_vars == 6:
        n_gen = 6
        c = 0.5 * np.diag(qp.H)
        b = qp.g
        p_max = qp.b_in[:n_gen]
        p_min = -qp.b_in[n_gen:2 * n_gen]
        demand = float(qp.b_eq[0])
        p_cand = _solve_ed_bisection(b, c, p_min, p_max, demand)
        x_cand = p_cand
    elif n_vars == 36:
        n_gen = 6
        n_bus = 30
        c = 0.5 * np.diag(qp.H[:n_gen, :n_gen])
        b = qp.g[:n_gen]
        p_max = qp.b_in[82:88]
        p_min = -qp.b_in[88:94]
        demand = float(np.sum(qp.b_eq[:n_bus]))
        p_cand = _solve_ed_bisection(b, c, p_min, p_max, demand)

        # Reference bus column index
        ref_bus_col = int(np.argmax(qp.A_eq[30, n_gen:]))
        non_ref = [k for k in range(n_bus) if k != ref_bus_col]

        A_bal_theta = qp.A_eq[:n_bus, n_gen:]
        rhs_theta = qp.b_eq[:n_bus] - qp.A_eq[:n_bus, :n_gen] @ p_cand

        theta_cand = np.zeros(n_bus, dtype=np.float64)
        theta_cand[non_ref] = np.linalg.solve(A_bal_theta[np.ix_(non_ref, non_ref)], rhs_theta[non_ref])
        x_cand = np.concatenate([p_cand, theta_cand])
    else:
        x_cand = None

    if x_cand is not None:
        eq_ok = np.max(np.abs(qp.A_eq @ x_cand - qp.b_eq)) <= tol
        in_ok = np.all(qp.A_in @ x_cand <= qp.b_in + tol)
        if eq_ok and in_ok:
            return x_cand

    # 2. Fallback to scipy.optimize.linprog (initialization only)
    c_lp = np.zeros(n_vars, dtype=np.float64)
    res = linprog(
        c=c_lp,
        A_ub=qp.A_in,
        b_ub=qp.b_in,
        A_eq=qp.A_eq,
        b_eq=qp.b_eq,
        bounds=(-np.inf, np.inf),
        method="highs",
    )
    if res.success:
        return res.x
    return None



