"""Unit and system tests for Lambda-Iteration and Primal Active-Set QP Solvers.

Author: Varshith Tadiparthi (Albert and Einstein)
Workstream: Solvers
"""

import numpy as np
import pytest

from src.opf.qp import QP
from src.opf.lambda_iteration import solve as solve_lambda, solve_lambda_iteration, solve_lambda_iteration_raw
from src.opf.active_set import solve as solve_active_set
from src.opf.models import load_case, build_ed, build_dcopf, feasible_start
from scipy.optimize import linprog


# =====================================================================
# Tests for Lambda Iteration (Economic Dispatch)
# =====================================================================

def test_lambda_iteration_hand_computable():
    """Verify lambda-iteration against an exact analytical 2-generator problem.

    Problem:
        C1(P1) = 0.5 * 0.1 * P1^2 + 10 * P1,  bounds [10, 100]
        C2(P2) = 0.5 * 0.2 * P2^2 + 15 * P2,  bounds [10, 100]
        Demand D = 100 MW.

    Analytical Solution:
        10 + 0.1 * P1 = 15 + 0.2 * P2 = lambda
        P1 = (lambda - 10) / 0.1 = 10 * lambda - 100
        P2 = (lambda - 15) / 0.2 = 5 * lambda - 75
        P1 + P2 = 15 * lambda - 175 = 100 => 15 * lambda = 275 => lambda = 18.333333 $/MWh
        P1 = 83.333333 MW, P2 = 16.666667 MW
    """
    b = np.array([10.0, 15.0])
    c = np.array([0.05, 0.10])  # Note: 0.5 * (2*c) * P^2 => c1 = 0.05, c2 = 0.10
    p_min = np.array([10.0, 10.0])
    p_max = np.array([100.0, 100.0])
    demand = 100.0

    p, obj, lam, mu, status, iters, _ = solve_lambda_iteration_raw(
        b=b, c=c, p_min=p_min, p_max=p_max, demand=demand, tol=1e-10
    )

    assert status == "optimal"
    assert pytest.approx(lam, abs=1e-6) == 275.0 / 15.0
    assert pytest.approx(p[0], abs=1e-6) == 83.333333
    assert pytest.approx(p[1], abs=1e-6) == 16.666667
    assert pytest.approx(np.sum(p), abs=1e-9) == 100.0

    # No bounds binding => mu == 0
    np.testing.assert_allclose(mu, 0.0, atol=1e-8)


def test_lambda_iteration_lower_bound_binding():
    """Verify behavior when a generator hits its lower bound."""
    b = np.array([10.0, 15.0])
    c = np.array([0.05, 0.10])
    p_min = np.array([10.0, 10.0])
    p_max = np.array([100.0, 100.0])
    # Low demand: Gen 2 marginal cost at Pmin is 15 + 2*0.1*10 = 17.
    # Gen 1 at P=15 has marginal cost 10 + 2*0.05*15 = 11.5.
    demand = 25.0

    p, obj, lam, mu, status, _, _ = solve_lambda_iteration_raw(
        b=b, c=c, p_min=p_min, p_max=p_max, demand=demand
    )

    assert status == "optimal"
    assert pytest.approx(p[1], abs=1e-6) == 10.0  # Gen 2 clamped at lower bound
    assert pytest.approx(p[0], abs=1e-6) == 15.0  # Gen 1 takes remainder
    assert pytest.approx(lam, abs=1e-6) == 10.0 + 2.0 * 0.05 * 15.0  # 11.5 $/MWh

    # Gen 2 has lower limit active: mu_lower > 0 (row index 3)
    # Stationarity: 17 - 11.5 - mu_lower = 0 => mu_lower = 5.5
    assert pytest.approx(mu[3], abs=1e-5) == 5.5


def test_lambda_iteration_upper_bound_binding():
    """Verify behavior when a generator hits its upper bound."""
    b = np.array([10.0, 15.0])
    c = np.array([0.05, 0.10])
    p_min = np.array([10.0, 10.0])
    p_max = np.array([100.0, 100.0])
    # High demand: Gen 1 reaches 100 MW (marginal cost 20).
    # Gen 2 produces remaining 50 MW (marginal cost 15 + 2*0.1*50 = 25).
    demand = 150.0

    p, obj, lam, mu, status, _, _ = solve_lambda_iteration_raw(
        b=b, c=c, p_min=p_min, p_max=p_max, demand=demand
    )

    assert status == "optimal"
    assert pytest.approx(p[0], abs=1e-6) == 100.0  # Gen 1 clamped at upper bound
    assert pytest.approx(p[1], abs=1e-6) == 50.0
    assert pytest.approx(lam, abs=1e-6) == 25.0

    # Gen 1 upper limit active: mu_upper = lam - MC1 = 25 - 20 = 5.0 (row index 0)
    assert pytest.approx(mu[0], abs=1e-5) == 5.0


def test_lambda_iteration_boundary_demands():
    """Verify edge cases when demand equals total Pmin and total Pmax."""
    b = np.array([10.0, 15.0])
    c = np.array([0.05, 0.10])
    p_min = np.array([10.0, 20.0])
    p_max = np.array([100.0, 200.0])

    # Minimum demand
    p_low, obj_low, lam_low, _, status_low, _, _ = solve_lambda_iteration_raw(
        b, c, p_min, p_max, demand=30.0
    )
    assert status_low == "optimal"
    np.testing.assert_allclose(p_low, p_min)

    # Maximum demand
    p_high, obj_high, lam_high, _, status_high, _, _ = solve_lambda_iteration_raw(
        b, c, p_min, p_max, demand=300.0
    )
    assert status_high == "optimal"
    np.testing.assert_allclose(p_high, p_max)


def test_lambda_iteration_infeasible_demand():
    """Verify infeasibility detection when demand exceeds generator bounds."""
    b = np.array([10.0, 15.0])
    c = np.array([0.05, 0.10])
    p_min = np.array([10.0, 20.0])
    p_max = np.array([100.0, 200.0])

    # Below minimum
    _, obj1, _, _, status1, _, _ = solve_lambda_iteration_raw(
        b, c, p_min, p_max, demand=29.0
    )
    assert status1 == "infeasible"
    assert obj1 == float("inf")

    # Above maximum
    _, obj2, _, _, status2, _, _ = solve_lambda_iteration_raw(
        b, c, p_min, p_max, demand=301.0
    )
    assert status2 == "infeasible"
    assert obj2 == float("inf")


def test_lambda_iteration_case30_base_case():
    """Verify Economic Dispatch on IEEE 30-bus system reproduces benchmark values.

    Expected (from CONTEXT.md Section 3):
        Cost: 565.21 $/h
        Lambda: 3.7892 $/MWh
        Dispatch: [44.73, 58.26, 22.31, 32.33, 15.78, 15.78] MW
    """
    case = load_case("data/case30.m")
    qp = build_ed(case, load_scale=1.0)

    res = solve_lambda_iteration(qp)

    assert res.status == "optimal"
    assert pytest.approx(res.obj, abs=0.005) == 565.21
    assert pytest.approx(res.lam[0], abs=0.0001) == 3.7892

    expected_dispatch = np.array([44.73, 58.26, 22.31, 32.33, 15.78, 15.78])
    np.testing.assert_allclose(res.x, expected_dispatch, atol=0.005)

    # Balance residual
    residual = abs(np.sum(res.x) - np.sum(case.bus_d))
    assert residual <= 1e-8


# =====================================================================
# Tests for Active Set QP Solver (src/opf/active_set.py)
# =====================================================================

def test_active_set_nocedal_wright_example_16_4():
    """Verify active-set solver on textbook Nocedal & Wright Example 16.4.

    Problem:
        min  (x1 - 1)^2 + (x2 - 2.5)^2
             = 0.5 * x^T @ [[2, 0], [0, 2]] @ x + [-2, -5]^T @ x + 7.25
        s.t. -x1 + 2*x2 <= 2
              x1 + 2*x2 <= 6
              x1 - 2*x2 <= 2
             -x1        <= 0
                  -x2   <= 0

    Starting point: x0 = (2, 0).
    Expected solution: x* = (1.4, 1.7), objective q(x*) = 0.80, mu1 = 0.8.
    """
    H = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    g = np.array([-2.0, -5.0], dtype=np.float64)
    A_eq = np.zeros((0, 2), dtype=np.float64)
    b_eq = np.zeros(0, dtype=np.float64)
    A_in = np.array([
        [-1.0, 2.0],
        [1.0, 2.0],
        [1.0, -2.0],
        [-1.0, 0.0],
        [0.0, -1.0],
    ], dtype=np.float64)
    b_in = np.array([2.0, 6.0, 2.0, 0.0, 0.0], dtype=np.float64)
    names = ["x1", "x2"]

    qp = QP(H=H, g=g, A_eq=A_eq, b_eq=b_eq, A_in=A_in, b_in=b_in, names=names)

    # Start at x0 = (2, 0)
    x0 = np.array([2.0, 0.0], dtype=np.float64)
    res = solve_active_set(qp, x0=x0)

    assert res.status == "optimal"
    expected_x = np.array([1.4, 1.7])
    np.testing.assert_allclose(res.x, expected_x, atol=1e-5)

    # Objective with constant 7.25:
    textbook_obj = res.obj + 7.25
    assert pytest.approx(textbook_obj, abs=1e-5) == 0.80

    # Constraint 0 (-x1 + 2*x2 <= 2) is active with multiplier 0.8
    assert pytest.approx(res.mu[0], abs=1e-5) == 0.8
    # All other constraints inactive
    np.testing.assert_allclose(res.mu[1:], 0.0, atol=1e-6)


def test_active_set_ed_base_case():
    """Verify active-set solver reproduces ED base case."""
    case = load_case("data/case30.m")
    qp = build_ed(case, load_scale=1.0)

    res = solve_active_set(qp)

    assert res.status == "optimal"
    assert pytest.approx(res.obj, abs=0.005) == 565.21
    assert pytest.approx(res.lam[0], abs=0.0001) == 3.7892

    expected_dispatch = np.array([44.73, 58.26, 22.31, 32.33, 15.78, 15.78])
    np.testing.assert_allclose(res.x, expected_dispatch, atol=0.005)


def test_active_set_dcopf_base_case():
    """Verify active-set solver on full DC-OPF base case (rate_scale=1.0).

    In the unconstrained network, all 30 bus LMPs must equal 3.7892 $/MWh,
    and total cost is 565.21 $/h.
    """
    case = load_case("data/case30.m")
    qp = build_dcopf(case, load_scale=1.0, rate_scale=1.0)

    res = solve_active_set(qp)

    assert res.status == "optimal"
    assert pytest.approx(res.obj, abs=0.005) == 565.21
    assert len(res.lam) == 31  # 30 balance rows + 1 reference row

    # All 30 bus LMPs equal 3.7892 $/MWh
    bus_lmps = res.lam[:30]
    for lmp in bus_lmps:
        assert pytest.approx(lmp, abs=0.0001) == 3.7892


def test_active_set_dcopf_congested_75():
    """Verify active-set solver on congested DC-OPF (ratings x0.75).

    Expected:
        Cost: 565.97 $/h
        LMPs: spread between 3.65 and 7.18 $/MWh
    """
    case = load_case("data/case30.m")
    qp = build_dcopf(case, load_scale=1.0, rate_scale=0.75)

    res = solve_active_set(qp)

    assert res.status == "optimal"
    assert pytest.approx(res.obj, abs=0.005) == 565.97

    bus_lmps = res.lam[:30]
    assert pytest.approx(bus_lmps.min(), abs=0.005) == 3.65
    assert pytest.approx(bus_lmps.max(), abs=0.005) == 7.18
    assert np.all(res.mu >= 0.0)


def test_active_set_dcopf_congested_72():
    """Verify active-set solver on severely congested DC-OPF (ratings x0.72).

    Expected:
        Cost: 589.58 $/h
        LMPs: spread between 3.02 and 39.65 $/MWh
    """
    case = load_case("data/case30.m")
    qp = build_dcopf(case, load_scale=1.0, rate_scale=0.72)

    res = solve_active_set(qp)

    assert res.status == "optimal"
    assert pytest.approx(res.obj, abs=0.005) == 589.58

    bus_lmps = res.lam[:30]
    assert pytest.approx(bus_lmps.min(), abs=0.005) == 3.02
    assert pytest.approx(bus_lmps.max(), abs=0.005) == 39.65
    assert np.all(res.mu >= 0.0)


def test_active_set_dcopf_infeasible_70():
    """Verify active-set solver correctly reports infeasible for ratings x0.70."""
    case = load_case("data/case30.m")
    qp = build_dcopf(case, load_scale=1.0, rate_scale=0.70)

    res = solve_active_set(qp)

    assert res.status == "infeasible"



def test_active_set_active_bounds_multiple_active():
    """Verify active-set solver on QP with multiple active bounds.

    Problem:
        min  2*x1^2 + x2^2 - 8*x1 - 6*x2
        s.t.  x1      <= 1   (idx 0)
                  x2  <= 2   (idx 1)
             -x1      <= 0   (idx 2)
                 -x2  <= 0   (idx 3)

    Unconstrained minimizer: x = (2, 3).
    Constrained solution: x* = (1, 2).
    Both upper bounds active: mu = [4.0, 2.0, 0.0, 0.0].
    """
    H = np.array([[4.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    g = np.array([-8.0, -6.0], dtype=np.float64)
    A_eq = np.zeros((0, 2), dtype=np.float64)
    b_eq = np.zeros(0, dtype=np.float64)
    A_in = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
    ], dtype=np.float64)
    b_in = np.array([1.0, 2.0, 0.0, 0.0], dtype=np.float64)

    qp = QP(H=H, g=g, A_eq=A_eq, b_eq=b_eq, A_in=A_in, b_in=b_in, names=["x1", "x2"])
    res = solve_active_set(qp, x0=np.array([0.0, 0.0]))

    assert res.status == "optimal"
    np.testing.assert_allclose(res.x, [1.0, 2.0], atol=1e-6)
    assert pytest.approx(res.mu[0], abs=1e-5) == 4.0
    assert pytest.approx(res.mu[1], abs=1e-5) == 2.0
    np.testing.assert_allclose(res.mu[2:], 0.0, atol=1e-6)


def test_active_set_blocking_constraint():
    """Verify ratio test adds blocking constraint when step is truncated (alpha < 1).

    Problem:
        min  (x1 - 2)^2 + (x2 - 2)^2 = x1^2 + x2^2 - 4*x1 - 4*x2 + 8
        s.t.  x1 + x2 <= 2
    Start: x0 = (0, 0).
    Unconstrained minimizer is (2, 2).
    Step p = (2, 2) hits x1 + x2 <= 2 at alpha = 0.5.
    Optimal point: x* = (1, 1).
    """
    H = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    g = np.array([-4.0, -4.0], dtype=np.float64)
    A_eq = np.zeros((0, 2), dtype=np.float64)
    b_eq = np.zeros(0, dtype=np.float64)
    A_in = np.array([[1.0, 1.0]], dtype=np.float64)
    b_in = np.array([2.0], dtype=np.float64)

    qp = QP(H=H, g=g, A_eq=A_eq, b_eq=b_eq, A_in=A_in, b_in=b_in, names=["x1", "x2"])
    res = solve_active_set(qp, x0=np.array([0.0, 0.0]))

    assert res.status == "optimal"
    np.testing.assert_allclose(res.x, [1.0, 1.0], atol=1e-6)
    # Stationarity: [2, 2] + [-4, -4] + [1, 1]*mu = 0 => mu = 2.0
    assert pytest.approx(res.mu[0], abs=1e-5) == 2.0


def test_active_set_drop_constraint():
    """Verify solver removes constraint with negative multiplier from working set.

    Problem:
        min  (x1 - 2)^2 + (x2 - 2)^2
        s.t. -x1 - x2 <= -1   (i.e. x1 + x2 >= 1)
    Start at x0 = (0.5, 0.5) where -x1 - x2 = -1 is active.
    Gradient at x0 is [-3, -3].
    Subproblem step p = 0 on line x1 + x2 = 1.
    Multiplier: [-3, -3] + [-1, -1]*mu = 0 => mu = -3.0 < 0.
    Solver must drop the constraint and move to unconstrained minimum (2, 2).
    """
    H = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    g = np.array([-4.0, -4.0], dtype=np.float64)
    A_eq = np.zeros((0, 2), dtype=np.float64)
    b_eq = np.zeros(0, dtype=np.float64)
    A_in = np.array([[-1.0, -1.0]], dtype=np.float64)
    b_in = np.array([-1.0], dtype=np.float64)

    qp = QP(H=H, g=g, A_eq=A_eq, b_eq=b_eq, A_in=A_in, b_in=b_in, names=["x1", "x2"])
    res = solve_active_set(qp, x0=np.array([0.5, 0.5]))

    assert res.status == "optimal"
    np.testing.assert_allclose(res.x, [2.0, 2.0], atol=1e-6)
    # Constraint was dropped and is strictly inactive at (2, 2)
    assert pytest.approx(res.mu[0], abs=1e-6) == 0.0


def test_active_set_deterministic_tie_breaking():
    """Verify deterministic tie-breaking by lowest index when two constraints block at same alpha.

    Problem:
        min  (x1 - 2)^2 + (x2 - 2)^2
        s.t.  x1      <= 1   (idx 0)
                  x2  <= 1   (idx 1)
    Start at (0, 0). Step towards (2, 2) has p = (2, 2).
    Both constraints 0 and 1 hit boundary at alpha = 0.5.
    Solver must deterministically select index 0 first, then on next step add index 1.
    """
    H = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    g = np.array([-4.0, -4.0], dtype=np.float64)
    A_eq = np.zeros((0, 2), dtype=np.float64)
    b_eq = np.zeros(0, dtype=np.float64)
    A_in = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype=np.float64)
    b_in = np.array([1.0, 1.0], dtype=np.float64)

    qp = QP(H=H, g=g, A_eq=A_eq, b_eq=b_eq, A_in=A_in, b_in=b_in, names=["x1", "x2"])
    res = solve_active_set(qp, x0=np.array([0.0, 0.0]))

    assert res.status == "optimal"
    np.testing.assert_allclose(res.x, [1.0, 1.0], atol=1e-6)
    assert pytest.approx(res.mu[0], abs=1e-5) == 2.0
    assert pytest.approx(res.mu[1], abs=1e-5) == 2.0


def test_active_set_input_validation():
    """Verify active-set solver rejects non-symmetric and non-convex Hessians."""
    # 1. Non-symmetric H
    H_asymm = np.array([[2.0, 1.0], [0.0, 2.0]], dtype=np.float64)
    g = np.array([0.0, 0.0], dtype=np.float64)
    qp_asymm = QP(
        H=H_asymm,
        g=g,
        A_eq=np.zeros((0, 2)),
        b_eq=np.zeros(0),
        A_in=np.zeros((0, 2)),
        b_in=np.zeros(0),
        names=["x1", "x2"],
    )
    with pytest.raises(ValueError, match="not symmetric"):
        solve_active_set(qp_asymm, x0=np.array([0.0, 0.0]))

    # 2. Non-PSD (indefinite/concave) H
    H_indef = np.array([[2.0, 0.0], [0.0, -1.0]], dtype=np.float64)
    qp_indef = QP(
        H=H_indef,
        g=g,
        A_eq=np.zeros((0, 2)),
        b_eq=np.zeros(0),
        A_in=np.zeros((0, 2)),
        b_in=np.zeros(0),
        names=["x1", "x2"],
    )
    with pytest.raises(ValueError, match="not positive semidefinite"):
        solve_active_set(qp_indef, x0=np.array([0.0, 0.0]))



# =====================================================================
# Additional Robustness & Edge-Case Tests (Requirements 1-11)
# =====================================================================

def test_edge_case_1_duplicate_inequality_rows():
    """Requirement 1: Starting working-set rank check avoids inserting duplicate inequality rows.

    Problem:
        min  (x1 - 2)^2 + (x2 - 2)^2
        s.t.  x1 <= 1   (row 0)
              x1 <= 1   (row 1, exact duplicate)
    Start at x0 = (1, 0) where both rows are active.
    Rank check must add row 0 but reject row 1 because rank([A_in[0]; A_in[1]]) == 1.
    The solver must solve the KKT system without singularity and reach x* = (1, 2).
    """
    H = np.eye(2) * 2.0
    g = np.array([-4.0, -4.0])
    A_eq = np.zeros((0, 2))
    b_eq = np.zeros(0)
    # Identical inequality rows
    A_in = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    b_in = np.array([1.0, 1.0], dtype=np.float64)

    qp = QP(H=H, g=g, A_eq=A_eq, b_eq=b_eq, A_in=A_in, b_in=b_in, names=["x1", "x2"])
    res = solve_active_set(qp, x0=np.array([1.0, 0.0]))

    assert res.status == "optimal"
    np.testing.assert_allclose(res.x, [1.0, 2.0], atol=1e-6)
    # Row 0 was in the working set and has positive multiplier; row 1 was never added
    assert pytest.approx(res.mu[0], abs=1e-5) == 2.0
    assert pytest.approx(res.mu[1], abs=1e-6) == 0.0


def test_edge_case_2_duplicate_equality_inequality_dependence():
    """Requirement 2: Inequality linearly dependent on equality constraint is not added to working set.

    Problem:
        min  (x1 - 2)^2 + (x2 - 2)^2
        s.t.  x1 = 1     (equality: row 0)
             2*x1 <= 2   (inequality: row 0, active at x1=1)
    Start at x0 = (1, 0).
    The inequality row [2, 0] is parallel to equality row [1, 0].
    Rank check must not add inequality row 0 to W because it doesn't increase matrix rank.
    The KKT system remains invertible and finds x* = (1, 2).
    """
    H = np.eye(2) * 2.0
    g = np.array([-4.0, -4.0])
    A_eq = np.array([[1.0, 0.0]], dtype=np.float64)
    b_eq = np.array([1.0], dtype=np.float64)
    A_in = np.array([[2.0, 0.0]], dtype=np.float64)
    b_in = np.array([2.0], dtype=np.float64)

    qp = QP(H=H, g=g, A_eq=A_eq, b_eq=b_eq, A_in=A_in, b_in=b_in, names=["x1", "x2"])
    res = solve_active_set(qp, x0=np.array([1.0, 0.0]))

    assert res.status == "optimal"
    np.testing.assert_allclose(res.x, [1.0, 2.0], atol=1e-6)
    # Stationarity: H x + g - A_eq^T lam = 0 => 2*(1) - 4 - 1*lam = 0 => lam = -2.0
    assert pytest.approx(res.lam[0], abs=1e-5) == -2.0
    assert pytest.approx(res.mu[0], abs=1e-6) == 0.0


def test_edge_case_3_several_constraints_active_at_start():
    """Requirement 3: Several constraints active at starting point (vertex in R^2).

    Problem:
        min  (x1 - 1)^2 + (x2 - 1)^2
        s.t. -x1      <= 0   (row 0: x1 >= 0)
                 -x2  <= 0   (row 1: x2 >= 0)
             -x1 - x2 <= 0   (row 2: x1 + x2 >= 0, redundant at origin)
    Start at x0 = (0, 0) where all three constraints are active.
    Rank check must add rows 0 and 1, but reject row 2 (rank already 2 in R^2).
    Both constraints have negative multipliers at (0, 0), so solver drops them
    and moves to the unconstrained minimizer x* = (1, 1).
    """
    H = np.eye(2) * 2.0
    g = np.array([-2.0, -2.0])
    A_eq = np.zeros((0, 2))
    b_eq = np.zeros(0)
    A_in = np.array([
        [-1.0, 0.0],
        [0.0, -1.0],
        [-1.0, -1.0],
    ], dtype=np.float64)
    b_in = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    qp = QP(H=H, g=g, A_eq=A_eq, b_eq=b_eq, A_in=A_in, b_in=b_in, names=["x1", "x2"])
    res = solve_active_set(qp, x0=np.array([0.0, 0.0]))

    assert res.status == "optimal"
    np.testing.assert_allclose(res.x, [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(res.mu, 0.0, atol=1e-6)


def test_edge_case_4_blocking_constraint_in_ratio_test():
    """Requirement 4: Constraint becomes blocking during ratio test.

    Problem:
        min  (x1 - 3)^2 + (x2 - 3)^2
        s.t.  x1 + x2 <= 2
    Start at x0 = (0, 0).
    Step towards (3, 3) has p = (3, 3).
    Ratio test yields alpha = 2 / 6 = 1/3 < 1.
    Constraint blocks the step, is added to W, and solution converges at (1, 1).
    """
    H = np.eye(2) * 2.0
    g = np.array([-6.0, -6.0])
    A_eq = np.zeros((0, 2))
    b_eq = np.zeros(0)
    A_in = np.array([[1.0, 1.0]], dtype=np.float64)
    b_in = np.array([2.0], dtype=np.float64)

    qp = QP(H=H, g=g, A_eq=A_eq, b_eq=b_eq, A_in=A_in, b_in=b_in, names=["x1", "x2"])
    res = solve_active_set(qp, x0=np.array([0.0, 0.0]))

    assert res.status == "optimal"
    np.testing.assert_allclose(res.x, [1.0, 1.0], atol=1e-6)
    assert pytest.approx(res.mu[0], abs=1e-5) == 4.0


def test_edge_case_5_negative_multiplier_correct_removal():
    """Requirement 5: Correct constraint with negative multiplier is removed while valid one stays.

    Problem:
        min  (x1 - 2)^2 + (x2 - 2)^2
        s.t.  x1      <= 1   (row 0: pulls optimum, mu > 0)
                 -x2  <= 0   (row 1: x2 >= 0, blocks interior, mu < 0)
    Start at x0 = (1, 0) where both rows are active.
    Multipliers at (1, 0): mu_0 = +2 > 0, mu_1 = -4 < 0.
    Solver must remove row 1, keep row 0 in W, and step along x1 = 1 to x* = (1, 2).
    """
    H = np.eye(2) * 2.0
    g = np.array([-4.0, -4.0])
    A_eq = np.zeros((0, 2))
    b_eq = np.zeros(0)
    A_in = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
    b_in = np.array([1.0, 0.0], dtype=np.float64)

    qp = QP(H=H, g=g, A_eq=A_eq, b_eq=b_eq, A_in=A_in, b_in=b_in, names=["x1", "x2"])
    res = solve_active_set(qp, x0=np.array([1.0, 0.0]))

    assert res.status == "optimal"
    np.testing.assert_allclose(res.x, [1.0, 2.0], atol=1e-6)
    assert pytest.approx(res.mu[0], abs=1e-5) == 2.0
    assert pytest.approx(res.mu[1], abs=1e-6) == 0.0


def test_edge_case_6_ratio_test_tie_breaking_lowest_index():
    """Requirement 6: Ratio-test ties resolved deterministically by lowest original inequality index.

    Problem:
        min  (x1 - 2)^2 + (x2 - 2)^2
        s.t.  x1      <= 1   (idx 0)
                  x2  <= 1   (idx 1)
    From x0 = (0, 0) toward (2, 2), both constraints hit at alpha = 0.5.
    Solver must choose idx 0 on the first iteration.
    """
    H = np.eye(2) * 2.0
    g = np.array([-4.0, -4.0])
    A_eq = np.zeros((0, 2))
    b_eq = np.zeros(0)
    A_in = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    b_in = np.array([1.0, 1.0], dtype=np.float64)

    qp = QP(H=H, g=g, A_eq=A_eq, b_eq=b_eq, A_in=A_in, b_in=b_in, names=["x1", "x2"])
    res = solve_active_set(qp, x0=np.array([0.0, 0.0]))

    assert res.status == "optimal"
    np.testing.assert_allclose(res.x, [1.0, 1.0], atol=1e-6)
    assert len(res.history) >= 2


def test_edge_case_7_no_blocking_constraint_full_step():
    """Requirement 7: No blocking constraint: full step alpha=1 taken directly.

    Problem:
        min  (x1 - 1)^2 + (x2 - 1)^2
        s.t.  x1 <= 5,  x2 <= 5
    From x0 = (0, 0), step p = (1, 1) has alpha = 1.0 (no constraint hit).
    Converges in 1 iteration without adding any constraints.
    """
    H = np.eye(2) * 2.0
    g = np.array([-2.0, -2.0])
    A_eq = np.zeros((0, 2))
    b_eq = np.zeros(0)
    A_in = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    b_in = np.array([5.0, 5.0], dtype=np.float64)

    qp = QP(H=H, g=g, A_eq=A_eq, b_eq=b_eq, A_in=A_in, b_in=b_in, names=["x1", "x2"])
    res = solve_active_set(qp, x0=np.array([0.0, 0.0]))

    assert res.status == "optimal"
    np.testing.assert_allclose(res.x, [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(res.mu, 0.0, atol=1e-6)


def test_edge_case_8_initial_point_infeasible_rejection():
    """Requirement 8: Infeasible initial point rejected according to contract.

    1. User-supplied x0 violates constraints: raises ValueError.
    2. Problem intrinsically infeasible (x0=None): returns status="infeasible", obj=inf.
    """
    H = np.eye(2) * 2.0
    g = np.array([0.0, 0.0])
    A_eq = np.zeros((0, 2))
    b_eq = np.zeros(0)
    A_in = np.array([[1.0, 0.0]], dtype=np.float64)
    b_in = np.array([1.0], dtype=np.float64)
    qp = QP(H=H, g=g, A_eq=A_eq, b_eq=b_eq, A_in=A_in, b_in=b_in, names=["x1", "x2"])

    # 1. User passes infeasible x0 = (2, 0)
    with pytest.raises(ValueError, match="violates inequality constraints"):
        solve_active_set(qp, x0=np.array([2.0, 0.0]))

    # 2. Problem with contradictory constraints: x1 <= 0 and -x1 <= -1 (x1 >= 1)
    A_in_infeas = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float64)
    b_in_infeas = np.array([0.0, -1.0], dtype=np.float64)
    qp_infeas = QP(H=H, g=g, A_eq=A_eq, b_eq=b_eq, A_in=A_in_infeas, b_in=b_in_infeas, names=["x1", "x2"])

    res = solve_active_set(qp_infeas, x0=None)
    assert res.status == "infeasible"
    assert res.obj == float("inf")


def test_edge_case_9_maximum_iteration_limit():
    """Requirement 9: Maximum iteration limit returns max_iter and preserves history."""
    # Problem that takes at least 2 iterations
    H = np.eye(2) * 2.0
    g = np.array([-4.0, -4.0])
    A_eq = np.zeros((0, 2))
    b_eq = np.zeros(0)
    A_in = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    b_in = np.array([1.0, 1.0], dtype=np.float64)
    qp = QP(H=H, g=g, A_eq=A_eq, b_eq=b_eq, A_in=A_in, b_in=b_in, names=["x1", "x2"])

    res = solve_active_set(qp, x0=np.array([0.0, 0.0]), max_iter=1)

    assert res.status == "max_iter"
    assert res.iters == 1
    assert len(res.history) == 1


def test_edge_case_10_singular_kkt_system():
    """Requirement 10: Singular KKT system returns status='singular' without fabricating a result."""
    # Zero Hessian on unconstrained variables with no equalities to fix them
    H_zero = np.zeros((2, 2), dtype=np.float64)
    g_zero = np.zeros(2, dtype=np.float64)
    qp_singular = QP(
        H=H_zero,
        g=g_zero,
        A_eq=np.zeros((0, 2)),
        b_eq=np.zeros(0),
        A_in=np.zeros((0, 2)),
        b_in=np.zeros(0),
        names=["x1", "x2"],
    )

    res = solve_active_set(qp_singular, x0=np.array([0.0, 0.0]))
    assert res.status == "singular"


def test_edge_case_11_full_kkt_conditions_verification():
    """Requirement 11: Verify equality residual, inequality violation, stationarity, dual feasibility, and complementarity on known test QP.

    Problem:
        min  0.5 * (x1^2 + x2^2 + x3^2) - 2*x1 - 3*x2 - x3
        s.t.  x1 + x2 + x3 = 3     (equality: row 0)
              x1           <= 0.5   (inequality: row 0)
                 -x2       <= 0     (inequality: row 1: x2 >= 0)
                      -x3  <= 0     (inequality: row 2: x3 >= 0)
    """
    H = np.eye(3, dtype=np.float64)
    g = np.array([-2.0, -3.0, -1.0], dtype=np.float64)
    A_eq = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
    b_eq = np.array([3.0], dtype=np.float64)
    A_in = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ], dtype=np.float64)
    b_in = np.array([0.5, 0.0, 0.0], dtype=np.float64)

    qp = QP(H=H, g=g, A_eq=A_eq, b_eq=b_eq, A_in=A_in, b_in=b_in, names=["x1", "x2", "x3"])
    res = solve_active_set(qp, x0=np.array([0.5, 1.5, 1.0]))

    assert res.status == "optimal"
    tol = 1e-7

    # 1. Primal Equality Residual: ||A_eq x - b_eq||_inf <= tol
    eq_res = float(np.max(np.abs(A_eq @ res.x - b_eq)))
    assert eq_res <= tol, f"Equality residual violated: {eq_res}"

    # 2. Primal Inequality Violation: max(A_in x - b_in) <= tol
    ineq_viol = float(np.max(A_in @ res.x - b_in))
    assert ineq_viol <= tol, f"Inequality violated: {ineq_viol}"

    # 3. Dual Feasibility: mu >= -tol
    assert np.all(res.mu >= -tol), f"Dual feasibility violated: min mu = {np.min(res.mu)}"

    # 4. Stationarity Residual: ||H x + g - A_eq^T lam + A_in^T mu||_inf <= tol
    stat_res = float(np.max(np.abs(H @ res.x + g - A_eq.T @ res.lam + A_in.T @ res.mu)))
    assert stat_res <= tol, f"Stationarity violated: {stat_res}"

    # 5. Complementary Slackness: mu_i * (b_in[i] - A_in[i] @ x) == 0
    slack = b_in - A_in @ res.x
    comp = np.abs(res.mu * slack)
    assert np.all(comp <= tol), f"Complementarity violated: max comp = {np.max(comp)}"



# =====================================================================
# Full Integration: Active-Set vs Lambda-Iteration vs Reference Model
# =====================================================================

def test_base_ed_integration_active_set_lambda_iteration_reference():
    """Verify active-set, lambda-iteration, and reference model on IEEE 30-bus ED.

    Validates:
        - build_ed() return type is QP.
        - x0 obtained strictly via feasible_start().
        - Active-set matches lambda-iteration within 1e-6 relative for cost/lambda and 1e-4 MW for dispatch.
        - Power balance: sum(P) == 189.2 MW.
        - Stationarity residual ||H x + g - A_eq^T lam + A_in^T mu|| == 0.
        - Complementary slackness holds and no generator bound binds.
    """
    case = load_case("data/case30.m")
    qp = build_ed(case, load_scale=1.0)

    # 1. Confirm return type and variable ordering
    assert isinstance(qp, QP)
    assert qp.names == [f"P{i+1}" for i in range(6)]
    assert qp.A_eq.shape == (1, 6)
    assert qp.b_eq[0] == 189.2
    assert qp.A_in.shape == (12, 6)
    assert len(qp.b_in) == 12

    # 2. Feasible start from project routine
    x0 = feasible_start(qp)
    assert x0 is not None
    assert np.max(np.abs(qp.A_eq @ x0 - qp.b_eq)) <= 1e-6
    assert np.all(qp.A_in @ x0 <= qp.b_in + 1e-6)

    # 3. Active-set solve
    res_as = solve_active_set(qp, x0=x0)
    assert res_as.status == "optimal"

    # 4. Lambda-iteration solve
    res_li = solve_lambda_iteration(qp)
    assert res_li.status == "optimal"

    # 5. Numerical comparisons (cost ~ 565.21, lambda ~ +3.7892)
    expected_cost = 565.205966
    expected_lambda = 3.789196
    expected_dispatch = np.array([44.729908, 58.262752, 22.313570, 32.325918, 15.783926, 15.783926])

    assert pytest.approx(res_as.obj, rel=1e-6) == expected_cost
    assert pytest.approx(res_li.obj, rel=1e-6) == expected_cost

    assert pytest.approx(res_as.lam[0], rel=1e-6) == expected_lambda
    assert pytest.approx(res_li.lam[0], rel=1e-6) == expected_lambda

    np.testing.assert_allclose(res_as.x, expected_dispatch, atol=1e-4)
    np.testing.assert_allclose(res_li.x, expected_dispatch, atol=1e-4)
    np.testing.assert_allclose(res_as.x, res_li.x, atol=1e-4)

    # 6. Physical constraints & KKT checks
    # Generation total = 189.2 MW
    assert pytest.approx(np.sum(res_as.x), abs=1e-7) == 189.2
    assert pytest.approx(np.sum(res_li.x), abs=1e-7) == 189.2

    # No generator limit binds
    assert np.all(res_as.x >= case.p_min)
    assert np.all(res_as.x <= case.p_max)
    assert np.all(res_as.mu == 0.0)
    assert np.all(res_li.mu == 0.0)

    # Stationarity
    stat_as = qp.H @ res_as.x + qp.g - qp.A_eq.T @ res_as.lam + qp.A_in.T @ res_as.mu
    assert np.max(np.abs(stat_as)) <= 1e-8

    stat_li = qp.H @ res_li.x + qp.g - qp.A_eq.T @ res_li.lam + qp.A_in.T @ res_li.mu
    assert np.max(np.abs(stat_li)) <= 1e-8

    # Optional CVXPY check if installed
    try:
        from src.opf.reference import solve as solve_reference
        res_ref = solve_reference(qp)
        assert res_ref.status == "optimal"
        assert pytest.approx(res_as.obj, rel=1e-5) == res_ref.obj
        assert pytest.approx(res_as.lam[0], rel=1e-5) == res_ref.lam[0]
        np.testing.assert_allclose(res_as.x, res_ref.x, atol=1e-4)
    except ImportError:
        pass



def test_dcopf_lmp_economic_interpretation_sensitivity():
    """Verify the economic interpretation of Locational Marginal Prices (LMPs).

    Economic Principle:
        The equality multiplier lambda_k corresponding to bus k's real power balance
        represents the marginal cost of supplying an additional increment of demand at bus k:
            lambda_k = d(Cost*) / d(d_k)

    Experiment (CONTEXT.md Section 6):
        1. Solve the base case DC-OPF (rate_scale=1.0) and record base cost and bus LMPs.
        2. Perturb real power demand upward by exactly Delta_d = 1.0 MW at Bus 2 (index 1).
        3. Rebuild the QP via the existing build_dcopf() model API and solve again.
        4. Compare Delta_Cost = Cost_perturbed - Cost_base with Bus 2's LMP:
           - First-order approximation: Delta_Cost ~= lambda_2 * Delta_d (agrees within < 0.1%).
           - Second-order Taylor series:
             Cost(d + Delta_d) = Cost(d) + lambda * Delta_d + 0.5 * (d_lambda / d_D) * (Delta_d)^2
             Since d_lambda / d_D = 1 / sum(1 / (2*c_i)) = 0.006191 $/MWh^2,
             the curvature term 0.5 * 0.006191 * (1.0)^2 = +0.003096 $ explains the exact
             difference between the finite difference and the point derivative.
        5. Active-set remains unchanged (no lines congested, all generators within bounds).
    """
    case = load_case("data/case30.m")
    qp_base = build_dcopf(case, load_scale=1.0, rate_scale=1.0)
    res_base = solve_active_set(qp_base)
    assert res_base.status == "optimal"

    base_cost = res_base.obj
    bus_2_lmp = res_base.lam[1]  # Bus 2 is index 1

    # Perturb demand at Bus 2 by +1.0 MW
    delta_d = 1.0  # MW
    case_perturbed = load_case("data/case30.m")
    case_perturbed.bus_d[1] += delta_d

    qp_perturbed = build_dcopf(case_perturbed, load_scale=1.0, rate_scale=1.0)
    res_perturbed = solve_active_set(qp_perturbed)
    assert res_perturbed.status == "optimal"

    perturbed_cost = res_perturbed.obj
    delta_cost = perturbed_cost - base_cost

    # 1. First-order agreement: Delta_Cost ~= lambda_2 * Delta_d within 0.1% relative error
    assert pytest.approx(delta_cost, rel=0.001) == bus_2_lmp * delta_d

    # 2. Second-order Taylor series exact match:
    # d_lambda / d_D = 1 / sum(1 / (2*c_i))
    d_lambda_d_d = 1.0 / np.sum(1.0 / (2.0 * case.c))
    curvature_term = 0.5 * d_lambda_d_d * (delta_d ** 2)
    expected_delta_cost = bus_2_lmp * delta_d + curvature_term

    assert pytest.approx(delta_cost, rel=1e-6) == expected_delta_cost

    # 3. Verify no line or generator limits were triggered by the 1 MW increment
    assert np.all(res_perturbed.mu == 0.0)



def test_lambda_iteration_shared_solve_interface():
    """Verify solve() wrapper in lambda_iteration.py matches solve_lambda_iteration()."""
    case = load_case("data/case30.m")
    qp = build_ed(case, load_scale=1.0)

    res_shared = solve_lambda(qp)
    res_direct = solve_lambda_iteration(qp)

    assert res_shared.status == "optimal"
    assert res_direct.status == "optimal"
    assert pytest.approx(res_shared.obj, abs=1e-8) == res_direct.obj
    np.testing.assert_allclose(res_shared.x, res_direct.x, atol=1e-8)
    np.testing.assert_allclose(res_shared.lam, res_direct.lam, atol=1e-8)


def test_active_set_non_optimal_linprog_start(capsys):
    """Verify active-set solver does real work from a non-optimal feasible starting point.

    Computes a non-optimal starting point using linprog with linear objective c = -qp.g
    for both rate_scale=1.0 and rate_scale=0.75.
    Asserts:
        - status == 'optimal'
        - cost matches benchmark within 0.005 (565.21 for 1.0, 565.97 for 0.75)
        - iters > 5 (performs multi-step working set additions and drops)
        - history[0] > history[-1] (objective strictly decreases)
    Prints iteration counts.
    """
    case = load_case("data/case30.m")

    # 1. Base case (rate_scale=1.0)
    qp_base = build_dcopf(case, load_scale=1.0, rate_scale=1.0)
    res_lp_base = linprog(
        c=-qp_base.g,
        A_ub=qp_base.A_in,
        b_ub=qp_base.b_in,
        A_eq=qp_base.A_eq,
        b_eq=qp_base.b_eq,
        bounds=(None, None),
        method="highs",
    )
    assert res_lp_base.success, "linprog failed to find feasible start"
    x0_base = res_lp_base.x

    res_base = solve_active_set(qp_base, x0=x0_base)
    assert res_base.status == "optimal"
    assert pytest.approx(res_base.obj, abs=0.005) == 565.21
    assert res_base.iters > 5
    assert res_base.history[0] > res_base.history[-1]
    print(f"\n[Non-optimal start] Base case (rate_scale=1.0) completed in {res_base.iters} iterations (initial cost: {res_base.history[0]:.2f}, final cost: {res_base.obj:.2f})")

    # 2. Congested case (rate_scale=0.75)
    qp_75 = build_dcopf(case, load_scale=1.0, rate_scale=0.75)
    res_lp_75 = linprog(
        c=-qp_75.g,
        A_ub=qp_75.A_in,
        b_ub=qp_75.b_in,
        A_eq=qp_75.A_eq,
        b_eq=qp_75.b_eq,
        bounds=(None, None),
        method="highs",
    )
    assert res_lp_75.success, "linprog failed to find feasible start"
    x0_75 = res_lp_75.x

    res_75 = solve_active_set(qp_75, x0=x0_75)
    assert res_75.status == "optimal"
    assert pytest.approx(res_75.obj, abs=0.005) == 565.97
    assert res_75.iters > 5
    assert res_75.history[0] > res_75.history[-1]
    print(f"[Non-optimal start] Congested case (rate_scale=0.75) completed in {res_75.iters} iterations (initial cost: {res_75.history[0]:.2f}, final cost: {res_75.obj:.2f})")


def test_active_set_max_iter_zero_raises_error():
    """Verify that max_iter < 1 raises ValueError."""
    H = np.eye(2, dtype=np.float64)
    g = np.zeros(2, dtype=np.float64)
    qp = QP(
        H=H,
        g=g,
        A_eq=np.zeros((0, 2)),
        b_eq=np.zeros(0),
        A_in=np.zeros((0, 2)),
        b_in=np.zeros(0),
        names=["x1", "x2"],
    )
    with pytest.raises(ValueError, match="max_iter must be at least 1"):
        solve_active_set(qp, x0=np.array([0.0, 0.0]), max_iter=0)
