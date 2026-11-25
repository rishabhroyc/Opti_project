"""Unit tests for src/opf/models.py.

Author: Rishabh Roy Choudhury (Albert and Einstein)
"""

import numpy as np
import pytest
from src.opf.check import (
    comparison_table,
    kkt_residual,
    kkt_residuals,
    marginal_cost_check,
    marginal_costs,
    print_comparison_table,
)
from src.opf.models import Case, build_dcopf, build_ed, feasible_start, load_case
from src.opf.qp import QPResult
from src.opf.reference import solve as solve_reference


def test_load_case_30():
    """Verify MATPOWER case30.m parsing against specification."""
    case = load_case("data/case30.m")

    # Dimensions
    assert len(case.bus_d) == 30, f"Expected 30 buses, got {len(case.bus_d)}"
    assert len(case.gen_bus) == 6, f"Expected 6 generators, got {len(case.gen_bus)}"
    assert len(case.branch_x) == 41, f"Expected 41 lines, got {len(case.branch_x)}"

    # System parameters
    assert case.base_mva == 100.0, f"Expected base MVA 100, got {case.base_mva}"
    assert case.ref_bus == 1, f"Expected reference bus 1, got {case.ref_bus}"
    assert pytest.approx(case.bus_d.sum(), rel=1e-5) == 189.2, (
        f"Expected total demand 189.2 MW, got {case.bus_d.sum()}"
    )

    # Line ratings: all between 16 and 130, none zero
    assert np.all(case.f_max >= 16.0), f"Found rating below 16: {case.f_max.min()}"
    assert np.all(case.f_max <= 130.0), f"Found rating above 130: {case.f_max.max()}"
    assert not np.any(case.f_max == 0), "Found zero rating"

    # Generator limits and costs sanity check
    assert len(case.p_min) == 6
    assert len(case.p_max) == 6
    assert len(case.b) == 6
    assert len(case.c) == 6
    assert np.all(case.p_min >= 0)
    assert np.all(case.p_max > case.p_min)
    assert np.all(case.c > 0)


def test_build_ed_kkt():
    """Verify build_ed formulation shapes and unconstrained KKT solution."""
    case = load_case("data/case30.m")
    qp = build_ed(case, load_scale=1.0)

    # Check shapes: 6 vars, 1 eq, 12 ineq
    assert len(qp.names) == 6
    assert qp.H.shape == (6, 6)
    assert qp.g.shape == (6,)
    assert qp.A_eq.shape == (1, 6)
    assert qp.b_eq.shape == (1,)
    assert qp.A_in.shape == (12, 6)
    assert qp.b_in.shape == (12,)

    # Solve equality-only KKT system:
    # [ H      -A_eq^T ] [  x  ] = [ -g   ]
    # [ A_eq      0    ] [ lam ]   [ b_eq ]
    kkt_mat = np.block([
        [qp.H, -qp.A_eq.T],
        [qp.A_eq, np.zeros((1, 1), dtype=np.float64)],
    ])
    rhs = np.concatenate([-qp.g, qp.b_eq])
    sol = np.linalg.solve(kkt_mat, rhs)
    x = sol[:6]
    lam = sol[6]

    # Reference dispatch: 44.73, 58.26, 22.31, 32.33, 15.78, 15.78 MW (tol 0.005)
    expected_dispatch = np.array([44.73, 58.26, 22.31, 32.33, 15.78, 15.78])
    np.testing.assert_allclose(x, expected_dispatch, atol=0.005)

    # Verify no limit violated
    assert np.all(x >= case.p_min), "Lower generation limit violated"
    assert np.all(x <= case.p_max), "Upper generation limit violated"

    # Verify b_i + 2 c_i P_i equal for all six generators (~3.7892)
    incremental_costs = case.b + 2.0 * case.c * x
    assert pytest.approx(lam, abs=1e-4) == 3.7892
    for ic in incremental_costs:
        assert pytest.approx(ic, abs=1e-4) == 3.7892


def test_build_dcopf():
    """Verify build_dcopf formulation, network matrices, and base case properties."""
    case = load_case("data/case30.m")
    qp = build_dcopf(case, load_scale=1.0, rate_scale=1.0)

    # 1. Shapes: 36 vars, 31 eq, 94 ineq
    assert len(qp.names) == 36
    assert qp.names[:6] == [f"P{i + 1}" for i in range(6)]
    assert qp.names[6:] == [f"theta{k + 1}" for k in range(30)]
    assert qp.H.shape == (36, 36)
    assert qp.g.shape == (36,)
    assert qp.A_eq.shape == (31, 36)
    assert qp.b_eq.shape == (31,)
    assert qp.A_in.shape == (94, 36)
    assert qp.b_in.shape == (94,)

    # 2. Equality rows rank = 31
    rank_aeq = np.linalg.matrix_rank(qp.A_eq)
    assert rank_aeq == 31, f"Expected rank 31 for A_eq, got {rank_aeq}"

    # 3. B_bus extraction and properties
    # In A_bal = [C_g, -S_base * B_bus], extract B_bus
    S_base = case.base_mva  # 100.0
    B_bus = -qp.A_eq[:30, 6:] / S_base

    # B_bus is symmetric
    np.testing.assert_allclose(B_bus, B_bus.T, atol=1e-12)

    # B_bus rows sum to zero
    np.testing.assert_allclose(B_bus.sum(axis=1), np.zeros(30), atol=1e-12)

    # B_bus has rank 29 (connected network with 30 buses)
    rank_bbus = np.linalg.matrix_rank(B_bus)
    assert rank_bbus == 29, f"Expected rank 29 for B_bus, got {rank_bbus}"

    # 4. H is PSD with zeros on theta
    eigenvalues = np.linalg.eigvalsh(qp.H)
    assert np.all(eigenvalues >= -1e-14), "H has negative eigenvalues"
    np.testing.assert_allclose(qp.H[6:, :], 0.0)
    np.testing.assert_allclose(qp.H[:, 6:], 0.0)
    np.testing.assert_allclose(qp.g[6:], 0.0)

    # 5. Take ED dispatch, solve angles from balance equations (reduced system)
    qp_ed = build_ed(case, load_scale=1.0)
    kkt_ed = np.block([
        [qp_ed.H, -qp_ed.A_eq.T],
        [qp_ed.A_eq, np.zeros((1, 1))],
    ])
    rhs_ed = np.concatenate([-qp_ed.g, qp_ed.b_eq])
    p_ed = np.linalg.solve(kkt_ed, rhs_ed)[:6]

    # Generator injection at each bus
    C_g = qp.A_eq[:30, :6]
    p_inj = C_g @ p_ed - case.bus_d  # Net bus real power injection

    # Reference bus angle is fixed to 0
    ref_idx = case.ref_bus - 1
    non_ref = [k for k in range(30) if k != ref_idx]

    # Solve S_base * B_bus_red * theta_red = P_inj_red
    B_red = S_base * B_bus[np.ix_(non_ref, non_ref)]
    theta = np.zeros(30, dtype=np.float64)
    theta[non_ref] = np.linalg.solve(B_red, p_inj[non_ref])

    # Show balance residual is ~0
    balance_residual = np.max(np.abs(C_g @ p_ed - S_base * B_bus @ theta - case.bus_d))
    assert balance_residual < 1e-10, f"Balance residual too large: {balance_residual}"

    # Full variable vector x = [P, theta]
    x_ed = np.concatenate([p_ed, theta])

    # Flow calculation: flows are rows 0..41 in A_in
    flows = qp.A_in[:41, :] @ x_ed
    ratings = qp.b_in[:41]
    loading = np.abs(flows) / ratings

    # Show all flows within ratings
    assert np.all(loading <= 1.0 + 1e-8), "Line flow exceeds rating in base case"

    # Show busiest line is 6-8 at 76.4% of 32 MVA (tol 0.0005)
    busiest_idx = int(np.argmax(loading))
    busiest_from = case.branch_from[busiest_idx]
    busiest_to = case.branch_to[busiest_idx]
    assert (busiest_from, busiest_to) == (6, 8) or (busiest_from, busiest_to) == (8, 6), (
        f"Expected busiest line 6-8, got {busiest_from}-{busiest_to}"
    )
    assert pytest.approx(ratings[busiest_idx], abs=1e-5) == 32.0
    assert pytest.approx(loading[busiest_idx], abs=0.0005) == 0.764

    # 6. Scaling test: load_scale and rate_scale scale only d and F_max
    qp_scaled = build_dcopf(case, load_scale=1.2, rate_scale=0.75)
    np.testing.assert_allclose(qp_scaled.H, qp.H)
    np.testing.assert_allclose(qp_scaled.g, qp.g)
    np.testing.assert_allclose(qp_scaled.A_eq, qp.A_eq)
    np.testing.assert_allclose(qp_scaled.A_in, qp.A_in)
    np.testing.assert_allclose(qp_scaled.b_eq[:30], qp.b_eq[:30] * 1.2)
    assert qp_scaled.b_eq[30] == 0.0
    np.testing.assert_allclose(qp_scaled.b_in[:41], qp.b_in[:41] * 0.75)
    np.testing.assert_allclose(qp_scaled.b_in[41:82], qp.b_in[41:82] * 0.75)
    np.testing.assert_allclose(qp_scaled.b_in[82:], qp.b_in[82:])  # Gen limits unchanged


def test_feasible_start():
    """Verify feasible_start finds valid points for feasible cases and reports None for infeasible."""
    case = load_case("data/case30.m")

    # 1. Base case DC-OPF (rate_scale=1.0): feasible start satisfies all constraints
    qp_base = build_dcopf(case, load_scale=1.0, rate_scale=1.0)
    x0_base = feasible_start(qp_base)
    assert x0_base is not None, "Expected feasible start for base case DC-OPF"
    assert x0_base.shape == (36,)
    assert np.max(np.abs(qp_base.A_eq @ x0_base - qp_base.b_eq)) <= 1e-6
    assert np.all(qp_base.A_in @ x0_base <= qp_base.b_in + 1e-6)

    # 2. Congested DC-OPF (rate_scale=0.75): linprog finds feasible start
    qp_75 = build_dcopf(case, load_scale=1.0, rate_scale=0.75)
    x0_75 = feasible_start(qp_75)
    assert x0_75 is not None, "Expected feasible start for rate_scale=0.75 DC-OPF"
    assert x0_75.shape == (36,)
    assert np.max(np.abs(qp_75.A_eq @ x0_75 - qp_75.b_eq)) <= 1e-6
    assert np.all(qp_75.A_in @ x0_75 <= qp_75.b_in + 1e-6)

    # 3. Infeasible DC-OPF (rate_scale=0.70): reports None
    qp_70 = build_dcopf(case, load_scale=1.0, rate_scale=0.70)
    x0_70 = feasible_start(qp_70)
    assert x0_70 is None, f"Expected None for infeasible case (rate_scale=0.70), got {x0_70}"

    # 4. Pure ED case
    qp_ed = build_ed(case, load_scale=1.0)
    x0_ed = feasible_start(qp_ed)
    assert x0_ed is not None, "Expected feasible start for ED"
    assert x0_ed.shape == (6,)
    assert np.max(np.abs(qp_ed.A_eq @ x0_ed - qp_ed.b_eq)) <= 1e-6
    assert np.all(qp_ed.A_in @ x0_ed <= qp_ed.b_in + 1e-6)


def test_reference_solver():
    """Verify CVXPY + Clarabel reference solver against verified benchmark results.

    Benchmark tolerances: half a unit of the last printed digit.
    """
    case = load_case("data/case30.m")

    # 1. ED base case: cost 565.21, lambda 3.7892, mu >= 0
    qp_ed = build_ed(case, load_scale=1.0)
    res_ed = solve_reference(qp_ed)
    assert res_ed.status == "optimal"
    assert pytest.approx(res_ed.obj, abs=0.005) == 565.21
    assert len(res_ed.lam) == 1
    assert pytest.approx(res_ed.lam[0], abs=0.00005) == 3.7892
    assert np.all(res_ed.mu >= 0.0)

    # 2. DC-OPF base case: cost 565.21, all 30 balance multipliers 3.7892, mu >= 0
    qp_base = build_dcopf(case, load_scale=1.0, rate_scale=1.0)
    res_base = solve_reference(qp_base)
    assert res_base.status == "optimal"
    assert pytest.approx(res_base.obj, abs=0.005) == 565.21
    # 30 bus balance multipliers (first 30 elements of lam)
    bus_lmps_base = res_base.lam[:30]
    for lmp in bus_lmps_base:
        assert pytest.approx(lmp, abs=0.00005) == 3.7892
    assert np.all(res_base.mu >= 0.0)

    # 3. DC-OPF ratings x0.75: cost 565.97, LMPs 3.65 to 7.18, mu >= 0
    qp_75 = build_dcopf(case, load_scale=1.0, rate_scale=0.75)
    res_75 = solve_reference(qp_75)
    assert res_75.status == "optimal"
    assert pytest.approx(res_75.obj, abs=0.005) == 565.97
    bus_lmps_75 = res_75.lam[:30]
    assert pytest.approx(bus_lmps_75.min(), abs=0.005) == 3.65
    assert pytest.approx(bus_lmps_75.max(), abs=0.005) == 7.18
    assert np.all(res_75.mu >= 0.0)

    # 4. DC-OPF ratings x0.72: cost 589.58, LMPs 3.02 to 39.65, mu >= 0
    qp_72 = build_dcopf(case, load_scale=1.0, rate_scale=0.72)
    res_72 = solve_reference(qp_72)
    assert res_72.status == "optimal"
    assert pytest.approx(res_72.obj, abs=0.005) == 589.58
    bus_lmps_72 = res_72.lam[:30]
    assert pytest.approx(bus_lmps_72.min(), abs=0.005) == 3.02
    assert pytest.approx(bus_lmps_72.max(), abs=0.005) == 39.65
    assert np.all(res_72.mu >= 0.0)

    # 5. DC-OPF ratings x0.71 and x0.70: status == 'infeasible'
    qp_71 = build_dcopf(case, load_scale=1.0, rate_scale=0.71)
    res_71 = solve_reference(qp_71)
    assert res_71.status == "infeasible"

    qp_70 = build_dcopf(case, load_scale=1.0, rate_scale=0.70)
    res_70 = solve_reference(qp_70)
    assert res_70.status == "infeasible"


def test_kkt_residuals_and_perturbation():
    """Verify KKT residual accuracy on reference solutions and sensitivity to perturbation."""
    case = load_case("data/case30.m")

    # 1. Reference KKT residual < 1e-6 on base case
    qp_base = build_dcopf(case, load_scale=1.0, rate_scale=1.0)
    res_base = solve_reference(qp_base)
    kkt_base = kkt_residuals(qp_base, res_base)
    assert kkt_base.stationarity < 1e-6
    assert kkt_base.primal < 1e-6
    assert kkt_base.dual < 1e-6
    assert kkt_base.complementarity < 1e-6
    assert kkt_residual(qp_base, res_base) < 1e-6

    # 2. Reference KKT residual < 1e-6 on ratings x0.75
    qp_75 = build_dcopf(case, load_scale=1.0, rate_scale=0.75)
    res_75 = solve_reference(qp_75)
    assert kkt_residual(qp_75, res_75) < 1e-6

    # 3. Perturbed solution has a large residual
    x_perturbed = res_base.x.copy()
    x_perturbed[0] += 5.0  # Perturb generator 1 by 5 MW
    res_perturbed = QPResult(
        x=x_perturbed,
        obj=res_base.obj,
        lam=res_base.lam,
        mu=res_base.mu,
        status="optimal",
        iters=res_base.iters,
        history=res_base.history,
    )
    res_bad = kkt_residual(qp_base, res_perturbed)
    assert res_bad > 0.05, f"Expected large residual for perturbed solution, got {res_bad}"


def test_comparison_table_and_marginal_costs():
    """Verify comparison table formatting and marginal cost verification."""
    case = load_case("data/case30.m")
    qp_base = build_dcopf(case, load_scale=1.0, rate_scale=1.0)
    res_base = solve_reference(qp_base)

    # 1. Marginal costs about 3.7892 for all six generators
    mc = marginal_costs(case, res_base)
    assert len(mc) == 6
    for val in mc:
        assert pytest.approx(val, abs=1e-4) == 3.7892

    # 2. Marginal cost check (+1 MW demand at bus 1) matches to first order
    mc_check = marginal_cost_check(qp_base, bus_idx=0)
    assert mc_check["delta_cost"] > 0
    assert pytest.approx(mc_check["bus_lambda"], abs=1e-4) == 3.7892
    assert mc_check["gap"] < 0.01  # Second-order gap from quadratic cost is ~0.0031

    # 3. Table columns right
    table_data = comparison_table([("Clarabel", res_base)], ref=res_base, qp=qp_base)
    assert len(table_data) == 1
    row = table_data[0]
    expected_keys = {"name", "cost", "max_dx", "max_dlambda", "kkt_residual", "iters", "status"}
    assert expected_keys.issubset(row.keys())
    assert row["name"] == "Clarabel"
    assert pytest.approx(row["cost"], abs=0.01) == 565.21
    assert row["max_dx"] == 0.0
    assert row["max_dlambda"] == 0.0
    assert row["kkt_residual"] < 1e-6

    # Text printer
    text = print_comparison_table(table_data)
    assert "Solver" in text
    assert "Cost" in text
    assert "Max |dx|" in text
    assert "Max |dlam|" in text
    assert "KKT Res" in text
    assert "Clarabel" in text





