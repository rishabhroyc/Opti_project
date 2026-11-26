"""Unit tests for experiments/common.py helpers.

Author: B. Rithwik (Albert and Einstein)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure src/ is on the path
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from experiments.common import branch_ends, binding_lines, kkt_residual, get_case
from src.opf.qp import QP, QPResult


class TestBranchEnds:
    """Tests for branch_ends()."""

    def test_returns_41_lines(self) -> None:
        case = get_case()
        ends = branch_ends(case)
        assert len(ends) == 41, f"Expected 41 lines, got {len(ends)}"

    def test_contains_6_8(self) -> None:
        case = get_case()
        ends = branch_ends(case)
        assert (6, 8) in ends, "Line (6, 8) not found"

    def test_contains_25_27(self) -> None:
        case = get_case()
        ends = branch_ends(case)
        assert (25, 27) in ends, "Line (25, 27) not found"

    def test_all_tuples_of_ints(self) -> None:
        case = get_case()
        ends = branch_ends(case)
        for f, t in ends:
            assert isinstance(f, int) and isinstance(t, int)


class TestBindingLines:
    """Tests for binding_lines()."""

    def test_no_binding_all_zero(self) -> None:
        mu = np.zeros(94)
        assert binding_lines(mu) == []

    def test_upper_binding(self) -> None:
        """Line 0 upper binding, line 5 lower binding."""
        mu = np.zeros(94)
        mu[0] = 1.0      # line 0 upper
        mu[41 + 5] = 2.0  # line 5 lower
        result = binding_lines(mu)
        assert result == [0, 5]

    def test_both_directions_same_line(self) -> None:
        """Same line binding in both directions → counted once."""
        mu = np.zeros(94)
        mu[3] = 0.5       # line 3 upper
        mu[41 + 3] = 0.5  # line 3 lower
        result = binding_lines(mu)
        assert result == [3]

    def test_tolerance(self) -> None:
        """Values below tolerance are not binding."""
        mu = np.zeros(94)
        mu[10] = 1e-8  # below default tol=1e-6
        assert binding_lines(mu) == []

        mu[10] = 1e-5  # above tol
        assert binding_lines(mu) == [10]


class TestKKTResidual:
    """Tests for kkt_residual() on a small QP."""

    def test_exact_solution_residual_zero(self) -> None:
        """For the QP  min x^2 - 2x  s.t. x <= 5,
        optimal at x=1, lam=[], mu=[0], residual should be 0.
        """
        H = np.array([[2.0]])
        g = np.array([-2.0])
        A_eq = np.zeros((0, 1))
        b_eq = np.zeros(0)
        A_in = np.array([[1.0]])
        b_in = np.array([5.0])

        qp = QP(H=H, g=g, A_eq=A_eq, b_eq=b_eq, A_in=A_in, b_in=b_in,
                 names=["x"])

        result = QPResult(
            x=np.array([1.0]),
            obj=-1.0,
            lam=np.array([], dtype=np.float64),
            mu=np.array([0.0]),
            status="optimal",
            iters=1,
            history=[-1.0],
        )

        residual = kkt_residual(qp, result)
        assert residual < 1e-12, f"Expected ~0, got {residual}"

    def test_active_constraint_residual(self) -> None:
        """For min x^2 - 10x  s.t. x <= 3,
        optimal at x=3, mu=4 (stationarity: 2*3 - 10 + 4 = 0).
        """
        H = np.array([[2.0]])
        g = np.array([-10.0])
        A_eq = np.zeros((0, 1))
        b_eq = np.zeros(0)
        A_in = np.array([[1.0]])
        b_in = np.array([3.0])

        qp = QP(H=H, g=g, A_eq=A_eq, b_eq=b_eq, A_in=A_in, b_in=b_in,
                 names=["x"])

        result = QPResult(
            x=np.array([3.0]),
            obj=3.0 * 3.0 - 10.0 * 3.0,
            lam=np.array([], dtype=np.float64),
            mu=np.array([4.0]),
            status="optimal",
            iters=1,
            history=[-21.0],
        )

        residual = kkt_residual(qp, result)
        assert residual < 1e-12, f"Expected ~0, got {residual}"

    def test_with_equality(self) -> None:
        """For min 0.5*(x1^2 + x2^2) s.t. x1 + x2 = 1,
        optimal at x1=x2=0.5, lam=[0.5].
        Stationarity: [0.5, 0.5] - [1,1]*0.5 = 0.
        """
        H = np.eye(2)
        g = np.zeros(2)
        A_eq = np.array([[1.0, 1.0]])
        b_eq = np.array([1.0])
        A_in = np.zeros((0, 2))
        b_in = np.zeros(0)

        qp = QP(H=H, g=g, A_eq=A_eq, b_eq=b_eq, A_in=A_in, b_in=b_in,
                 names=["x1", "x2"])

        result = QPResult(
            x=np.array([0.5, 0.5]),
            obj=0.25,
            lam=np.array([0.5]),
            mu=np.array([], dtype=np.float64),
            status="optimal",
            iters=1,
            history=[0.25],
        )

        residual = kkt_residual(qp, result)
        assert residual < 1e-12
