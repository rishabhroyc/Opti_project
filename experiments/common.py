"""Shared utilities for experiment scripts.

Provides a single swap-point for the solver, path helpers, and
convenience functions for case loading and result inspection.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Path setup: add src/ to sys.path so ``from opf.…`` imports work.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# ---------------------------------------------------------------------------
# Results directory (created lazily by scripts that write output).
# ---------------------------------------------------------------------------
RESULTS = _REPO_ROOT / "results"

# ---------------------------------------------------------------------------
# Re-exports for convenience (these have no heavy optional deps).
# ---------------------------------------------------------------------------
from opf.models import load_case, build_ed, build_dcopf, Case  # noqa: E402
from opf.qp import QP, QPResult  # noqa: E402

# ---------------------------------------------------------------------------
# Solver swap-point.
# On Friday this becomes:  from opf.active_set import solve
#
# The import is wrapped in a function so that tests which don't call
# ``solve`` can still import the other helpers without cvxpy installed.
# ---------------------------------------------------------------------------
_solve_fn = None


def solve(qp: QP, **kwargs) -> QPResult:
    """Solve a QP using the currently-selected solver."""
    global _solve_fn
    if _solve_fn is None:
        from opf.reference import solve as _ref_solve  # noqa: E402
        _solve_fn = _ref_solve
    return _solve_fn(qp, **kwargs)


def get_case(filepath: str = "data/case30.m") -> Case:
    """Load the MATPOWER case file, resolving relative to the repo root.

    Args:
        filepath: Path relative to the repository root.

    Returns:
        Case dataclass with system parameters.
    """
    return load_case(str(_REPO_ROOT / filepath))


def branch_ends(case: Case) -> list[tuple[int, int]]:
    """Return (from_bus, to_bus) pairs for every branch in the case.

    Args:
        case: Case dataclass from load_case().

    Returns:
        List of 41 (from_bus, to_bus) tuples (1-based bus indices).
    """
    return [
        (int(case.branch_from[i]), int(case.branch_to[i]))
        for i in range(len(case.branch_from))
    ]


def binding_lines(mu: np.ndarray, tol: float = 1e-6) -> list[int]:
    """Return indices of binding line constraints from inequality multipliers.

    In the DC-OPF ordering (CONTEXT.md §4):
        mu[:41]   – line-upper  (+flow ≤ F_max)
        mu[41:82] – line-lower  (−flow ≤ F_max)

    A line is binding when its upper *or* lower multiplier exceeds *tol*.

    Args:
        mu: Inequality multiplier vector, shape (94,).
        tol: Threshold for considering a multiplier active.

    Returns:
        Sorted list of 0-based line indices that are binding.
    """
    n_branch = 41
    upper = mu[:n_branch]
    lower = mu[n_branch : 2 * n_branch]
    active = set()
    for i in range(n_branch):
        if upper[i] > tol or lower[i] > tol:
            active.add(i)
    return sorted(active)


def kkt_residual(qp: QP, result: QPResult) -> float:
    """Compute the KKT stationarity residual ‖Hx + g − A_eqᵀλ + A_inᵀμ‖_∞.

    This is a lightweight wrapper; when opf.check is available it
    delegates there, otherwise computes it directly.

    Args:
        qp: The QP problem.
        result: A QPResult with status "optimal".

    Returns:
        Infinity-norm of the stationarity residual.
    """
    try:
        from opf.check import kkt_residual as _kkt  # type: ignore
        return _kkt(qp, result)
    except ImportError:
        pass

    # Fallback: compute directly from the sign convention.
    r = qp.H @ result.x + qp.g
    if len(result.lam) > 0:
        r -= qp.A_eq.T @ result.lam
    if len(result.mu) > 0:
        r += qp.A_in.T @ result.mu
    return float(np.max(np.abs(r)))
