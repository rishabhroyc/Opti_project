"""Shared Quadratic Program (QP) interface and data structures."""

from dataclasses import dataclass
import numpy as np

# Sign convention (from CONTEXT.md section 4):
# L = f - lam^T (A_eq x - b_eq) + mu^T (A_in x - b_in), with mu >= 0.
# Stationarity: Hx + g - A_eq^T lam + A_in^T mu = 0.
# Under this convention:
# - lam is positive (+3.7892 in the base case) and equals d(cost*)/d(b_eq).
# - Each balance-row lam is that bus's LMP.
# - Status values: "optimal" | "infeasible" | "max_iter" | "singular".


@dataclass
class QP:
    H: np.ndarray
    g: np.ndarray
    A_eq: np.ndarray
    b_eq: np.ndarray
    A_in: np.ndarray
    b_in: np.ndarray
    names: list[str]  # variable names


@dataclass
class QPResult:
    x: np.ndarray
    obj: float
    lam: np.ndarray  # equality multipliers
    mu: np.ndarray  # inequality multipliers (>= 0)
    status: str  # "optimal" | "infeasible" | "max_iter" | "singular"
    iters: int
    history: list[float]  # objective per iteration


def solve(
    qp: QP,
    x0: np.ndarray | None = None,
    tol: float = 1e-8,
    max_iter: int = 500,
) -> QPResult:
    """Solve the quadratic program.

    Real versions live in active_set.py and reference.py.
    """
    raise NotImplementedError("solve() is implemented in active_set.py and reference.py")
