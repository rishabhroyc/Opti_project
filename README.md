# Economic Dispatch and DC Optimal Power Flow

**Team:** Albert and Einstein

| Member | Role |
|---|---|
| Rishabh Roy Choudhury | Models, reference solver, infrastructure |
| Varshith Tadiparthi | Active-set QP solver, lambda-iteration solver |
| B. Rithwik | Experiments, figures, report |

## Description

We find the cheapest way to dispatch generators on the IEEE 30-bus
power system (6 generators, 41 lines, 189.2 MW demand). **Economic
Dispatch (ED)** ignores the network; **DC Optimal Power Flow (DC-OPF)**
adds line-flow constraints. Both are convex quadratic programs solved
with our own active-set solver (pure NumPy iterations) in all experiments;
the CVXPY + Clarabel reference solver is used only for validation.
The key result: Lagrange multipliers of the balance constraints are
electricity prices (Locational Marginal Prices), equal everywhere until
lines congest, then they split apart.

**Data source:** The IEEE 30-bus test case (`data/case30.m`) is adapted from the MATPOWER distribution \cite{zimmerman2011}.

## Requirements

Python 3.12 or newer is required.

| Package | Version | Purpose |
|---|---|---|
| numpy | == 2.5.5 | Core numerics |
| scipy | == 1.18.1 | `linprog` for feasible-start only |
| cvxpy | == 1.9.3 | Reference solver (Clarabel backend) for validation |
| matplotlib | == 3.11.2 | Figure generation |
| pandas | == 3.1.1 | CSV data handling and table processing |
| pytest | == 9.2.0 | Test suite |

## Setup

All commands should be executed from the repository root.

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv

# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# Linux / macOS:
source .venv/bin/activate

# Install exact pinned dependencies
pip install -r requirements.txt
```

## How to Run

All commands are run from the repository root:

```bash
# Run tests (runs cleanly via pytest.ini)
pytest

# Run all experiments and regenerate results + figures
python run_all.py

# Individual experiments
python -m experiments.congestion
python -m experiments.load_sweep
python -m experiments.validation
python -m experiments.plots
```

All outputs are saved to `results/`.

## Repository Structure

```
data/case30.m               IEEE 30-bus test case (MATPOWER format)
src/opf/qp.py               Shared QP interface (QP, QPResult)
src/opf/models.py            Case loading, build_ed(), build_dcopf()
src/opf/check.py             KKT residual and feasibility verification
src/opf/reference.py         CVXPY + Clarabel reference solver
src/opf/active_set.py        Primal active-set QP solver
src/opf/lambda_iteration.py  Lambda-iteration ED solver
experiments/common.py        Shared experiment utilities
experiments/congestion.py    Rating-sweep experiment
experiments/load_sweep.py    ED load-sweep experiment
experiments/validation.py    Solver validation and sensitivity
experiments/plots.py         Figure generation
run_all.py                   Master script to regenerate everything
tests/                       pytest test suite
report/                      LaTeX report source
```
