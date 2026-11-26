# Economic Dispatch and DC Optimal Power Flow

**Team:** Albert and Einstein

| Member | Role |
|---|---|
| Rishabh Roy Choudhury | Models, reference solver, infrastructure |
| Varshith Tadiparthi | Active-set QP solver, lambda-iteration solver |
| B. Rithwik | Experiments, figures, report |

## Description

We find the cheapest way to dispatch generators on the IEEE 30-bus
power system (6 generators, 41 lines, 189.2 MW demand).  **Economic
Dispatch (ED)** ignores the network; **DC Optimal Power Flow (DC-OPF)**
adds line-flow constraints.  Both are convex quadratic programs solved
with our own active-set solver (NumPy only) and validated against a
CVXPY + Clarabel reference.  The key result: Lagrange multipliers of
the balance constraints are electricity prices (Locational Marginal
Prices), equal everywhere until lines congest, then they split apart.

## Libraries

| Package | Version | Purpose |
|---|---|---|
| numpy | ≥ 1.24 | Core numerics |
| scipy | ≥ 1.10 | `linprog` for feasible-start only |
| cvxpy | ≥ 1.4 | Reference solver (Clarabel backend) |
| matplotlib | ≥ 3.7 | Figures |
| pandas | ≥ 2.0 | CSV reading in plots |
| pytest | ≥ 7.0 | Tests |

<!-- Rishabh: edit the setup section below when finalizing requirements -->
## Setup

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## How to Run

```bash
# Run all experiments and generate results + figures
python run_all.py

# Run tests
pytest

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
