# CONTEXT.md: Economic Dispatch and DC Optimal Power Flow

> Keep this file in the repository root (rename to `CLAUDE.md` or `AGENTS.md` if your AI tool expects that name).
> Start each AI session with who you are and your task, e.g. "I'm Varshith, working on the active-set solver."

## 1. The project

Team **Albert and Einstein** (Rishabh Roy Choudhury, Varshith Tadiparthi, B. Rithwik), optimization course project.

We find the cheapest way to run a power grid. On the IEEE 30-bus test system (6 generators with quadratic costs, 41 lines, 189.2 MW demand) we choose each generator's output so demand is met, generators stay within limits and lines stay within ratings. **Economic Dispatch (ED)** ignores the network; **DC Optimal Power Flow (DC-OPF)** adds it. Both are convex QPs, solved with **our own active-set solver** (NumPy only) and checked against a cvxpy reference. The main finding: the Lagrange multipliers of the balance constraints are electricity prices (LMPs), equal at every bus until lines congest, then split apart.

**Graded on:** formulation 30%, implementation 30%, results and analysis 20%, demo and viva 20%.
**Deliverables:** report (problem and motivation; formulation; methods and solver; results and discussion), `albert_and_einstein.zip` with a README (team name, members, title, short description, libraries, setup), demo and viva.
**Code freeze:** Sat 28 Nov 2026 (submission window 28 Nov to 4 Dec).

## 2. Model

```
ED:      min Σ_i (b_i P_i + c_i P_i²)   s.t.  Σ_i P_i = D  (λ),   P_min ≤ P ≤ P_max

DC-OPF:  min Σ_i (b_i P_i + c_i P_i²)
         s.t.  Σ_{i at k} P_i − d_k = Σ_l f_kl         ∀ bus k   (λ_k = LMP)
               f_kl = S_base (θ_k − θ_l) / x_kl
               −F_max ≤ f_kl ≤ F_max                   ∀ line
               P_min ≤ P ≤ P_max,   θ_ref = 0
```

- Convex: Hessian diag(2c_i) > 0 on P, zero on θ; linear constraints, so KKT conditions are necessary and sufficient.
- Unconstrained generators satisfy b_i + 2c_i P_i = λ (equal incremental cost).
- B_bus has rank n − 1, hence the reference angle. H is singular on θ, but the KKT matrix stays invertible because the balance rows fix θ once P is chosen.

## 3. Data and reference values

Data: MATPOWER `data/case30.m` (unmodified). 30 buses, 6 generators, 41 lines, 189.2 MW, base 100 MVA, reference bus 1. All line ratings are 16–130 MVA (none is 0, which MATPOWER would treat as unlimited). `gencost` rows are `[2 0 0 3 c2 c1 c0]`.

Every solver must reproduce these (checked three independent ways):

| Scenario | Result |
|---|---|
| Base case (ED = DC-OPF) | cost **565.21**, λ = **3.7892** at every bus, dispatch 44.73, 58.26, 22.31, 32.33, 15.78, 15.78 MW, no limit binding |
| Busiest line, base case | 6–8 at 76.4% of 32 MVA |
| Ratings × 0.75 | cost 565.97, lines 6–8 and 25–27 bind, LMPs 3.65–7.18 |
| Ratings × 0.72 | cost 589.58, four lines bind, LMPs 3.02–39.65 |
| Ratings × 0.71 or less | infeasible |

Compare cost and λ to ~1e-6 relative, dispatch to 1e-4 MW.

## 4. Code

```
data/case30.m
src/opf/qp.py                SHARED interface (change only if all three agree)
src/opf/models.py            Rishabh: load_case(), build_ed(), build_dcopf(), feasible_start()
src/opf/reference.py         Rishabh: cvxpy + Clarabel, same signature as our solver
src/opf/check.py             Rishabh: kkt_residual(), comparison table
src/opf/lambda_iteration.py  Varshith
src/opf/active_set.py        Varshith
experiments/congestion.py    Rithwik: rating sweep
experiments/load_sweep.py    Rithwik: ED price vs demand
experiments/plots.py         Rithwik: the four figures
run_all.py                   Rithwik: regenerates every result
tests/test_models.py         Rishabh
tests/test_solvers.py        Varshith
report/                      formulation.tex (Rishabh), methods.tex (Varshith), main.tex + results.tex (Rithwik)
```

### Interface (`src/opf/qp.py`)

Standard form: minimize ½xᵀHx + gᵀx subject to A_eq x = b_eq, A_in x ≤ b_in.

```python
@dataclass
class QP:
    H: np.ndarray; g: np.ndarray
    A_eq: np.ndarray; b_eq: np.ndarray
    A_in: np.ndarray; b_in: np.ndarray
    names: list[str]          # variable names

@dataclass
class QPResult:
    x: np.ndarray; obj: float
    lam: np.ndarray           # equality multipliers
    mu: np.ndarray            # inequality multipliers (>= 0)
    status: str               # "optimal" | "infeasible" | "max_iter" | "singular"
    iters: int
    history: list[float]      # objective per iteration

def solve(qp: QP, x0=None, tol=1e-8, max_iter=500) -> QPResult: ...
```

### Conventions

- **Sign:** L = f − λᵀ(A_eq x − b_eq) + μᵀ(A_in x − b_in). Stationarity: Hx + g − A_eqᵀλ + A_inᵀμ = 0. So λ is positive (+3.7892 in the base case) and each balance-row λ is that bus's LMP. Fix cvxpy's sign in `reference.py` if needed.
- **ED:** x = [P1..P6]; 1 equality; 12 inequalities (6 upper, then 6 lower as −P ≤ −P_min).
- **DC-OPF:** x = [P1..P6, θ1..θ30] (36); equalities = 30 balance rows then θ_ref = 0 (31); inequalities = 41 line upper, 41 line lower, 6 gen upper, 6 gen lower (94).
- **Matrices:** balance rows [C_g, −S_base·B_bus] x = d with B_bus = Cᵀ diag(1/x) C; line rows ±S_base·diag(1/x)·C θ ≤ F_max. H = diag(2c) on P (not c); g = b on P.
- **Units:** MW, radians, cost per hour as in case30. `build_dcopf(case, load_scale=1.0, rate_scale=1.0)`.

## 5. Solvers

**Lambda iteration (ED):** P_i(λ) = clip((λ − b_i)/(2c_i), P_min, P_max); bisect λ until Σ P_i = D.

**Active set (Nocedal and Wright, Algorithm 16.3):**
1. Start from a feasible x0 (`feasible_start()`); initial working set W = active constraints at x0, added one at a time only if they raise the rank of [A_eq; A_W].
2. Solve the KKT system of equalities + W for step p and multipliers.
3. If p ≈ 0: stop if all μ in W ≥ −tol, else drop the most negative.
4. Else ratio test: α = min(1, min over i ∉ W with a_iᵀp > 0 of (b_i − a_iᵀx)/(a_iᵀp)); x ← x + αp; if α < 1 add the blocking constraint.
5. Ties by lowest index; stop at max_iter; return "singular" if the KKT solve fails.

**Feasible start:** use the ED solution if it satisfies all constraints, else `scipy.optimize.linprog` (initialization only; say so in the report). If linprog finds none, the scenario is "infeasible".

## 6. Results to produce

- **Validation table:** lambda iteration, active set and reference on the base case and ratings × 0.75: cost, max |Δx|, max |Δλ|, KKT residual, iterations. Plus one row: add 1 MW at a bus and show the cost rises by that bus's λ.
- **Figure 1:** ED price λ vs demand (load × 0.5 to 1.75), kinks where generators hit limits.
- **Figure 2:** cost and LMP spread vs rating scale (1.00 down to 0.70, step 0.005), infeasible region shaded.
- **Figure 3:** LMP per bus at rating scales 1.00, 0.75 and 0.72.
- **Figure 4:** active-set objective per iteration.

## 7. Rules for the AI

- Ask which team member you are helping; edit only that person's files (section 4). Never change `qp.py` or the conventions on your own.
- `lambda_iteration.py` and `active_set.py` use NumPy only, never cvxpy, scipy.optimize or any QP library.
- cvxpy only in `reference.py`, always with `solver=cp.CLARABEL` (OSQP gave wrong answers near infeasibility). `linprog` only in `feasible_start()`.
- Never invent numbers. Every result in code comments, README or report comes from running the code. Never loosen tolerances to make tests pass.
- Write a `pytest` test with each function and run `pytest` before saying a task is done.
- Keep code simple and readable (it is explained in a viva): type hints, short docstrings with shapes and units, figures with labelled axes saved to `results/`.
- Git: one branch per task, pull request into `main`, never push to `main` directly.
- When unsure, ask instead of guessing.

## 8. Done when

- [ ] `pytest` passes and our active-set solver matches the reference on both scenarios (KKT residual < 1e-6).
- [ ] `python run_all.py` from a fresh clone regenerates the table and four figures.
- [ ] Report, README and `albert_and_einstein.zip` complete and tested from a fresh folder.
- [ ] All three of us can explain every part.