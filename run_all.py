#!/usr/bin/env python
"""Regenerate all results, tables and figures from scratch.

Usage:
    python run_all.py

Requires:
    numpy, scipy, cvxpy, matplotlib, pandas  (see requirements.txt)
"""

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

# Create results/ if missing
(_ROOT / "results").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Run experiments in order
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("run_all.py – Albert and Einstein OPF results")
    print("=" * 60)

    print("\n[1/4] Congestion sweep (rating scale 1.00 → 0.70) ...")
    from experiments.congestion import run as run_congestion
    run_congestion()

    print("\n[2/4] Load sweep (ED, load scale 0.50 → 1.75) ...")
    from experiments.load_sweep import run as run_load
    run_load()

    print("\n[3/4] Validation table and +1 MW check ...")
    from experiments.validation import run as run_validation
    run_validation()

    print("\n[4/4] Generating figures ...")
    from experiments.plots import run as run_plots
    run_plots()

    print("\n" + "=" * 60)
    print("All results saved to results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
