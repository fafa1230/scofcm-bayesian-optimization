"""Step 0 (Section 2.7): verify the SCO-FCM implementation on the case study of Yang et al. (2026).

Fill three CSV files from the baseline paper into data/yang_case/ (no header, comma-separated):
    F.csv    20 x 4  synthesis scores of the 20 DMs on alternatives a1..a4 (Table 3)
    hbc.csv  20 x 20 historical-background correlation matrix (alpha, Appendix)
    crc.csv  20 x 20 current-research correlation matrix (beta, Appendix)

Expected result reported by the baseline with zeta = 0.4, kappa = 0.6:
    optimal number of clusters = 8, ranking a4 > a1 > a3 > a2.

Usage:
    python 04_verify_baseline.py
"""
from pathlib import Path

import numpy as np

import config
from src.data import _pagerank
from src.scofcm import SCOFCMPipeline

CASE = config.ROOT / "data" / "yang_case"


def main():
    files = {k: CASE / f"{k}.csv" for k in ("F", "hbc", "crc")}
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise SystemExit("Missing input files:\n  " + "\n  ".join(missing) + "\nSee the docstring.")
    F = np.loadtxt(files["F"], delimiter=",")
    alpha = np.loadtxt(files["hbc"], delimiter=",")
    beta = np.loadtxt(files["crc"], delimiter=",")
    np.fill_diagonal(alpha, 1.0)
    np.fill_diagonal(beta, 1.0)
    n = len(F)
    for mode in ("sum", "mean"):
        pipe = SCOFCMPipeline(F, alpha, beta, _pagerank(alpha), _pagerank(beta),
                              c_candidates=range(2, n // 2 + 1), cc_mode=mode)
        res = pipe.evaluate((0.4, 0.6, 0.5))
        ranking = " > ".join(f"a{i + 1}" for i in res.ranking) if res.ranking is not None else "n/a"
        print(f"cc_mode={mode:4s}  c* = {res.c_star}  ranking: {ranking}  "
              f"converged: {sum(res.converged_by_c.values())}/{len(res.converged_by_c)}")
    print("Expected (baseline): c* = 8, ranking a4 > a1 > a3 > a2")


if __name__ == "__main__":
    main()
