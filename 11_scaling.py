"""Step 10 (Sections 2.7 and 3.2.2): evaluation time of Phi as the rater population grows.

Larger populations are built by resampling the 1,149 raters with replacement (profiles, observed
masks, relation matrices, and PageRank values follow the resampled raters), and one evaluation of
Phi is timed at three configurations for each size. A power law t = a n^b fitted to the medians
gives the population at which an evaluation reaches the MOBO break-even of Section 3.1.2 and the
expected time for all 49,382 raters of the dataset.

Outputs results/analysis/scaling.md and tables/scaling.csv.

Usage:
    python 11_scaling.py
"""
import time

import numpy as np
import pandas as pd

import config
from src.data import load_instance
from src.scofcm import SCOFCMPipeline

MULTIPLIERS = (1, 2, 4, 8)
CONFIGS = ((0.484, 0.3, 0.5), (0.4, 0.6, 0.5), (0.5, 0.5, 0.5))
BREAK_EVEN = 0.84                     # s per evaluation (Section 3.1.2)
ALL_RATERS = 49_382


def resampled(inst, k, rng):
    n = len(inst["F"])
    idx = np.arange(n) if k == 1 else rng.integers(0, n, k * n)
    pr_h, pr_c = inst["pr_h"][idx], inst["pr_c"][idx]
    return SCOFCMPipeline(inst["F"][idx], inst["alpha"][np.ix_(idx, idx)], inst["beta"][np.ix_(idx, idx)],
                          pr_h / pr_h.sum(), pr_c / pr_c.sum(), bounds=(inst["bounds_lo"], inst["bounds_hi"]),
                          observed=inst["observed"][idx])


def main():
    out = config.RESULTS / "analysis"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    inst = load_instance(config.DATA_PROCESSED / "instance.npz")
    rng = np.random.default_rng(0)
    rows = []
    for k in MULTIPLIERS:
        pipe = resampled(inst, k, rng)
        for x in CONFIGS:
            t0 = time.perf_counter()
            r = pipe.evaluate(x)
            rows.append({"n": k * len(inst["F"]), "x": x, "seconds": time.perf_counter() - t0,
                         "iterations": r.f_raw[2]})
        print(rows[-1]["n"], [round(r["seconds"], 3) for r in rows[-len(CONFIGS):]], flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(out / "tables" / "scaling.csv", index=False)

    med = t.groupby("n")["seconds"].median()
    b, log_a = np.polyfit(np.log(med.index.to_numpy(float)), np.log(med.to_numpy()), 1)
    n_even = np.exp((np.log(BREAK_EVEN) - log_a) / b)
    t_all = np.exp(log_a) * ALL_RATERS ** b
    md = ["# Evaluation time of Phi versus the number of raters (resampled populations)",
          *[f"- n = {n:,}: median {s:.3f} s per evaluation" for n, s in med.items()],
          f"- Power law: t = {np.exp(log_a):.3g} * n^{b:.2f}",
          f"- One evaluation reaches the MOBO break-even of {BREAK_EVEN} s at about n = {n_even:,.0f} raters",
          f"- Extrapolated time for all {ALL_RATERS:,} raters: about {t_all:.0f} s per evaluation"]
    (out / "scaling.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
