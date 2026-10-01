"""Sanity check: the same SOBO/MOBO code on two synthetic 3-objective problems in [0, 1]^3.

  conflicting : DTLZ2 (M = 3, d = 3), a concave spherical Pareto front with genuine trade-offs.
  aligned     : three objectives that share one minimizer, so the Pareto front is a single point.

If the implementation is correct, MOBO should clearly beat SOBO on hypervolume for the
conflicting problem and tie with it on the aligned one.

Usage:
    python 06_sanity_benchmark.py --seeds 1 2 3 4 5 --budget 40 --jobs 6
"""
import warnings

warnings.filterwarnings("ignore", message=".*torch.jit.script.*", category=FutureWarning)
import argparse
import multiprocessing as mp
from types import SimpleNamespace

import numpy as np
import pandas as pd

import config

A = np.array([0.48, 0.5, 0.5])


def dtlz2(x):
    g = (x[2] - 0.5) ** 2
    c1, s1 = np.cos(x[0] * np.pi / 2), np.sin(x[0] * np.pi / 2)
    c2, s2 = np.cos(x[1] * np.pi / 2), np.sin(x[1] * np.pi / 2)
    return (1 + g) * np.array([c1 * c2, c1 * s2, s1]) / 1.25          # scaled into [0, 1]


def aligned(x):
    d = np.sum((x - A) ** 2) / np.sum(np.maximum(A, 1 - A) ** 2)      # in [0, 1], minimum at A
    return np.array([d, np.sqrt(d), d ** 2])


PROBLEMS = {"conflicting": dtlz2, "aligned": aligned}


def task(args):
    import torch
    from src.metrics import hypervolume, nondominated_mask
    from src.optimizers import run_bo, sobol_design

    problem, mode, seed, budget = args
    torch.set_num_threads(2)
    fun = PROBLEMS[problem]

    def evaluate(x):
        f = fun(np.asarray(x, dtype=float))
        return SimpleNamespace(f_norm=f, f_raw=f, c_star=0)

    weights = config.WEIGHTS["E"] if mode == "SOBO" else None
    recs = run_bo(evaluate, mode, sobol_design(seed), seed, weights=weights, budget=budget)
    Y = np.array([r["f_norm"] for r in recs])
    return {"problem": problem, "mode": mode, "seed": seed, "HV": hypervolume(Y),
            "pareto_points": int(nondominated_mask(Y).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    tasks = [(p, m, s, args.budget) for p in PROBLEMS for m in ("SOBO", "MOBO") for s in args.seeds]
    with mp.get_context("spawn").Pool(args.jobs) as pool:
        rows = pool.map(task, tasks)
    df = pd.DataFrame(rows)
    out = config.RESULTS / "analysis" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "sanity_benchmark.csv", index=False)
    summary = df.groupby(["problem", "mode"])[["HV", "pareto_points"]].median().round(4)
    print(summary.to_string())


if __name__ == "__main__":
    main()
