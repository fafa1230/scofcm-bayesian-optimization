"""Step 15 (revision; Sections 2.7, 3.1.4 and 3.2.2): synthetic problems with a controlled degree of conflict.

Extends 06_sanity_benchmark.py. The problems interpolate between the two synthetic problems used there,
    f_lambda(x) = (1 - lambda) * aligned(x) + lambda * DTLZ2(x),   x in [0, 1]^3, three objectives,
so lambda = 0 is the problem whose objectives share one minimizer and lambda = 1 is DTLZ2 (M = 3), with
a concave spherical front. For every lambda, the conflict is measured as in the pilot-sample rule of
Section 3.2.2: the smallest pairwise Spearman correlation of the objectives on 512 scrambled Sobol points
(seed 0). SOBO (equal weights, EI) and MOBO (EHVI) then run with the main settings and a budget of 40
evaluations (8 initial points) in ten paired replications (seeds 1-10). The summary relates the pilot
correlation to the hypervolume advantage of MOBO, which tests the rule under controlled conditions.

Outputs results/revision/synthetic_family.csv and synthetic_summary.md.

Usage:
    python 16_synthetic_conflict.py --jobs 6
"""
import warnings

warnings.filterwarnings("ignore", message=".*torch.jit.script.*", category=FutureWarning)

import argparse  # noqa: E402
import multiprocessing as mp  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import config  # noqa: E402

OUT = config.RESULTS / "revision"
LAMBDAS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
SEEDS = tuple(range(1, 11))
BUDGET = 40
A = np.array([0.48, 0.5, 0.5])


def dtlz2(x):
    g = (x[2] - 0.5) ** 2
    c1, s1 = np.cos(x[0] * np.pi / 2), np.sin(x[0] * np.pi / 2)
    c2, s2 = np.cos(x[1] * np.pi / 2), np.sin(x[1] * np.pi / 2)
    return (1 + g) * np.array([c1 * c2, c1 * s2, s1]) / 1.25          # scaled into [0, 1]


def aligned(x):
    d = np.sum((x - A) ** 2) / np.sum(np.maximum(A, 1 - A) ** 2)      # in [0, 1], minimum at A
    return np.array([d, np.sqrt(d), d ** 2])


def problem(lam):
    return lambda x: (1.0 - lam) * aligned(x) + lam * dtlz2(x)


def pilot_rho(lam):
    import torch

    X = torch.quasirandom.SobolEngine(config.DIM, scramble=True, seed=config.PILOT_SEED).draw(
        config.PILOT_SIZE, dtype=torch.double).numpy()
    Y = np.array([problem(lam)(x) for x in X])
    return min(stats.spearmanr(Y[:, i], Y[:, j])[0] for i, j in ((0, 1), (0, 2), (1, 2)))


def task(args):
    import torch

    from src.metrics import hypervolume, nondominated_mask
    from src.optimizers import run_bo, sobol_design
    from src.scofcm import scalarize

    lam, mode, seed = args
    torch.set_num_threads(2)
    fun = problem(lam)

    def evaluate(x):
        f = fun(np.asarray(x, dtype=float))
        return SimpleNamespace(f_norm=f, f_raw=f, c_star=0)

    recs = run_bo(evaluate, mode, sobol_design(seed), seed,
                  weights=config.WEIGHTS["E"] if mode == "SOBO" else None, budget=BUDGET)
    Y = np.array([r["f_norm"] for r in recs])
    return {"lambda": lam, "mode": mode, "seed": seed, "HV": hypervolume(Y),
            "best_g": min(scalarize(y, config.WEIGHTS["E"]) for y in Y),
            "pareto_points": int(nondominated_mask(Y).sum()),
            "t_acq": float(sum(r["t_acq"] for r in recs))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [(lam, m, s) for lam in LAMBDAS for m in ("MOBO", "SOBO") for s in SEEDS]
    with mp.get_context("spawn").Pool(args.jobs) as pool:
        rows = pool.map(task, tasks)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "synthetic_family.csv", index=False)
    W = D.pivot_table(index=["lambda", "seed"], columns="mode", values=["HV", "best_g", "pareto_points", "t_acq"])
    W.columns = [f"{a}_{b}" for a, b in W.columns]
    W = W.reset_index()
    W["dHV"] = W["HV_MOBO"] - W["HV_SOBO"]
    rho = {lam: pilot_rho(lam) for lam in LAMBDAS}
    W["rho"] = W["lambda"].map(rho)
    md = ["# Synthetic problems with controlled conflict (10 paired replications, 40 evaluations)", "",
          "| lambda | pilot min rho | HV SOBO median [IQR] | HV MOBO median [IQR] | MOBO larger HV | p (Wilcoxon) | "
          "best g SOBO / MOBO | Pareto points SOBO / MOBO | acquisition s SOBO / MOBO |",
          "|---|---|---|---|---|---|---|---|---|"]
    for lam, g in W.groupby("lambda"):
        q = lambda s: f"{s.median():.3f} [{s.quantile(0.25):.3f}, {s.quantile(0.75):.3f}]"
        p = stats.wilcoxon(g["HV_MOBO"], g["HV_SOBO"])[1] if (g["dHV"] != 0).any() else 1.0
        md.append(f"| {lam:.1f} | {rho[lam]:.2f} | {q(g['HV_SOBO'])} | {q(g['HV_MOBO'])} | "
                  f"{int((g['dHV'] > 0).sum())}/{len(g)} | {p:.3g} | {g['best_g_SOBO'].median():.3f} / "
                  f"{g['best_g_MOBO'].median():.3f} | {g['pareto_points_SOBO'].median():.0f} / "
                  f"{g['pareto_points_MOBO'].median():.0f} | {g['t_acq_SOBO'].median():.0f} / {g['t_acq_MOBO'].median():.0f} |")
    sp = stats.spearmanr(W["rho"], W["dHV"])
    neg = W["rho"] < 0
    table = [[int((W.loc[neg, "dHV"] > 0).sum()), int((W.loc[neg, "dHV"] <= 0).sum())],
             [int((W.loc[~neg, "dHV"] > 0).sum()), int((W.loc[~neg, "dHV"] <= 0).sum())]]
    md += ["", f"- Spearman, pilot rho vs HV difference over {len(W)} replications: {sp[0]:.2f} (p = {sp[1]:.3g})",
           f"- Negative pilot rho: MOBO larger HV in {table[0][0]} of {sum(table[0])}; non-negative: "
           f"{table[1][0]} of {sum(table[1])}; one-sided Fisher exact p = {stats.fisher_exact(table, alternative='greater')[1]:.3g}"]
    (OUT / "synthetic_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
