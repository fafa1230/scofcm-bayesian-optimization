"""Step 8 (Sections 2.6 and 3.1.1): random search as a reference, and confidence intervals for RQ1.

Random search keeps the 8 initial Sobol points of each seed and replaces the 52 BO iterations by
uniform random configurations, so each seed gives a paired comparison with the same budget of 60.
SOBO and MOBO are compared with it by the paired Wilcoxon test (Holm over the four tests) and A12.
The median paired differences MOBO - SOBO of Table 4 get 95% bootstrap confidence intervals.

Outputs results/analysis/random_search.md and tables/random_search_per_seed.csv.

Usage:
    python 09_random_search.py
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

import config
from src.data import load_instance
from src.metrics import a12, a12_magnitude, holm, hypervolume
from src.scofcm import pipeline_from_instance, scalarize

N_BOOT = 10_000


def load_logs(log_dir):
    rows = []
    for path in sorted(log_dir.glob("seed*_*.jsonl")):
        with open(path) as fh:
            rows.extend(json.loads(line) for line in fh)
    return pd.DataFrame(rows)


def arrays(df, seed, run):
    d = df[(df["seed"] == seed) & (df["run"] == run)].sort_values("iter")
    return np.array(d["x"].tolist()), np.array(d["f_norm"].tolist())


def median_ci(diff, rng):
    """Median of paired differences with a 95% percentile bootstrap interval."""
    diff = np.asarray(diff, float)
    boot = np.median(rng.choice(diff, size=(N_BOOT, len(diff))), axis=1)
    return float(np.median(diff)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main():
    out = config.RESULTS / "analysis"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    wE = config.WEIGHTS["E"]
    pipe = pipeline_from_instance(load_instance(config.DATA_PROCESSED / "instance.npz"))
    df = load_logs(config.LOGS)

    rows = []
    for s in sorted(df["seed"].unique()):
        _, Ys = arrays(df, s, "SOBO_E")
        _, Ym = arrays(df, s, "MOBO")
        rng = np.random.default_rng(1000 + int(s))
        X_rand = rng.random((config.BUDGET - config.N_INIT, config.DIM))
        Yr = np.vstack([Ys[:config.N_INIT], [pipe.evaluate(x).f_norm for x in X_rand]])
        rec = {"seed": s}
        for name, Y in (("SOBO", Ys), ("MOBO", Ym), ("RS", Yr)):
            rec[f"HV_{name}"] = hypervolume(Y)
            rec[f"bestg_{name}"] = float(min(scalarize(y, wE) for y in Y))
        rows.append(rec)
    t = pd.DataFrame(rows)
    t.to_csv(out / "tables" / "random_search_per_seed.csv", index=False)

    # BO vs random search: larger HV and smaller g are better
    tests = [(m, k) for m in ("SOBO", "MOBO") for k in ("HV", "bestg")]
    p = [float(stats.wilcoxon(t[f"{k}_{m}"] - t[f"{k}_RS"]).pvalue) for m, k in tests]
    p_adj = holm(p)
    md = ["# Random search (same 8 initial points + 52 uniform points) and RQ1 confidence intervals",
          f"- Random search: HV median {t['HV_RS'].median():.3f} [{t['HV_RS'].quantile(.25):.3f}, "
          f"{t['HV_RS'].quantile(.75):.3f}], best g median {t['bestg_RS'].median():.4f} "
          f"[{t['bestg_RS'].quantile(.25):.4f}, {t['bestg_RS'].quantile(.75):.4f}]"]
    for (m, k), pr, pa in zip(tests, p, p_adj):
        a = a12(t[f"{k}_{m}"], t[f"{k}_RS"]) if k == "HV" else a12(t[f"{k}_RS"], t[f"{k}_{m}"])
        wins = int((t[f"{k}_{m}"] > t[f"{k}_RS"]).sum() if k == "HV" else (t[f"{k}_{m}"] < t[f"{k}_RS"]).sum())
        md.append(f"- {m} vs random search, {k}: median {t[f'{k}_{m}'].median():.4f} vs {t[f'{k}_RS'].median():.4f}; "
                  f"better in {wins} of {len(t)} seeds; p = {pr:.2e}, p_Holm = {pa:.2e}; "
                  f"A12 ({m} better) = {a:.2f} ({a12_magnitude(a)})")

    rng = np.random.default_rng(0)
    md.append("\n## Median paired difference MOBO - SOBO with 95% bootstrap interval")
    for k, label in (("HV", "hypervolume"), ("bestg", "best weighted sum")):
        med, lo, hi = median_ci(t[f"{k}_MOBO"] - t[f"{k}_SOBO"], rng)
        md.append(f"- {label}: {med:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    (out / "random_search.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
