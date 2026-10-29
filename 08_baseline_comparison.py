"""Step 7 (Section 3.1.1 and the Discussion): gain of the optimized settings over the baseline setting.

The baseline fixes (zeta, kappa, mu) = (0.4, 0.6, 0.5). For every seed, the best equally weighted sum
g_wE found by SOBO and by MOBO is compared with g_wE of the baseline setting by a one-sample Wilcoxon
signed-rank test (Holm-adjusted over the two methods). The raw objectives and the decision of the
selected configurations are listed next to those of the baseline setting.

Outputs results/analysis/baseline_comparison.md and tables/baseline_comparison.csv.

Usage:
    python 08_baseline_comparison.py
"""
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score

import config
from src.data import load_instance
from src.metrics import compromise_index, holm
from src.scofcm import pipeline_from_instance, scalarize

BASELINE = (0.4, 0.6, 0.5)               # (zeta, kappa, mu) of the baseline


def load_logs(log_dir):
    rows = []
    for path in sorted(log_dir.glob("seed*_*.jsonl")):
        with open(path) as fh:
            rows.extend(json.loads(line) for line in fh)
    return pd.DataFrame(rows)


def arrays(df, seed, run):
    d = df[(df["seed"] == seed) & (df["run"] == run)].sort_values("iter")
    return np.array(d["x"].tolist()), np.array(d["f_norm"].tolist())


def main():
    out = config.RESULTS / "analysis"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    wE = config.WEIGHTS["E"]
    pipe = pipeline_from_instance(load_instance(config.DATA_PROCESSED / "instance.npz"))
    df = load_logs(config.LOGS)

    base = pipe.evaluate(BASELINE)
    g_base = scalarize(base.f_norm, wE)
    rows = []
    for s in sorted(df["seed"].unique()):
        Xs, Ys = arrays(df, s, "SOBO_E")
        Xm, Ym = arrays(df, s, "MOBO")
        gs = np.array([scalarize(y, wE) for y in Ys])
        gm = np.array([scalarize(y, wE) for y in Ym])
        r_s = pipe.evaluate(Xs[int(np.argmin(gs))])
        r_m = pipe.evaluate(Xm[compromise_index(Ym)])
        rows.append({"seed": s, "bestg_SOBO": gs.min(), "bestg_MOBO": gm.min(),
                     "f1_SOBO": r_s.f_raw[0], "f2_SOBO": r_s.f_raw[1], "f3_SOBO": r_s.f_raw[2],
                     "f1_MOBO": r_m.f_raw[0], "f2_MOBO": r_m.f_raw[1], "f3_MOBO": r_m.f_raw[2],
                     "ARI_base_SOBO": adjusted_rand_score(base.labels, r_s.labels),
                     "same_ranking_base_SOBO": bool(np.array_equal(base.ranking, r_s.ranking))})
    t = pd.DataFrame(rows)
    t.to_csv(out / "tables" / "baseline_comparison.csv", index=False)

    p = [float(stats.wilcoxon(t[f"bestg_{m}"] - g_base).pvalue) for m in ("SOBO", "MOBO")]
    p_adj = holm(p)
    md = ["# Gain over the baseline setting (zeta, kappa, mu) = (0.4, 0.6, 0.5)",
          f"- Baseline: g_wE = {g_base:.4f}; f1 = {base.f_raw[0]:.4f}, f2 = {base.f_raw[1]:.5f}, "
          f"f3 = {base.f_raw[2]:.0f}; c* = {base.c_star}"]
    for (m, pa) in zip(("SOBO", "MOBO"), p_adj):
        g = t[f"bestg_{m}"]
        md.append(f"- {m}: best g_wE median {g.median():.4f} [{g.quantile(.25):.4f}, {g.quantile(.75):.4f}], "
                  f"{100 * (g_base - g.median()) / g_base:.1f}% lower than the baseline; better than the baseline "
                  f"in {int((g < g_base).sum())} of {len(g)} seeds; one-sample Wilcoxon p_Holm = {pa:.2e}")
        md.append(f"  selected decision: f1 median {t[f'f1_{m}'].median():.4f}, f2 median {t[f'f2_{m}'].median():.5f}, "
                  f"f3 median {t[f'f3_{m}'].median():.0f}")
    md.append(f"- Partition of the baseline setting vs SOBO decision: ARI median {t['ARI_base_SOBO'].median():.2f} "
              f"(min {t['ARI_base_SOBO'].min():.2f}); same ranking in {int(t['same_ranking_base_SOBO'].sum())} of {len(t)} seeds")
    (out / "baseline_comparison.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
