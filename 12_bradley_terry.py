"""Step 11 (Sections 2.6 and 3.1.3): consensus ranking versus Bradley-Terry rankings.

LMArena ranks models with the Bradley-Terry model fitted to pairwise battles. Here it is fitted by
maximum likelihood (logistic regression without intercept; a negligible L2 term fixes the free
constant) to the battles between two of the 30 models, once for the 1,149 retained raters and once
for all raters, with ties of both kinds counted as half a win for each model. The rankings are
compared with the consensus ranking of the SOBO decision and with the flat ranking by Kendall's
tau, with a 95% bootstrap interval over battles.

Outputs results/analysis/bradley_terry.md and tables/bradley_terry.csv.

Usage:
    python 12_bradley_terry.py
"""
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

import config
from src.data import load_battles, load_instance
from src.scofcm import pipeline_from_instance, scalarize

N_BOOT = 200


def bt_strengths(df, models):
    """Bradley-Terry log-strengths of `models` from battles with columns model_a, model_b, winner."""
    idx = {m: j for j, m in enumerate(models)}
    a = df["model_a"].map(idx).to_numpy()
    b = df["model_b"].map(idx).to_numpy()
    X = np.zeros((len(df), len(models)))
    X[np.arange(len(df)), a] = 1.0
    X[np.arange(len(df)), b] = -1.0
    win_a = (df["winner"] == "model_a").to_numpy()
    win_b = (df["winner"] == "model_b").to_numpy()
    tie = ~(win_a | win_b)
    # decisive battles once; ties twice, as a half win for each side
    Xs = np.vstack([X[~tie], X[tie], X[tie]])
    ys = np.concatenate([win_a[~tie], np.ones(tie.sum()), np.zeros(tie.sum())]).astype(int)
    ws = np.concatenate([np.ones((~tie).sum()), np.full(2 * tie.sum(), 0.5)])
    model = LogisticRegression(fit_intercept=False, C=1e6, max_iter=10_000)
    model.fit(Xs, ys, sample_weight=ws)
    return model.coef_[0]


def tau(x, y):
    return float(stats.kendalltau(x, y)[0])


def main():
    out = config.RESULTS / "analysis"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    inst = load_instance(config.DATA_PROCESSED / "instance.npz")
    models = list(inst["models"])
    battles = load_battles()
    among = battles[battles["model_a"].isin(models) & battles["model_b"].isin(models)]
    retained = among[among["judge_hash"].isin(set(inst["raters"]))]

    # consensus of the SOBO decision (all seeds select the same partition and ranking; seed 1 is used)
    with open(config.LOGS / "seed01_SOBO_E.jsonl") as fh:
        recs = [json.loads(line) for line in fh]
    g = [scalarize(r["f_norm"], config.WEIGHTS["E"]) for r in recs]
    decision = pipeline_from_instance(inst).evaluate(recs[int(np.argmin(g))]["x"])
    consensus = decision.collective
    flat = np.nanmean(np.where(inst["observed"], inst["F"], np.nan), axis=0)

    bt_ret, bt_all = bt_strengths(retained, models), bt_strengths(among, models)
    rng = np.random.default_rng(0)
    boot = {"retained": [], "all": []}
    for _ in range(N_BOOT):
        for key, df in (("retained", retained), ("all", among)):
            sample = df.iloc[rng.integers(0, len(df), len(df))]
            boot[key].append(tau(consensus, bt_strengths(sample, models)))
    ci = {k: np.percentile(v, [2.5, 97.5]) for k, v in boot.items()}

    t = pd.DataFrame({"model": models, "consensus": consensus, "flat": flat, "bt_retained": bt_ret, "bt_all": bt_all})
    for col in ("consensus", "flat", "bt_retained", "bt_all"):
        t[f"rank_{col}"] = t[col].rank(ascending=False, method="min").astype(int)
    t.sort_values("rank_consensus").to_csv(out / "tables" / "bradley_terry.csv", index=False)
    top = lambda col: t.loc[t[col].idxmax(), "model"]
    md = ["# Consensus ranking vs Bradley-Terry rankings (30 models)",
          f"- Battles between two of the 30 models: {len(retained):,} by retained raters, {len(among):,} by all raters",
          f"- Kendall tau, consensus vs flat: {tau(consensus, flat):.3f}",
          f"- Kendall tau, consensus vs Bradley-Terry (retained raters): {tau(consensus, bt_ret):.3f} "
          f"[{ci['retained'][0]:.3f}, {ci['retained'][1]:.3f}]",
          f"- Kendall tau, consensus vs Bradley-Terry (all raters): {tau(consensus, bt_all):.3f} "
          f"[{ci['all'][0]:.3f}, {ci['all'][1]:.3f}]",
          f"- Kendall tau, Bradley-Terry retained vs all raters: {tau(bt_ret, bt_all):.3f}",
          f"- Top model: consensus {top('consensus')}, BT retained {top('bt_retained')}, BT all {top('bt_all')}",
          f"- Largest rank differences (consensus vs BT retained): " + ", ".join(
              f"{r.model} {r.rank_consensus} vs {r.rank_bt_retained}" for r in
              t.assign(d=(t["rank_consensus"] - t["rank_bt_retained"]).abs()).nlargest(3, "d").itertuples())]
    (out / "bradley_terry.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
