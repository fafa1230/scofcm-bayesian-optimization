"""Step 5 (Section III-A-3 and the Discussion): what the selected decisions look like.

Outputs results/analysis/decisions.md and tables/decisions_*.csv:
  - the SOBO and MOBO decisions of every seed: parameters, subgroup sizes and weights, agreement
    of the two subgroups' rankings, and agreement of the consensus with the flat ranking;
  - the effect of mu on the subgroup weights and on f2 at the selected zeta;
  - the band of zeta with a low weighted sum (from 05_diagnostics.py);
  - how many configurations in the MOBO Pareto sets change the partition or the ranking,
    for the main run and for the sensitivity run with imputed subgroup scores.

Usage:
    python 07_decision_analysis.py
"""
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score

import config
from src.data import load_instance
from src.metrics import compromise_index, nondominated_mask
from src.scofcm import pipeline_from_instance, scalarize

RUNS = {
    "main": {"instance": config.DATA_PROCESSED / "instance.npz", "logs": config.LOGS, "scores": "observed"},
    "imputed": {"instance": config.DATA_PROCESSED / "archive_imputed_scores" / "instance.npz",
                "logs": config.RESULTS / "archive_imputed_scores" / "logs", "scores": "imputed"},
}


def load_logs(log_dir):
    rows = []
    for path in sorted(log_dir.glob("seed*_*.jsonl")):
        with open(path) as fh:
            rows.extend(json.loads(line) for line in fh)
    return pd.DataFrame(rows)


def arrays(df, seed, run):
    d = df[(df["seed"] == seed) & (df["run"] == run)].sort_values("iter")
    return np.array(d["x"].tolist()), np.array(d["f_norm"].tolist())


def build(name):
    spec = RUNS[name]
    config.SUBGROUP_SCORES = spec["scores"]           # read by pipeline_from_instance
    inst = load_instance(spec["instance"])
    pipe = pipeline_from_instance(inst)
    df = load_logs(spec["logs"])
    seeds = sorted(df["seed"].unique())
    return inst, pipe, df, seeds


def decisions(pipe, df, seeds):
    wE = config.WEIGHTS["E"]
    out = {}
    for s in seeds:
        Xs, Ys = arrays(df, s, "SOBO_E")
        Xm, Ym = arrays(df, s, "MOBO")
        x_s = Xs[int(np.argmin([scalarize(y, wE) for y in Ys]))]
        x_m = Xm[compromise_index(Ym)]
        out[s] = (pipe.evaluate(x_s), pipe.evaluate(x_m), Xm, Ym)
    return out


def subgroup_scores(F, observed, labels):
    """Mean real score of each subgroup on each model (NaN where no member rated the model)."""
    c = labels.max() + 1
    G = np.full((c, F.shape[1]), np.nan)
    for r in range(c):
        m = labels == r
        cnt = observed[m].sum(axis=0)
        G[r] = np.where(cnt > 0, (F[m] * observed[m]).sum(axis=0) / np.maximum(cnt, 1), np.nan)
    return G


def tau(a, b):
    ok = ~(np.isnan(a) | np.isnan(b))
    return float(stats.kendalltau(a[ok], b[ok])[0])


def same(r1, r2):
    return adjusted_rand_score(r1.labels, r2.labels) > 1 - 1e-12, np.array_equal(r1.ranking, r2.ranking)


def main():
    out = config.RESULTS / "analysis"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    md = ["# Decisions (Section III-A-3 and the Discussion)"]

    # ---- main run: SOBO and MOBO decisions per seed --------------------------------------------
    inst, pipe, df, seeds = build("main")
    F, observed = inst["F"], inst["observed"].astype(bool)
    flat = np.nanmean(np.where(observed, F, np.nan), axis=0)          # mean real score per model
    dec = decisions(pipe, df, seeds)
    rows = []
    for s, (r_s, r_m, _, _) in dec.items():
        sizes = np.bincount(r_s.labels)
        order = np.argsort(-sizes)
        G = subgroup_scores(F, observed, r_s.labels)
        rows.append({
            "seed": s,
            "zeta_SOBO": r_s.x[0], "kappa_SOBO": r_s.x[1], "mu_SOBO": r_s.x[2],
            "zeta_MOBO": r_m.x[0], "kappa_MOBO": r_m.x[1], "mu_MOBO": r_m.x[2],
            "c_SOBO": r_s.c_star, "c_MOBO": r_m.c_star,
            "sizes_SOBO": "/".join(map(str, sizes[order])),
            "weights_SOBO": "/".join(f"{w:.3f}" for w in r_s.weights[order]),
            "tau_subgroups_SOBO": tau(G[order[0]], G[order[1]]) if r_s.c_star == 2 else np.nan,
            "tau_consensus_vs_flat_SOBO": tau(r_s.collective, flat),
            "ARI_SOBO_MOBO": adjusted_rand_score(r_s.labels, r_m.labels),
            "tau_SOBO_MOBO": tau(r_s.collective, r_m.collective),
            "same_decision": all(same(r_s, r_m)),
        })
    t = pd.DataFrame(rows)
    t.to_csv(out / "tables" / "decisions_per_seed.csv", index=False)
    rng = lambda c: f"{t[c].min():.3f} to {t[c].max():.3f}"
    md += ["\n## SOBO and MOBO decisions (20 seeds)",
           f"- c*: SOBO {sorted(t['c_SOBO'].unique().tolist())}, MOBO {sorted(t['c_MOBO'].unique().tolist())}",
           f"- Same partition and ranking: {int(t['same_decision'].sum())} of {len(t)} seeds; "
           f"minimum ARI {t['ARI_SOBO_MOBO'].min():.2f}, minimum Kendall tau {t['tau_SOBO_MOBO'].min():.3f}",
           f"- zeta: SOBO median {t['zeta_SOBO'].median():.3f} ({rng('zeta_SOBO')}), "
           f"MOBO median {t['zeta_MOBO'].median():.3f} ({rng('zeta_MOBO')})",
           f"- kappa: SOBO {t['kappa_SOBO'].min():.2f}-{t['kappa_SOBO'].max():.2f}, "
           f"MOBO {t['kappa_MOBO'].min():.2f}-{t['kappa_MOBO'].max():.2f}",
           f"- mu: SOBO {t['mu_SOBO'].min():.2f}-{t['mu_SOBO'].max():.2f}, "
           f"MOBO {t['mu_MOBO'].min():.2f}-{t['mu_MOBO'].max():.2f}",
           f"- SOBO subgroup sizes: {sorted(t['sizes_SOBO'].unique().tolist())}; "
           f"weights: {sorted(t['weights_SOBO'].unique().tolist())}",
           f"- Kendall tau between the two SOBO subgroups (real votes): {rng('tau_subgroups_SOBO')}",
           f"- Kendall tau between the SOBO consensus and the flat ranking: {rng('tau_consensus_vs_flat_SOBO')}"]

    # ---- effect of mu at the selected zeta ------------------------------------------------------
    zeta_star = float(t["zeta_SOBO"].median())
    md.append(f"\n## Effect of mu at zeta = {zeta_star:.3f}, kappa = 0.5")
    for mu in (0.0, 1.0):
        r = pipe.evaluate((zeta_star, 0.5, mu))
        order = np.argsort(-np.bincount(r.labels))
        md.append(f"- mu = {mu:.0f}: c* = {r.c_star}, weights {'/'.join(f'{w:.3f}' for w in r.weights[order])}, "
                  f"f2 = {r.f_raw[1]:.5f}")

    # ---- zeta band (profile from 05_diagnostics.py) ---------------------------------------------
    prof = pd.read_csv(out / "tables" / "diag_zeta_profile.csv")
    good = prof[prof["g_E"] < 0.15]
    best = prof.loc[prof["g_E"].idxmin()]
    at = lambda z: float(prof.loc[(prof["zeta"] - z).abs().idxmin(), "g_E"])
    md += [f"\n## Zeta profile (kappa = mu = 0.5)",
           f"- g_E < 0.15 only for zeta in [{good['zeta'].min():.2f}, {good['zeta'].max():.2f}]",
           f"- lowest g_E = {best['g_E']:.3f} at zeta = {best['zeta']:.2f}; g_E(0.40) = {at(0.40):.3f}, "
           f"g_E(0.39) = {at(0.39):.3f}"]

    # ---- MOBO Pareto sets, main and imputed runs ------------------------------------------------
    md.append("\n## Configurations in the MOBO Pareto sets compared with the SOBO decision of the same seed")
    prow = []
    for name in ("main", "imputed"):
        _, pipe_n, df_n, seeds_n = build(name)
        dec_n = decisions(pipe_n, df_n, seeds_n)
        for s, (r_s, _, Xm, Ym) in dec_n.items():
            for x in Xm[nondominated_mask(Ym)]:
                r = pipe_n.evaluate(x)
                same_part, same_rank = same(r_s, r) if r.c_star > 0 else (False, False)
                prow.append({"run": name, "seed": s, "x": np.round(x, 4).tolist(), "c_star": r.c_star,
                             "same_partition": same_part, "same_ranking": same_rank,
                             "tau_vs_SOBO": tau(r_s.collective, r.collective) if r.c_star > 0 else np.nan})
    p = pd.DataFrame(prow)
    p.to_csv(out / "tables" / "decisions_pareto_points.csv", index=False)
    for name, g in p.groupby("run", sort=False):
        md.append(f"- {name}: {len(g)} configurations; different partition {100 * (~g['same_partition']).mean():.0f}% "
                  f"(c* = 3 in {int((g['c_star'] == 3).sum())}); different ranking {100 * (~g['same_ranking']).mean():.0f}% "
                  f"(minimum Kendall tau {g['tau_vs_SOBO'].min():.2f})")
    config.SUBGROUP_SCORES = RUNS["main"]["scores"]

    (out / "decisions.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
