"""Step 3 (Section 2.6): two metrics per research question and the statistical tests.

Outputs results/summary.md, results/tables/*.csv and results/figures/*.png.

Usage:
    python 03_analyze.py            # full experiment logs
    python 03_analyze.py --quick    # smoke-test logs
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score

import config
from src.data import load_instance
from src.metrics import a12, a12_magnitude, anytime_best_g, anytime_hv, compromise_index, holm, hypervolume
from src.plotstyle import TEXT_WIDTH, use_paper_style
from src.scofcm import pipeline_from_instance, scalarize

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

use_paper_style()

MIN_N_FOR_TESTS = 5


def md_table(df: pd.DataFrame) -> str:
    """Markdown table without extra dependencies."""
    def cell(v):
        return f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def load_logs(log_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(log_dir.glob("seed*_*.jsonl")):
        with open(path) as fh:
            rows.extend(json.loads(line) for line in fh)
    if not rows:
        raise SystemExit(f"No logs found in {log_dir}. Run 02_run_experiment.py first.")
    return pd.DataFrame(rows)


def arrays(df, seed, run):
    d = df[(df["seed"] == seed) & (df["run"] == run)].sort_values("iter")
    return np.array(d["x"].tolist()), np.array(d["f_norm"].tolist()), d


def wilcoxon_p(x, y=None):
    diff = np.asarray(x, float) - (0 if y is None else np.asarray(y, float))
    if len(diff) < MIN_N_FOR_TESTS or np.allclose(diff, 0):
        return np.nan
    return float(stats.wilcoxon(diff).pvalue)


def rq1(df, seeds, out):
    rows, curves = [], {"SOBO_E": {"hv": [], "g": []}, "MOBO": {"hv": [], "g": []}}
    wE = config.WEIGHTS["E"]
    for s in seeds:
        rec = {"seed": s}
        for run in ("SOBO_E", "MOBO"):
            _, Y, _ = arrays(df, s, run)
            rec[f"HV_{run}"] = hypervolume(Y)
            rec[f"bestg_{run}"] = float(min(scalarize(y, wE) for y in Y))
            curves[run]["hv"].append(anytime_hv(Y))
            curves[run]["g"].append(anytime_best_g(Y, wE))
        rows.append(rec)
    t = pd.DataFrame(rows)
    t.to_csv(out / "tables" / "rq1_per_seed.csv", index=False)

    p = [wilcoxon_p(t["HV_MOBO"], t["HV_SOBO_E"]), wilcoxon_p(t["bestg_MOBO"], t["bestg_SOBO_E"])]
    p_adj = holm(np.nan_to_num(p, nan=1.0))
    a_hv = a12(t["HV_MOBO"], t["HV_SOBO_E"])                 # P(MOBO has larger HV)
    a_g = a12(t["bestg_SOBO_E"], t["bestg_MOBO"])           # P(MOBO has smaller g)
    summary = pd.DataFrame([
        {"metric": "Hypervolume (higher is better)",
         "SOBO median": t["HV_SOBO_E"].median(), "MOBO median": t["HV_MOBO"].median(),
         "p (Wilcoxon)": p[0], "p (Holm)": p_adj[0], "A12 (MOBO better)": a_hv,
         "effect": a12_magnitude(a_hv)},
        {"metric": "Best weighted sum g_wE (lower is better)",
         "SOBO median": t["bestg_SOBO_E"].median(), "MOBO median": t["bestg_MOBO"].median(),
         "p (Wilcoxon)": p[1], "p (Holm)": p_adj[1], "A12 (MOBO better)": a_g,
         "effect": a12_magnitude(a_g)},
    ])
    summary.to_csv(out / "tables" / "rq1_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 1.9))
    for ax, key, ylabel, tag in ((axes[0], "hv", "Hypervolume", "(a)"),
                                 (axes[1], "g", "Best weighted sum $g_{w_E}$", "(b)")):
        for run, label in (("SOBO_E", "SOBO"), ("MOBO", "MOBO")):
            c = np.array(curves[run][key])
            x = np.arange(1, c.shape[1] + 1)
            med, lo, hi = np.median(c, 0), np.percentile(c, 25, 0), np.percentile(c, 75, 0)
            ax.plot(x, med, label=label)
            ax.fill_between(x, lo, hi, alpha=0.2)
        ax.axvline(config.N_INIT, color="grey", lw=0.8, ls=":")
        ax.set_xlabel(f"Number of evaluations\n{tag}")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
    fig.tight_layout(pad=0.3, w_pad=1.5)
    fig.savefig(out / "figures" / "rq1_anytime.png")
    plt.close(fig)
    return summary


def rq2(df, seeds, out):
    init = df[(df["run"] == "SOBO_E") & (df["phase"] == "init") & (df["seed"].isin(seeds))]
    Y0 = np.array(init["f_norm"].tolist())
    names = ["f1 validity", "f2 consensus cost", "f3 convergence"]
    rows, pvals = [], []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        rho, p = stats.spearmanr(Y0[:, i], Y0[:, j])
        rows.append({"test": f"Spearman rho: {names[i]} vs {names[j]}", "n": len(Y0),
                     "statistic": rho, "p": p})
        pvals.append(p)

    delta_rows = []
    for s in seeds:
        _, Ym, dm = arrays(df, s, "MOBO")
        rec = {"seed": s,
               "time_MOBO_s": float((dm.loc[dm["phase"] == "bo", "t_eval"] + dm.loc[dm["phase"] == "bo", "t_acq"]).sum())}
        t_sobo = 0.0
        for wkey in ("E", "V", "C", "T"):
            _, Ys, ds = arrays(df, s, f"SOBO_{wkey}")
            t_sobo += float((ds.loc[ds["phase"] == "bo", "t_eval"] + ds.loc[ds["phase"] == "bo", "t_acq"]).sum())
            if wkey == "E":
                continue
            w = config.WEIGHTS[wkey]
            rec[f"delta_{wkey}"] = (min(scalarize(y, w) for y in Ym) - min(scalarize(y, w) for y in Ys))
        rec["time_SOBO_4runs_s"] = t_sobo
        delta_rows.append(rec)
    d = pd.DataFrame(delta_rows)
    d.to_csv(out / "tables" / "rq2_delta_per_seed.csv", index=False)
    for wkey in ("V", "C", "T"):
        p = wilcoxon_p(d[f"delta_{wkey}"])
        rows.append({"test": f"Weight-coverage gap Delta_w{wkey} (median)", "n": len(d),
                     "statistic": d[f"delta_{wkey}"].median(), "p": p})
        pvals.append(p)
    adj = holm(np.nan_to_num(pvals, nan=1.0))
    for r, pa in zip(rows, adj):
        r["p (Holm)"] = pa
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "tables" / "rq2_summary.csv", index=False)
    cost = pd.DataFrame([{
        "evaluations to cover 4 weightings": f"SOBO {4 * config.BUDGET} vs MOBO {config.BUDGET}",
        "median time SOBO (4 runs, s)": d["time_SOBO_4runs_s"].median(),
        "median time MOBO (1 run, s)": d["time_MOBO_s"].median(),
    }])
    cost.to_csv(out / "tables" / "rq2_cost.csv", index=False)
    return summary, cost


def rq3(df, seeds, out):
    pipe = pipeline_from_instance(load_instance())
    wE = config.WEIGHTS["E"]
    rows = []
    for s in seeds:
        Xs, Ys, _ = arrays(df, s, "SOBO_E")
        Xm, Ym, _ = arrays(df, s, "MOBO")
        x_s = Xs[int(np.argmin([scalarize(y, wE) for y in Ys]))]
        x_m = Xm[compromise_index(Ym)]
        r_s, r_m = pipe.evaluate(x_s), pipe.evaluate(x_m)
        ari = adjusted_rand_score(r_s.labels, r_m.labels) if r_s.c_star > 0 and r_m.c_star > 0 else np.nan
        tau = stats.kendalltau(r_s.collective, r_m.collective)[0] if r_s.c_star > 0 and r_m.c_star > 0 else np.nan
        rows.append({"seed": s, "x_SOBO": np.round(x_s, 4).tolist(), "x_MOBO": np.round(x_m, 4).tolist(),
                     "c*_SOBO": r_s.c_star, "c*_MOBO": r_m.c_star, "ARI": ari, "Kendall_tau": tau})
    t = pd.DataFrame(rows)
    t.to_csv(out / "tables" / "rq3_per_seed.csv", index=False)
    summary = pd.DataFrame([
        {"metric": m, "median": t[m].median(), "Q1": t[m].quantile(0.25), "Q3": t[m].quantile(0.75)}
        for m in ("ARI", "Kendall_tau")
    ])
    summary.to_csv(out / "tables" / "rq3_summary.csv", index=False)
    return summary, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    log_dir = config.RESULTS / "logs_quick" if args.quick else config.LOGS
    out = config.RESULTS / ("analysis_quick" if args.quick else "analysis")
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    df = load_logs(log_dir)
    complete = df.groupby("seed")["run"].nunique()
    seeds = sorted(complete[complete == len(config.RUNS)].index)
    if not seeds:
        raise SystemExit("No seed has all runs finished yet.")
    note = "" if len(seeds) >= MIN_N_FOR_TESTS else (
        f"\n> Only {len(seeds)} seed(s): statistical tests are skipped (p = NaN).\n")

    n_fb = int((df["phase"] == "bo_fallback").sum())
    note += f"\n> GP-fit fallbacks (quasi-random points): {n_fb} of {int((df['phase'] != 'init').sum())} BO iterations.\n"

    s1 = rq1(df, seeds, out)
    s2, cost = rq2(df, seeds, out)
    s3, _ = rq3(df, seeds, out)
    md = [f"# Results summary ({len(seeds)} seeds){note}",
          "## RQ1: solution quality", md_table(s1), "",
          "![anytime](figures/rq1_anytime.png)", "",
          "## RQ2: trade-offs", md_table(s2), "", md_table(cost), "",
          "## RQ3: decisions (SOBO vs MOBO)", md_table(s3), ""]
    (out / "summary.md").write_text("\n\n".join(md))
    print("\n\n".join(md))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
