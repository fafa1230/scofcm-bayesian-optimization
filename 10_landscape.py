"""Step 9 (Sections 3.1.3 and 3.2, Figure 3): shape of the objective landscape.

  - The equally weighted sum along zeta in steps of 0.001 (kappa = mu = 0.5), with the configurations
    selected by SOBO and the baseline setting marked; the band of zeta around the selections in which
    SCO-FCM keeps two subgroups, and the distance of the selected values to its edges (Figure 3).
  - Variance-based (Sobol) sensitivity indices of g_wE and of the three normalized objectives with
    respect to (zeta, kappa, mu): Saltelli design with N = 512 base points, first-order indices with
    the Saltelli (2010) estimator, total indices with the Jansen estimator, bootstrap intervals.

Outputs results/analysis/landscape.md, tables/zeta_profile_fine.csv, tables/sobol_indices.csv and
figures/zeta_profile_fine.png (Figure 3 of the manuscript).

Usage:
    python 10_landscape.py
"""
import json

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import qmc

import config
from src.data import load_instance
from src.plotstyle import use_paper_style
from src.scofcm import pipeline_from_instance, scalarize

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASELINE = (0.4, 0.6, 0.5)
FIG_WIDTH = 6.1                      # inches: IJADIS text width
N_SOBOL = 512
N_BOOT = 1_000
NAMES = ("zeta", "kappa", "mu")


def load_logs(log_dir):
    rows = []
    for path in sorted(log_dir.glob("seed*_*.jsonl")):
        with open(path) as fh:
            rows.extend(json.loads(line) for line in fh)
    return pd.DataFrame(rows)


def outputs(pipe, X, wE):
    """g_wE and the three normalized objectives for every row of X."""
    Y = np.array([pipe.evaluate(x).f_norm for x in X])
    return np.column_stack([[scalarize(y, wE) for y in Y], Y])


def sobol_indices(fA, fB, fAB):
    var = np.var(np.concatenate([fA, fB]), ddof=1)
    first = np.array([np.mean(fB * (fAB[i] - fA)) / var for i in range(len(fAB))])
    total = np.array([0.5 * np.mean((fA - fAB[i]) ** 2) / var for i in range(len(fAB))])
    return first, total


def main():
    out = config.RESULTS / "analysis"
    for sub in ("tables", "figures"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    wE = config.WEIGHTS["E"]
    pipe = pipeline_from_instance(load_instance(config.DATA_PROCESSED / "instance.npz"))
    md = ["# Objective landscape"]

    # ---- fine zeta profile -----------------------------------------------------------------
    zs = np.round(np.linspace(0, 1, 1001), 3)
    prof = []
    for z in zs:
        r = pipe.evaluate((z, 0.5, 0.5))
        prof.append({"zeta": z, "c_star": r.c_star, "f1": r.f_raw[0], "f2": r.f_raw[1], "f3": r.f_raw[2],
                     "g_E": scalarize(r.f_norm, wE)})
    prof = pd.DataFrame(prof)
    prof.to_csv(out / "tables" / "zeta_profile_fine.csv", index=False)

    df = load_logs(config.LOGS)
    sel = []
    for s in sorted(df["seed"].unique()):
        d = df[(df["seed"] == s) & (df["run"] == "SOBO_E")]
        g = [scalarize(y, wE) for y in d["f_norm"]]
        sel.append((d["x"].iloc[int(np.argmin(g))][0], min(g)))
    sel = np.array(sel)
    base = pipe.evaluate(BASELINE)
    g_base = scalarize(base.f_norm, wE)

    two = prof["c_star"] == 2
    # contiguous run of two-subgroup values that contains the SOBO selections
    i = int(np.argmin(np.abs(prof["zeta"].to_numpy() - np.median(sel[:, 0]))))
    lo, hi = i, i
    while lo > 0 and two.iloc[lo - 1]:
        lo -= 1
    while hi < len(prof) - 1 and two.iloc[hi + 1]:
        hi += 1
    lo_edge, hi_edge = prof["zeta"].iloc[lo], prof["zeta"].iloc[hi]
    gap_lo, gap_hi = sel[:, 0].min() - lo_edge, hi_edge - sel[:, 0].max()
    in_band = prof[(prof["zeta"] >= lo_edge) & (prof["zeta"] <= hi_edge)]
    near = prof[np.min(np.abs(prof["zeta"].to_numpy()[:, None] - sel[:, 0][None, :]), axis=1) <= 0.005]
    md += ["\n## Zeta profile (step 0.001, kappa = mu = 0.5)",
           f"- Band with the selected two-subgroup partition (contiguous, around the SOBO selections): zeta in [{lo_edge:.3f}, {hi_edge:.3f}]; "
           f"{int(two.sum())} of {len(prof)} grid values give two subgroups overall",
           f"- SOBO selections: zeta {sel[:, 0].min():.3f}-{sel[:, 0].max():.3f}, g {sel[:, 1].min():.4f}-{sel[:, 1].max():.4f}; "
           f"distance to the band edges at least {gap_lo:.3f} (lower) and {gap_hi:.3f} (upper)",
           f"- Baseline setting: g = {g_base:.4f}, f3 = {base.f_raw[2]:.0f}; zeta = 0.4 lies {0.4 - lo_edge:.3f} above the lower edge",
           f"- Inside the band: g from {in_band['g_E'].min():.4f} to {in_band['g_E'].max():.4f}, "
           f"f3 from {in_band['f3'].min():.0f} to {in_band['f3'].max():.0f} iterations; distinct values of "
           f"f1: {in_band['f1'].round(6).nunique()}, of f2: {in_band['f2'].round(6).nunique()}",
           f"- Within 0.005 of a SOBO selection: g from {near['g_E'].min():.4f} to {near['g_E'].max():.4f}, "
           f"f3 from {near['f3'].min():.0f} to {near['f3'].max():.0f}"]

    use_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 1.75), gridspec_kw={"width_ratios": [1, 1.25]})
    for ax, (a, b), tag in ((axes[0], (0, 1), "(a)"), (axes[1], (0.38, 0.53), "(b)")):
        p = prof[(prof["zeta"] >= a) & (prof["zeta"] <= b)]
        ax.plot(p["zeta"], p["g_E"], lw=0.6, color="C0")
        ax.axvspan(lo_edge, hi_edge, color="0.85", lw=0, zorder=0, label="band with the selected partition")
        ax.plot([0.4], [g_base], ls="none", marker="s", ms=3.5, color="0.3", label="baseline setting")
        if tag == "(b)":
            ax.plot(sel[:, 0], sel[:, 1], ls="none", marker="o", ms=3, mfc="none", color="C3",
                    label="selected by SOBO (20 seeds)")
        ax.set_xlim(a, b)
        ax.set_xlabel(f"$\\zeta$ (with $\\kappa$ = $\\mu$ = 0.5)\n{tag}")
        ax.set_ylabel("Weighted sum $g_{w_E}$")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, handletextpad=0.4, columnspacing=1.2,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.3, w_pad=1.2, rect=(0, 0, 1, 0.88))
    fig.savefig(out / "figures" / "zeta_profile_fine.png")
    plt.close(fig)

    # ---- Sobol indices ---------------------------------------------------------------------
    base_pts = qmc.Sobol(d=2 * config.DIM, scramble=True, seed=0).random(N_SOBOL)
    A, B = base_pts[:, :config.DIM], base_pts[:, config.DIM:]
    ABs = []
    for i in range(config.DIM):
        AB = A.copy()
        AB[:, i] = B[:, i]
        ABs.append(AB)
    fA, fB = outputs(pipe, A, wE), outputs(pipe, B, wE)
    fAB = [outputs(pipe, AB, wE) for AB in ABs]
    rng = np.random.default_rng(0)
    rows = []
    for j, target in enumerate(("g_wE", "f1_norm", "f2_norm", "f3_norm")):
        S1, ST = sobol_indices(fA[:, j], fB[:, j], [f[:, j] for f in fAB])
        boot = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, N_SOBOL, N_SOBOL)
            boot.append(np.concatenate(sobol_indices(fA[idx, j], fB[idx, j], [f[idx, j] for f in fAB])))
        lo, hi = np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)
        for i, name in enumerate(NAMES):
            rows.append({"output": target, "parameter": name, "S1": S1[i], "S1_lo": lo[i], "S1_hi": hi[i],
                         "ST": ST[i], "ST_lo": lo[3 + i], "ST_hi": hi[3 + i]})
    sob = pd.DataFrame(rows)
    sob.to_csv(out / "tables" / "sobol_indices.csv", index=False)
    md.append(f"\n## Sobol indices ({N_SOBOL * (config.DIM + 2)} evaluations; 95% bootstrap intervals)")
    for target, g in sob.groupby("output", sort=False):
        md.append(f"- {target}: " + "; ".join(
            f"{r.parameter} S1 = {r.S1:.2f} [{r.S1_lo:.2f}, {r.S1_hi:.2f}], ST = {r.ST:.2f} [{r.ST_lo:.2f}, {r.ST_hi:.2f}]"
            for r in g.itertuples()))
    (out / "landscape.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
