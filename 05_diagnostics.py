"""Step 4 (supporting the Discussion): why SOBO and MOBO end up at the same decisions.

Outputs results/analysis/diagnostics.md, tables/diag_*.csv and figures/diag_zeta_profile.png:
  - size of the Pareto front found by MOBO and number of distinct objective vectors;
  - objectives over a (kappa, mu) grid at the zeta most often selected;
  - objectives along zeta with kappa = mu = 0.5;
  - acquisition vs evaluation time and the break-even evaluation cost for MOBO.

Usage:
    python 05_diagnostics.py
"""
import glob
import json

import matplotlib
import numpy as np
import pandas as pd

import config
from src.data import load_instance
from src.metrics import nondominated_mask
from src.plotstyle import use_paper_style
from src.scofcm import pipeline_from_instance, scalarize

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

use_paper_style()


def main():
    out = config.RESULTS / "analysis"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([json.loads(line) for f in sorted(glob.glob(str(config.LOGS / "*.jsonl")))
                       for line in open(f)])
    wE = config.WEIGHTS["E"]
    md = ["# Diagnostics"]

    # Pareto front size and distinct objective vectors
    nd = [int(nondominated_mask(np.array(d.sort_values("iter")["f_norm"].tolist())).sum())
          for _, d in df[df["run"] == "MOBO"].groupby("seed")]
    F = np.array(df["f_norm"].tolist())
    n_distinct = len(np.unique(np.round(F, 10), axis=0))
    md += [f"- Non-dominated points per MOBO run: median {np.median(nd):.0f} (range {min(nd)}-{max(nd)}).",
           f"- Distinct objective vectors: {n_distinct} of {len(F)} evaluations."]

    # Selected configurations of the main SOBO runs
    best = []
    for s, d in df[df["run"] == "SOBO_E"].groupby("seed"):
        d = d.sort_values("iter")
        g = [scalarize(f, wE) for f in d["f_norm"]]
        best.append(d["x"].iloc[int(np.argmin(g))])
    best = np.array(best)
    zeta_star = float(np.median(best[:, 0]))
    md += [f"- Selected zeta (SOBO, 20 seeds): median {zeta_star:.3f}, range {best[:, 0].min():.3f}-{best[:, 0].max():.3f}; "
           f"kappa range {best[:, 1].min():.2f}-{best[:, 1].max():.2f}; mu range {best[:, 2].min():.2f}-{best[:, 2].max():.2f}."]

    pipe = pipeline_from_instance(load_instance())

    # (kappa, mu) grid at zeta_star
    rows = []
    for k in np.linspace(0, 1, 11):
        for m in np.linspace(0, 1, 11):
            r = pipe.evaluate((zeta_star, k, m))
            rows.append({"kappa": k, "mu": m, "c_star": r.c_star, "f1n": r.f_norm[0], "f2n": r.f_norm[1],
                         "f3n": r.f_norm[2], "g_E": scalarize(r.f_norm, wE)})
    grid = pd.DataFrame(rows)
    grid.to_csv(out / "tables" / "diag_kappa_mu_grid.csv", index=False)
    by_c = grid.groupby("c_star")[["f1n", "f2n", "f3n", "g_E"]].agg(["min", "max"]).round(4)
    md += [f"\n## Objectives over 121 (kappa, mu) pairs at zeta = {zeta_star:.3f}",
           "Rows per c*: " + ", ".join(f"c*={c}: {n}" for c, n in grid["c_star"].value_counts().sort_index().items()),
           "```", by_c.to_string(), "```"]

    # zeta profile
    prof = []
    for z in np.linspace(0, 1, 101):
        r = pipe.evaluate((z, 0.5, 0.5))
        prof.append({"zeta": z, "c_star": r.c_star, "g_E": scalarize(r.f_norm, wE),
                     "f1n": r.f_norm[0], "f2n": r.f_norm[1], "f3n": r.f_norm[2]})
    prof = pd.DataFrame(prof)
    prof.to_csv(out / "tables" / "diag_zeta_profile.csv", index=False)
    fig, ax = plt.subplots(figsize=(3.4, 1.62))
    ax.plot(prof["zeta"], prof["g_E"])
    ax.axvline(0.4, color="grey", ls=":", lw=1, label="baseline OFAT, $\\zeta$ = 0.4")
    ax.axvline(zeta_star, color="C3", ls="--", lw=1, label=f"selected by SOBO/MOBO, $\\zeta$ ≈ {zeta_star:.2f}")
    ax.set_xlabel("$\\zeta$ (with $\\kappa$ = $\\mu$ = 0.5)")
    ax.set_ylabel("Weighted sum $g_{w_E}$")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              handlelength=1.4, columnspacing=0.8, handletextpad=0.4, borderaxespad=0.1)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "figures" / "diag_zeta_profile.png")
    plt.close(fig)

    # Time: acquisition vs evaluation, break-even
    bo = df[df["phase"] == "bo"]
    n_bo = config.BUDGET - config.N_INIT
    acq_m = bo[bo["run"] == "MOBO"].groupby("seed")["t_acq"].sum().median()
    acq_s = bo[bo["mode"] == "SOBO"].groupby("seed")["t_acq"].sum().median()
    breakeven = (acq_m - acq_s) / (4 * n_bo - n_bo)
    md += ["\n## Cost",
           f"- Median acquisition time per iteration: SOBO {bo[bo['mode'] == 'SOBO']['t_acq'].median():.2f} s, "
           f"MOBO {bo[bo['mode'] == 'MOBO']['t_acq'].median():.2f} s; median evaluation of Phi "
           f"{bo['t_eval'].median() * 1000:.0f} ms.",
           f"- Median acquisition time per seed: MOBO (1 run) {acq_m:.1f} s, SOBO (4 runs) {acq_s:.1f} s.",
           f"- Covering the four weightings costs MOBO {n_bo} BO evaluations vs {4 * n_bo} for SOBO; "
           f"MOBO is faster overall once one evaluation of Phi takes more than {breakeven:.2f} s."]
    (out / "diagnostics.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
