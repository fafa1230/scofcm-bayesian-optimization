"""Step 13 (revision; Sections 2.2 and 3.1.4, Table 6): robustness of the rater relations and profiles.

Four variants address how the inputs of SCO-FCM are built from anonymous votes:
  bothperiods  only the raters with votes in both periods, so no period is filled in (482 raters);
  emptyperiod  a rater without votes in a period has an empty winner set there (no relation)
               instead of the full observation window;
  coexposure   Jaccard similarity over the models both raters encountered in the period, so that
               different exposure to models is not read as different preferences;
  pds          no imputation: rater distances, FCM distances, and centres use the observed scores
               only (partial distance strategy of Hathaway and Bezdek, 2001).
Each variant gets its own pilot bounds and five paired replications (seeds 1-5) of SOBO (equal
weights) and MOBO, as in 13_sensitivity.py. The summary covers these variants together with the main
instance and the nine variants of 13_sensitivity.py, and adds two comparisons: the agreement of each
variant's SOBO decision with the main decision of the same seed (ARI on the common raters, Kendall's
tau of the consensus scores), and Kendall's tau between the consensus and the flat ranking.

Outputs data/raw/sensitivity/<variant>.npz, results/sensitivity/logs/<variant>/*.jsonl,
results/sensitivity/revision_summary.md, revision_variants.csv and revision_replications.csv.

Usage:
    python 14_revision_variants.py --jobs 6     # build, run, and summarize
    python 14_revision_variants.py --summary    # summarize existing logs only
"""
import warnings

warnings.filterwarnings("ignore", message=".*torch.jit.script.*", category=FutureWarning)

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

import config  # noqa: E402

INST_DIR = config.DATA_RAW / "sensitivity"
OUT = config.RESULTS / "sensitivity"
SEEDS = (1, 2, 3, 4, 5)
RUNS = (("SOBO_E", "SOBO", "E"), ("MOBO", "MOBO", None))
VARIANTS = {
    "bothperiods": {"build": {"both_periods": True}},
    "emptyperiod": {"build": {"missing_period": "empty"}},
    "coexposure": {"build": {"similarity": "coexposure"}},
    "pds": {"build": {}, "missing": "pds"},
}
EARLIER = ("votes5", "votes20", "bothbad0", "decisive", "sub80_1", "sub80_2", "sub80_3", "sub80_4", "sub80_5")


def pipeline_for(inst):
    from src.scofcm import pipeline_from_instance

    missing = str(inst["missing"]) if "missing" in inst else "impute"
    return pipeline_from_instance(inst, missing=missing)


# ---------------------------------------------------------------------------
# 1. Instances
# ---------------------------------------------------------------------------
def build_variant(name, battles):
    from src.data import build_instance, pilot_bounds
    from src.scofcm import SCOFCMPipeline

    spec = VARIANTS[name]
    inst = build_instance(battles, **spec["build"])
    missing = spec.get("missing", "impute")
    pipe = SCOFCMPipeline(inst["F"], inst["alpha"], inst["beta"], inst["pr_h"], inst["pr_c"],
                          observed=inst["observed"], missing=missing)
    lo, hi, pilot = pilot_bounds(pipe)
    inst.update(bounds_lo=lo, bounds_hi=hi, pilot_f_raw=pilot, missing=np.array(missing))
    inst.pop("stats")
    INST_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(INST_DIR / f"{name}.npz", **inst)


def _build_task(name):
    from src.data import load_battles

    t0 = time.perf_counter()
    build_variant(name, load_battles())
    return f"built {name} in {time.perf_counter() - t0:.0f} s"


def _init_worker(threads):
    import torch

    torch.set_num_threads(threads)


def build_all(jobs):
    todo = [v for v in VARIANTS if not (INST_DIR / f"{v}.npz").exists()]
    if not todo:
        return
    threads = max(1, (mp.cpu_count() or 2) // jobs)
    with mp.get_context("spawn").Pool(min(jobs, len(todo)), initializer=_init_worker, initargs=(threads,)) as pool:
        for msg in pool.imap_unordered(_build_task, todo):
            print(msg, flush=True)


# ---------------------------------------------------------------------------
# 2. Paired SOBO / MOBO runs
# ---------------------------------------------------------------------------
WORKER = {}


def _pipeline(name):
    from src.data import load_instance

    if name not in WORKER:
        WORKER[name] = (pipeline_for(load_instance(INST_DIR / f"{name}.npz")), {})
    return WORKER[name]


def run_task(task):
    from src.optimizers import run_bo, sobol_design

    name, seed, run, mode, wkey = task
    path = OUT / "logs" / name / f"seed{seed:02d}_{run}.jsonl"
    if path.exists() and sum(1 for _ in open(path)) >= config.BUDGET:
        return f"{name} seed {seed} {run}: done before"
    pipe, cache = _pipeline(name)

    def evaluate(x):
        key = tuple(np.round(np.asarray(x, dtype=float), 12))
        if key not in cache:
            cache[key] = pipe.evaluate(key)
        return cache[key]

    t0 = time.perf_counter()
    records = run_bo(evaluate, mode, sobol_design(seed), seed,
                     weights=config.WEIGHTS[wkey] if wkey else None, budget=config.BUDGET)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps({**rec, "seed": seed, "run": run}) + "\n")
    return f"{name} seed {seed} {run}: {time.perf_counter() - t0:.0f} s"


def run_all(jobs):
    tasks = [(v, s, run, mode, w) for v in VARIANTS for s in SEEDS for run, mode, w in RUNS]
    tasks.sort(key=lambda t: t[2] != "MOBO")
    threads = max(1, (mp.cpu_count() or 2) // jobs)
    with mp.get_context("spawn").Pool(jobs, initializer=_init_worker, initargs=(threads,)) as pool:
        for msg in pool.imap_unordered(run_task, tasks):
            print(msg, flush=True)


# ---------------------------------------------------------------------------
# 3. Summary
# ---------------------------------------------------------------------------
def _decisions(pipe, log_dir, seed):
    """SOBO and MOBO decisions of one replication, with HV and best g of both runs."""
    from src.metrics import compromise_index, hypervolume
    from src.scofcm import scalarize

    out = {}
    for run, label in (("SOBO_E", "SOBO"), ("MOBO", "MOBO")):
        with open(log_dir / f"seed{seed:02d}_{run}.jsonl") as fh:
            recs = [json.loads(line) for line in fh]
        X = np.array([r["x"] for r in recs])
        Y = np.array([r["f_norm"] for r in recs])
        gs = np.array([scalarize(y, config.WEIGHTS["E"]) for y in Y])
        x = X[int(np.argmin(gs))] if label == "SOBO" else X[compromise_index(Y)]
        out[label] = (pipe.evaluate(x), hypervolume(Y), float(gs.min()))
    return out


def _pilot_rho(inst, pipe):
    lo, hi = inst["bounds_lo"], inst["bounds_hi"]
    if "pilot_f_raw" in inst:
        P = inst["pilot_f_raw"]
    else:                                 # main instance: the same 512-point pilot sample (seed 0)
        from src.data import pilot_bounds
        P = pilot_bounds(pipe)[2]
    Pn = np.column_stack([(hi[0] - P[:, 0]) / (hi[0] - lo[0]), (P[:, 1] - lo[1]) / (hi[1] - lo[1]),
                          (P[:, 2] - lo[2]) / (hi[2] - lo[2])])
    return float(stats.spearmanr(Pn[:, 0], Pn[:, 1])[0])


def summarize():
    from src.data import load_battles, load_instance

    battles = load_battles()
    tie_vote = ~battles["winner"].isin(["model_a", "model_b"])
    main = load_instance(config.DATA_PROCESSED / "instance.npz")
    main_pipe = pipeline_for(main)
    main_dec = {s: _decisions(main_pipe, config.LOGS, s) for s in config.SEEDS}

    sources = [("main", config.DATA_PROCESSED / "instance.npz", config.LOGS, config.SEEDS)]
    sources += [(v, INST_DIR / f"{v}.npz", OUT / "logs" / v, SEEDS) for v in EARLIER + tuple(VARIANTS)]
    rows, reps = [], []
    for name, inst_path, log_dir, seeds in sources:
        inst = main if name == "main" else load_instance(inst_path)
        pipe = main_pipe if name == "main" else pipeline_for(inst)
        rho = _pilot_rho(inst, pipe)
        flat = np.nanmean(np.where(inst["observed"], inst["F"], np.nan), axis=0)
        models = list(inst["models"])
        common_m = [m for m in models if m in set(main["models"])]
        im = [models.index(m) for m in common_m]
        jm = [list(main["models"]).index(m) for m in common_m]
        rpos = {r: i for i, r in enumerate(main["raters"])}
        for s in seeds:
            dec = main_dec[s] if name == "main" else _decisions(pipe, log_dir, s)
            (rs, hv_s, g_s), (rm, hv_m, g_m) = dec["SOBO"], dec["MOBO"]
            ref = main_dec[s]["SOBO"][0]
            common_r = [i for i, r in enumerate(inst["raters"]) if r in rpos]
            small = int(np.argmin(np.bincount(rs.labels)))
            members = set(inst["raters"][rs.labels == small])
            reps.append({
                "variant": name, "seed": s, "raters": len(inst["raters"]), "rho_f1_f2": rho,
                "HV_SOBO": hv_s, "HV_MOBO": hv_m, "g_SOBO": g_s, "g_MOBO": g_m,
                "ARI_SOBO_MOBO": adjusted_rand_score(rs.labels, rm.labels),
                "tau_SOBO_MOBO": float(stats.kendalltau(rs.collective, rm.collective)[0]),
                "same_decision": bool(adjusted_rand_score(rs.labels, rm.labels) > 1 - 1e-12
                                      and np.array_equal(rs.ranking, rm.ranking)),
                "c_SOBO": rs.c_star, "c_MOBO": rm.c_star,
                "ARI_vs_main": adjusted_rand_score(rs.labels[common_r],
                                                   ref.labels[[rpos[inst["raters"][i]] for i in common_r]]),
                "tau_vs_main": float(stats.kendalltau(rs.collective[im], ref.collective[jm])[0]),
                "tau_consensus_flat": float(stats.kendalltau(rs.collective, flat)[0]),
                "smallest_share": len(members) / len(rs.labels),
                "smallest_tie_share": float(tie_vote[battles["judge_hash"].isin(members)].mean()),
                "smallest_density": float(inst["observed"][rs.labels == small].mean()),
                "others_density": float(inst["observed"][rs.labels != small].mean()),
            })
    R = pd.DataFrame(reps)
    R.to_csv(OUT / "revision_replications.csv", index=False)
    for name, g in R.groupby("variant", sort=False):
        rows.append({
            "variant": name, "raters": int(g["raters"].iloc[0]), "rho_f1_f2": g["rho_f1_f2"].iloc[0],
            "HV_SOBO": g["HV_SOBO"].median(), "HV_MOBO": g["HV_MOBO"].median(),
            "MOBO_larger_HV": int((g["HV_MOBO"] > g["HV_SOBO"]).sum()), "replications": len(g),
            "same_decision": int(g["same_decision"].sum()),
            "ARI_SOBO_MOBO": g["ARI_SOBO_MOBO"].median(), "tau_SOBO_MOBO": g["tau_SOBO_MOBO"].median(),
            "c_star": "/".join(map(str, sorted(set(g["c_SOBO"]) | set(g["c_MOBO"])))),
            "ARI_vs_main": g["ARI_vs_main"].median(), "tau_vs_main": g["tau_vs_main"].median(),
            "tau_consensus_flat": g["tau_consensus_flat"].median(),
            "tau_consensus_flat_min": g["tau_consensus_flat"].min(),
            "smallest_share": g["smallest_share"].median(), "smallest_tie_share": g["smallest_tie_share"].median(),
            "smallest_density": g["smallest_density"].median(), "others_density": g["others_density"].median(),
        })
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "revision_variants.csv", index=False)

    # Pilot-conflict rule (Section 3.2.2): does a negative pilot correlation predict a MOBO advantage?
    R["dHV"] = R["HV_MOBO"] - R["HV_SOBO"]
    neg = R["rho_f1_f2"] < 0
    table = [[int((R.loc[neg, "dHV"] > 0).sum()), int((R.loc[neg, "dHV"] <= 0).sum())],
             [int((R.loc[~neg, "dHV"] > 0).sum()), int((R.loc[~neg, "dHV"] <= 0).sum())]]
    fisher = stats.fisher_exact(table, alternative="greater")
    by_var = R.groupby("variant", sort=False).agg(rho=("rho_f1_f2", "first"), dHV=("dHV", "median"))
    sp = stats.spearmanr(by_var["rho"], by_var["dHV"])

    md = ["# Robustness of the inputs and the pilot-conflict rule (revision)", "",
          "| Variant | Raters | rho(f1,f2) pilot | HV SOBO / MOBO | MOBO larger HV | same decision | "
          "ARI; tau SOBO vs MOBO | c* | ARI; tau vs main decision | tau consensus vs flat (min) | "
          "smallest subgroup: share; ties; profile density (others) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in T.itertuples():
        md.append(f"| {r.variant} | {r.raters} | {r.rho_f1_f2:.2f} | {r.HV_SOBO:.3f} / {r.HV_MOBO:.3f} | "
                  f"{r.MOBO_larger_HV}/{r.replications} | {r.same_decision}/{r.replications} | "
                  f"{r.ARI_SOBO_MOBO:.2f}; {r.tau_SOBO_MOBO:.3f} | {r.c_star} | {r.ARI_vs_main:.2f}; {r.tau_vs_main:.3f} | "
                  f"{r.tau_consensus_flat:.3f} ({r.tau_consensus_flat_min:.3f}) | {100 * r.smallest_share:.1f}%; "
                  f"{100 * r.smallest_tie_share:.0f}%; {100 * r.smallest_density:.0f}% ({100 * r.others_density:.0f}%) |")
    md += ["", "## Pilot-conflict rule over all replications",
           f"- Negative pilot rho: MOBO larger HV in {table[0][0]} of {sum(table[0])}; "
           f"non-negative: {table[1][0]} of {sum(table[1])}; one-sided Fisher exact p = {fisher[1]:.3g}",
           f"- Spearman over {len(by_var)} variants, pilot rho vs median HV difference (MOBO - SOBO): "
           f"{sp[0]:.2f} (p = {sp[1]:.3g})"]
    (OUT / "revision_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if not args.summary:
        build_all(args.jobs)
        run_all(args.jobs)
    summarize()


if __name__ == "__main__":
    main()
