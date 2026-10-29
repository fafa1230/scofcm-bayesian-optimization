"""Step 12 (Section 3.1.4, Table 7): sensitivity of the findings to the data choices.

Nine variants of the instance are built from the same battles:
  votes5, votes20  raters with at least 5 or 20 votes instead of 10;
  bothbad0         "tie (bothbad)" coded as a loss (gamma = 0) for both models, "tie" kept at 0.5;
  decisive         ties left out of the P-HFE scores (decisive outcomes only);
  sub80_1..5       random 80% of the raters with at least 10 votes (seeds 1-5).
Each variant gets its own pilot bounds (512 scrambled Sobol points) and five paired replications
(seeds 1-5) of SOBO (equal weights) and MOBO with the main settings. The summary lists, per variant,
the conflict among the objectives on the pilot sample, the solution quality of both methods, how
often they select the same decision, and the decision itself.

Outputs data/raw/sensitivity/*.npz (instances), results/sensitivity/logs/<variant>/*.jsonl,
results/sensitivity/summary.md and results/sensitivity/sensitivity.csv.

Usage:
    python 13_sensitivity.py --jobs 6     # build, run, and summarize (about 40 min)
    python 13_sensitivity.py --summary    # summarize existing logs only
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
    "votes5": {"min_votes": 5},
    "votes20": {"min_votes": 20},
    "bothbad0": {"tie_values": {"tie": 0.5, "tie (bothbad)": 0.0}},
    "decisive": {"tie_values": {"tie": None, "tie (bothbad)": None}},
    **{f"sub80_{k}": {"subsample": k} for k in range(1, 6)},
}


# ---------------------------------------------------------------------------
# 1. Instances
# ---------------------------------------------------------------------------
def build_variant(name, battles):
    from src.data import build_instance, pilot_bounds
    from src.scofcm import SCOFCMPipeline

    spec = VARIANTS[name]
    df = battles
    if "subsample" in spec:
        votes = df.groupby("judge_hash").size()
        raters = np.array(sorted(votes[votes >= config.MIN_VOTES_PER_RATER].index))
        rng = np.random.default_rng(spec["subsample"])
        df = df[df["judge_hash"].isin(rng.choice(raters, size=round(0.8 * len(raters)), replace=False))]
    inst = build_instance(df, min_votes=spec.get("min_votes"), tie_values=spec.get("tie_values"))
    pipe = SCOFCMPipeline(inst["F"], inst["alpha"], inst["beta"], inst["pr_h"], inst["pr_c"],
                          observed=inst["observed"])
    lo, hi, pilot = pilot_bounds(pipe)
    inst.update(bounds_lo=lo, bounds_hi=hi, pilot_f_raw=pilot)
    inst.pop("stats")
    INST_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(INST_DIR / f"{name}.npz", **inst)


def _build_task(name):
    from src.data import load_battles

    t0 = time.perf_counter()
    build_variant(name, load_battles())
    return f"built {name} in {time.perf_counter() - t0:.0f} s"


def build_all(jobs):
    todo = [v for v in VARIANTS if not (INST_DIR / f"{v}.npz").exists()]
    if not todo:
        return
    threads = max(1, (mp.cpu_count() or 2) // jobs)
    with mp.get_context("spawn").Pool(jobs, initializer=_init_worker, initargs=(threads,)) as pool:
        for msg in pool.imap_unordered(_build_task, todo):
            print(msg, flush=True)


# ---------------------------------------------------------------------------
# 2. Paired SOBO / MOBO runs
# ---------------------------------------------------------------------------
WORKER = {}


def _init_worker(threads):
    import torch

    torch.set_num_threads(threads)


def _pipeline(name):
    from src.data import load_instance
    from src.scofcm import pipeline_from_instance

    if name not in WORKER:
        WORKER[name] = (pipeline_from_instance(load_instance(INST_DIR / f"{name}.npz")), {})
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
    # longest tasks first: MOBO, and the large votes5 instance
    tasks = [(v, s, run, mode, w) for v in VARIANTS for s in SEEDS for run, mode, w in RUNS]
    tasks.sort(key=lambda t: (t[2] != "MOBO", t[0] != "votes5"))
    threads = max(1, (mp.cpu_count() or 2) // jobs)
    with mp.get_context("spawn").Pool(jobs, initializer=_init_worker, initargs=(threads,)) as pool:
        for msg in pool.imap_unordered(run_task, tasks):
            print(msg, flush=True)


# ---------------------------------------------------------------------------
# 3. Summary
# ---------------------------------------------------------------------------
def summarize():
    from src.data import load_battles, load_instance
    from src.metrics import compromise_index, hypervolume
    from src.scofcm import pipeline_from_instance, scalarize

    wE = config.WEIGHTS["E"]
    battles = load_battles()
    tie_vote = ~battles["winner"].isin(["model_a", "model_b"])
    rows = []
    # reference row: the main instance with its 20 replications, summarized the same way
    sources = [("main", config.DATA_PROCESSED / "instance.npz", config.LOGS, config.SEEDS)]
    sources += [(v, INST_DIR / f"{v}.npz", OUT / "logs" / v, SEEDS) for v in VARIANTS]
    for name, inst_path, log_dir, seeds in sources:
        inst = load_instance(inst_path)
        pipe = pipeline_from_instance(inst)
        lo, hi = inst["bounds_lo"], inst["bounds_hi"]
        if "pilot_f_raw" in inst:
            P = inst["pilot_f_raw"]
        else:                             # main instance: same 512-point pilot sample (seed 0)
            from src.data import pilot_bounds
            from src.scofcm import SCOFCMPipeline
            P = pilot_bounds(SCOFCMPipeline(inst["F"], inst["alpha"], inst["beta"], inst["pr_h"], inst["pr_c"],
                                            observed=inst["observed"]))[2]
        Pn = np.column_stack([(hi[0] - P[:, 0]) / (hi[0] - lo[0]), (P[:, 1] - lo[1]) / (hi[1] - lo[1]),
                              (P[:, 2] - lo[2]) / (hi[2] - lo[2])])
        rho12 = stats.spearmanr(Pn[:, 0], Pn[:, 1])
        rho_min = min(stats.spearmanr(Pn[:, i], Pn[:, j])[0] for i, j in ((0, 1), (0, 2), (1, 2)))
        rec = {"variant": name, "raters": int(len(inst["raters"])), "density": float(inst["observed"].mean()),
               "rho_f1_f2": float(rho12[0]), "p_f1_f2": float(rho12[1]), "rho_min": float(rho_min)}
        hv, g, same, cstar, minority, tie_share = {"SOBO": [], "MOBO": []}, {"SOBO": [], "MOBO": []}, 0, [], [], []
        aris, taus = [], []
        for s in seeds:
            res = {}
            for run, label in (("SOBO_E", "SOBO"), ("MOBO", "MOBO")):
                with open(log_dir / f"seed{s:02d}_{run}.jsonl") as fh:
                    recs = [json.loads(line) for line in fh]
                X = np.array([r["x"] for r in recs])
                Y = np.array([r["f_norm"] for r in recs])
                gs = np.array([scalarize(y, wE) for y in Y])
                hv[label].append(hypervolume(Y))
                g[label].append(gs.min())
                res[label] = pipe.evaluate(X[int(np.argmin(gs))] if label == "SOBO" else X[compromise_index(Y)])
            aris.append(adjusted_rand_score(res["SOBO"].labels, res["MOBO"].labels))
            taus.append(stats.kendalltau(res["SOBO"].collective, res["MOBO"].collective)[0])
            same += int(aris[-1] > 1 - 1e-12 and np.array_equal(res["SOBO"].ranking, res["MOBO"].ranking))
            r = res["SOBO"]
            cstar += [r.c_star, res["MOBO"].c_star]
            small = int(np.argmin(np.bincount(r.labels)))
            members = set(inst["raters"][r.labels == small])
            minority.append(len(members) / len(r.labels))
            tie_share.append(float(tie_vote[battles["judge_hash"].isin(members)].mean()))
        rec.update({
            "HV_SOBO": np.median(hv["SOBO"]), "HV_MOBO": np.median(hv["MOBO"]),
            "replications": len(seeds), "MOBO_larger_HV": int(sum(m > s for m, s in zip(hv["MOBO"], hv["SOBO"]))),
            "g_SOBO": np.median(g["SOBO"]), "g_MOBO": np.median(g["MOBO"]),
            "same_decision": same, "ARI_median": float(np.median(aris)), "tau_median": float(np.median(taus)),
            "c_star": "/".join(map(str, sorted(set(cstar)))),
            "smallest_subgroup_share": float(np.median(minority)), "smallest_subgroup_tie_share": float(np.median(tie_share)),
        })
        rows.append(rec)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "sensitivity.csv", index=False)
    md = [f"# Sensitivity to the data choices ({len(SEEDS)} paired replications per variant; main: 20)", "",
          "| Variant | Raters | rho(f1,f2) | min rho | HV SOBO / MOBO | MOBO larger HV | best g SOBO / MOBO | "
          "same decision | ARI; tau (median) | c* | smallest subgroup (share; ties in its votes) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in t.itertuples():
        md.append(f"| {r.variant} | {r.raters} | {r.rho_f1_f2:.2f} (p={r.p_f1_f2:.1g}) | {r.rho_min:.2f} | "
                  f"{r.HV_SOBO:.3f} / {r.HV_MOBO:.3f} | {r.MOBO_larger_HV}/{r.replications} | {r.g_SOBO:.3f} / {r.g_MOBO:.3f} | "
                  f"{r.same_decision}/{r.replications} | {r.ARI_median:.2f}; {r.tau_median:.3f} | {r.c_star} | "
                  f"{100 * r.smallest_subgroup_share:.1f}%; "
                  f"{100 * r.smallest_subgroup_tie_share:.0f}% |")
    (OUT / "summary.md").write_text("\n".join(md) + "\n")
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
