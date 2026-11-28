"""Step 14 (revision; Sections 2.6 and 3.1.3): do better objective values mean more meaningful subgroups?

The three objectives are internal criteria. This check asks whether configurations that score better on
them also find rater subgroups that predict the raters' own unseen votes, and how well the consensus
ranking predicts unseen battles compared with a pooled Bradley-Terry ranking.

Two kinds of split hold out 20% of every retained rater's battles: five random splits (seeds 1-5), which
test the reconstruction of missing votes, and one temporal split, which holds out each rater's latest
battles and so tests the prediction of later votes. The instance is rebuilt from the other 80% with the
same raters and models (fixed by vote counts and coverage, not by outcomes); the held-out battles enter
no stage of the rebuilt pipeline: profiles and their imputation, winner sets, PageRank, normalization
bounds, the optimization, partitions, subgroup scores, and the Bradley-Terry fit all use the training
battles only. All 512 configurations of a scrambled Sobol pilot sample (seed 0) are evaluated on the
training instance, which also gives the normalization bounds. For each configuration:
  - subgroup scores are the mean training scores of the members who rated a model (as in Stage 5);
  - every held-out battle of rater d_k between two of the 30 models is predicted from the scores of
    d_k's subgroup as 0.5 + (f_a - f_b) / 2 and compared with the outcome code (1, 0.5, 0) by the
    squared error; the pooled prediction uses the flat scores of all raters instead;
  - skill = 1 - error(subgroups) / error(pooled), so positive skill means the subgroups carry
    information about the raters' own unseen votes.
The skill is related to the normalized objectives over the 512 configurations (Spearman). SOBO (equal
weights, main settings) is then run again on every training instance, and the partition of its decision
is compared with 200 random partitions of the same sizes; the configuration of the main SOBO decision
(seed 1, optimized on all votes) is kept as a secondary check. Finally, the consensus ranking of the
re-optimized decision, the flat ranking, and a Bradley-Terry ranking fitted to the training battles are
scored by their accuracy on the decisive held-out battles, averaged over battles and over raters.

Outputs results/revision/holdout_summary.md and holdout_configs.csv.

Usage:
    python 15_holdout_validation.py --jobs 5
"""
import warnings

warnings.filterwarnings("ignore", message=".*torch.jit.script.*", category=FutureWarning)

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

import config  # noqa: E402

OUT = config.RESULTS / "revision"
SPLITS = tuple(("random", s) for s in (1, 2, 3, 4, 5)) + (("temporal", 1),)
TEST_SHARE = 0.2
N_NULL = 200


def bt_strengths(df, models):
    """Bradley-Terry log-strengths with ties as half wins (as in 12_bradley_terry.py)."""
    idx = {m: j for j, m in enumerate(models)}
    a, b = df["model_a"].map(idx).to_numpy(), df["model_b"].map(idx).to_numpy()
    X = np.zeros((len(df), len(models)))
    X[np.arange(len(df)), a] = 1.0
    X[np.arange(len(df)), b] = -1.0
    win_a, win_b = (df["winner"] == "model_a").to_numpy(), (df["winner"] == "model_b").to_numpy()
    tie = ~(win_a | win_b)
    Xs = np.vstack([X[~tie], X[tie], X[tie]])
    ys = np.concatenate([win_a[~tie], np.ones(tie.sum()), np.zeros(tie.sum())]).astype(int)
    ws = np.concatenate([np.ones((~tie).sum()), np.full(2 * tie.sum(), 0.5)])
    model = LogisticRegression(fit_intercept=False, C=1e6, max_iter=10_000)
    model.fit(Xs, ys, sample_weight=ws)
    return model.coef_[0]


def group_scores(labels, F_obs0, obs, flat):
    """Mean observed training score of every subgroup on every model; flat score where none exists."""
    c = labels.max() + 1
    onehot = np.eye(c)[labels]
    num, cnt = onehot.T @ F_obs0, onehot.T @ obs
    return np.where(cnt > 0, num / np.maximum(cnt, 1), flat[None, :])


def sq_error(scores_of_rater, k, a, b, gamma):
    p = np.clip(0.5 + (scores_of_rater[k, a] - scores_of_rater[k, b]) / 2.0, 0.0, 1.0)
    return (gamma - p) ** 2


def one_split(spec):
    import torch

    from src.data import build_instance, load_battles, load_instance
    from src.optimizers import run_bo, sobol_design
    from src.scofcm import SCOFCMPipeline, normalize, scalarize

    torch.set_num_threads(2)
    kind, split = spec
    main = load_instance(config.DATA_PROCESSED / "instance.npz")
    raters, models = list(main["raters"]), list(main["models"])
    battles = load_battles()
    B = battles[battles["judge_hash"].isin(set(raters))].reset_index(drop=True)
    rng = np.random.default_rng(split if kind == "random" else 100 + split)
    tstamp = B["tstamp"].to_numpy()
    test = np.zeros(len(B), dtype=bool)
    for idx in B.groupby("judge_hash").indices.values():
        size = max(1, int(round(TEST_SHARE * len(idx))))
        if kind == "random":
            test[rng.choice(idx, size=size, replace=False)] = True
        else:                                        # each rater's latest battles
            test[idx[np.argsort(tstamp[idx], kind="stable")[-size:]]] = True
    train_df, test_df = B[~test], B[test]
    assert not set(train_df.index) & set(test_df.index)

    inst = build_instance(train_df, raters=raters, models=models)
    pipe = SCOFCMPipeline(inst["F"], inst["alpha"], inst["beta"], inst["pr_h"], inst["pr_c"],
                          observed=inst["observed"])
    obs = inst["observed"].astype(float)
    F_obs0 = np.where(inst["observed"], inst["F"], 0.0)
    flat = F_obs0.sum(0) / obs.sum(0)
    r_idx = {r: i for i, r in enumerate(inst["raters"])}
    m_idx = {m: j for j, m in enumerate(models)}
    T = test_df[test_df["model_a"].isin(m_idx) & test_df["model_b"].isin(m_idx) & test_df["judge_hash"].isin(r_idx)]
    k = T["judge_hash"].map(r_idx).to_numpy()
    a, b = T["model_a"].map(m_idx).to_numpy(), T["model_b"].map(m_idx).to_numpy()
    gamma = np.where(T["winner"] == "model_a", 1.0, np.where(T["winner"] == "model_b", 0.0, 0.5))
    decisive = gamma != 0.5
    err_pool = sq_error(np.broadcast_to(flat, (len(inst["raters"]), len(models))), k, a, b, gamma).mean()

    X = torch.quasirandom.SobolEngine(config.DIM, scramble=True, seed=config.PILOT_SEED).draw(
        config.PILOT_SIZE, dtype=torch.double).numpy()
    results = [pipe.evaluate(x) for x in X]
    Fr = np.array([r.f_raw for r in results])
    ok = np.all(np.isfinite(Fr), axis=1)
    bounds = (Fr[ok].min(0), Fr[ok].max(0))
    rows = []
    for x, res, good in zip(X, results, ok):
        if not good:
            continue
        fn = normalize(res.f_raw, bounds=bounds)
        G = group_scores(res.labels, F_obs0, obs, flat)
        err = sq_error(G[res.labels], k, a, b, gamma).mean()
        rows.append({"kind": kind, "split": split, "zeta": x[0], "kappa": x[1], "mu": x[2], "c_star": res.c_star,
                     "f1n": fn[0], "f2n": fn[1], "f3n": fn[2], "g": scalarize(fn, config.WEIGHTS["E"]),
                     "skill": 1.0 - err / err_pool})
    C = pd.DataFrame(rows)

    # SOBO run again on the training instance (equal weights, main settings), so that the decision itself
    # never sees the held-out battles; the main SOBO decision (seed 1, all votes) is a secondary check
    pipe_b = SCOFCMPipeline(inst["F"], inst["alpha"], inst["beta"], inst["pr_h"], inst["pr_c"],
                            observed=inst["observed"], bounds=bounds)
    cache = {}

    def evaluate(x):
        key = tuple(np.round(np.asarray(x, dtype=float), 12))
        if key not in cache:
            cache[key] = pipe_b.evaluate(key)
        return cache[key]

    recs = run_bo(evaluate, "SOBO", sobol_design(split), split, weights=config.WEIGHTS["E"], budget=config.BUDGET)
    x_train = recs[int(np.argmin([scalarize(r["f_norm"], config.WEIGHTS["E"]) for r in recs]))]["x"]
    with open(config.LOGS / "seed01_SOBO_E.jsonl") as fh:
        main_recs = [json.loads(line) for line in fh]
    x_main = main_recs[int(np.argmin([scalarize(r["f_norm"], config.WEIGHTS["E"]) for r in main_recs]))]["x"]
    dec_main = pipe.evaluate(x_main)
    skill_main = 1.0 - sq_error(group_scores(dec_main.labels, F_obs0, obs, flat)[dec_main.labels],
                                k, a, b, gamma).mean() / err_pool
    dec = pipe.evaluate(x_train)
    G = group_scores(dec.labels, F_obs0, obs, flat)
    skill_dec = 1.0 - sq_error(G[dec.labels], k, a, b, gamma).mean() / err_pool
    null = []
    for _ in range(N_NULL):
        perm = rng.permutation(dec.labels)
        Gp = group_scores(perm, F_obs0, obs, flat)
        null.append(1.0 - sq_error(Gp[perm], k, a, b, gamma).mean() / err_pool)
    small = int(np.argmin(np.bincount(dec.labels)))
    in_small = dec.labels[k] == small
    err_dec = sq_error(G[dec.labels], k, a, b, gamma)
    err_flat = sq_error(np.broadcast_to(flat, G[dec.labels].shape), k, a, b, gamma)

    # rankings on decisive held-out battles
    train_among = train_df[train_df["model_a"].isin(m_idx) & train_df["model_b"].isin(m_idx)]
    bt = bt_strengths(train_among, models)

    def accuracy(score):
        hit = (score[a[decisive]] > score[b[decisive]]) == (gamma[decisive] == 1.0)
        per_rater = pd.Series(hit).groupby(k[decisive]).mean()
        return float(hit.mean()), float(per_rater.mean())

    acc = {"consensus": accuracy(dec.collective), "flat": accuracy(flat), "bradley_terry": accuracy(bt)}
    summary = {
        "kind": kind, "split": split, "test_battles": int(len(T)), "decisive": int(decisive.sum()),
        "raters_in_training": int(len(inst["raters"])), "c_star_decision": dec.c_star,
        "sizes_decision": "/".join(map(str, sorted(np.bincount(dec.labels), reverse=True))),
        "x_train": "/".join(f"{v:.3f}" for v in x_train),
        "skill_decision": skill_dec, "null_mean": float(np.mean(null)), "null_max": float(np.max(null)),
        "beats_all_null": bool(skill_dec > np.max(null)),
        "skill_main_decision": skill_main,
        "skill_small_battles": 1.0 - err_dec[in_small].mean() / err_flat[in_small].mean(),
        "skill_large_battles": 1.0 - err_dec[~in_small].mean() / err_flat[~in_small].mean(),
        "share_small_battles": float(in_small.mean()),
        "rho_g_skill": float(stats.spearmanr(C["g"], C["skill"])[0]),
        "rho_f1_skill": float(stats.spearmanr(C["f1n"], C["skill"])[0]),
        "rho_f2_skill": float(stats.spearmanr(C["f2n"], C["skill"])[0]),
        "rho_f3_skill": float(stats.spearmanr(C["f3n"], C["skill"])[0]),
        "skill_best_decile_g": float(C.loc[C["g"] <= C["g"].quantile(0.1), "skill"].median()),
        "skill_other_g": float(C.loc[C["g"] > C["g"].quantile(0.1), "skill"].median()),
        "tau_consensus_bt": float(stats.kendalltau(dec.collective, bt)[0]),
        **{f"acc_{n}_battles": v[0] for n, v in acc.items()},
        **{f"acc_{n}_raters": v[1] for n, v in acc.items()},
    }
    return summary, C


def describe(S, title):
    med = S.median(numeric_only=True)
    rng_ = lambda col: f"{S[col].min():.3f} to {S[col].max():.3f}"
    return [f"## {title}", "",
            f"- Held-out battles between two of the 30 models: median {med.test_battles:.0f} "
            f"({med.decisive:.0f} decisive); raters in training {rng_('raters_in_training')}",
            f"- SOBO re-run on the training instances: x {list(S.x_train)}, c* {sorted(set(S.c_star_decision))}, "
            f"sizes {sorted(set(S.sizes_decision))}",
            f"- Skill of its subgroups (1 - error/pooled error): median {med.skill_decision:.4f} ({rng_('skill_decision')}); "
            f"beats all {N_NULL} random partitions of the same sizes in {int(S.beats_all_null.sum())} of {len(S)}; "
            f"random partitions: mean {med.null_mean:.4f}, max {S.null_max.max():.4f}",
            f"- Secondary: configuration of the main SOBO decision (all votes): skill median {med.skill_main_decision:.4f}",
            f"- Skill on battles of the smallest subgroup ({100 * med.share_small_battles:.1f}% of held-out battles): "
            f"{med.skill_small_battles:.3f}; on the others: {med.skill_large_battles:.3f}",
            f"- Spearman over the 512 pilot configurations, objective vs skill (negative = better objective, better skill): "
            f"g {med.rho_g_skill:.2f} ({rng_('rho_g_skill')}), f1 {med.rho_f1_skill:.2f}, f2 {med.rho_f2_skill:.2f}, "
            f"f3 {med.rho_f3_skill:.2f}",
            f"- Median skill of the best 10% configurations by g: {med.skill_best_decile_g:.4f}; of the rest: {med.skill_other_g:.4f}",
            f"- Accuracy on decisive held-out battles (battle-averaged / rater-averaged): consensus "
            f"{med.acc_consensus_battles:.3f} / {med.acc_consensus_raters:.3f}; flat {med.acc_flat_battles:.3f} / "
            f"{med.acc_flat_raters:.3f}; Bradley-Terry {med.acc_bradley_terry_battles:.3f} / {med.acc_bradley_terry_raters:.3f}",
            f"- Kendall tau, consensus vs Bradley-Terry on the training battles: median {med.tau_consensus_bt:.3f}", ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with mp.get_context("spawn").Pool(min(args.jobs, len(SPLITS))) as pool:
        out = pool.map(one_split, SPLITS)
    S = pd.DataFrame([s for s, _ in out])
    C = pd.concat([c for _, c in out], ignore_index=True)
    C.to_csv(OUT / "holdout_configs.csv", index=False)
    S.to_csv(OUT / "holdout_splits.csv", index=False)
    md = ["# Held-out validation of the objectives and decisions", ""]
    md += describe(S[S.kind == "random"], "Five random 80/20 splits per rater (missing votes)")
    md += describe(S[S.kind == "temporal"], "Temporal split: each rater's latest 20% of battles (later votes)")
    (OUT / "holdout_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
