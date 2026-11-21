"""Step 16 (revision; Sections 3.1.2 and 3.1.3): who the small subgroup is, and how close MOBO comes to SOBO.

Part 1 profiles the two subgroups of the main SOBO decision (seed 1; all SOBO decisions share this
partition): votes per rater, tie types, prompt languages, activity in the two periods, model coverage,
and the share of ties of every member. It tests whether the near-reverse ranking of the small subgroup
(Kendall's tau of the two subgroup rankings) differs from zero with a bootstrap over its members, and
how often the decisive votes of each subgroup agree with the ranking of the large subgroup.

Part 2 qualifies the weight-coverage result of RQ2. For every alternative weight vector w, it compares
the gap Delta_w between the MOBO archive and the dedicated SOBO run (Eq. 18) with the spread between
two SOBO runs for w that differ only in the seed (all pairs of seeds), and reports the median gaps
with 95% bootstrap intervals, the gaps relative to the SOBO optimum, and the share of seeds in which
MOBO came within the median run-to-run spread.

Outputs results/revision/minority_coverage.md.

Usage:
    python 17_minority_and_coverage.py
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

import config
from src.data import load_battles, load_instance
from src.scofcm import pipeline_from_instance, scalarize

OUT = config.RESULTS / "revision"
N_BOOT = 2000


def read(seed, run):
    with open(config.LOGS / f"seed{seed:02d}_{run}.jsonl") as fh:
        return [json.loads(line) for line in fh]


def load_prompt_meta():
    """Prompt metadata of every battle (same row order as load_battles), cached in data/raw/.

    Only small columns are read: user tokens and turns, LMArena's high-frequency-prompt flag, and its
    seven hard-prompt criteria (a prompt meeting at least six counts as hard).
    """
    import pyarrow.parquet as pq

    from src.data import _HTTPRangeFile

    cache = config.DATA_RAW / "prompt_meta.parquet"
    if not cache.exists():
        pf = pq.ParquetFile(_HTTPRangeFile(config.DATASET_URL), pre_buffer=True)
        t = pf.read(columns=["judge_hash", "tstamp", "conv_metadata", "dedup_tag", "category_tag"]).to_pandas()
        crit = pd.DataFrame(t["category_tag"].map(lambda c: c["criteria_v0.1"]).tolist())
        pd.DataFrame({
            "judge_hash": t["judge_hash"], "tstamp": t["tstamp"],
            "user_tokens": t["conv_metadata"].map(lambda c: c["sum_user_tokens"]),
            "turns": t["conv_metadata"].map(lambda c: c["turns"]),
            "high_freq": t["dedup_tag"].map(lambda c: bool(c["high_freq"])),
            "criteria": crit.fillna(False).astype(int).sum(axis=1),
        }).to_parquet(cache)
    return pd.read_parquet(cache)


def subgroup_profile(inst, battles, labels):
    lines = []
    raters = inst["raters"]
    cutoff = pd.Timestamp(config.CUTOFF_DATE, tz="UTC").timestamp()
    meta = load_prompt_meta()
    assert (meta["judge_hash"].to_numpy() == battles["judge_hash"].to_numpy()).all()
    battles = battles.assign(user_tokens=meta["user_tokens"].to_numpy(), turns=meta["turns"].to_numpy(),
                             high_freq=meta["high_freq"].to_numpy(), hard=(meta["criteria"] >= 6).to_numpy())
    rows = []
    for g in np.unique(labels):
        members = set(raters[labels == g])
        b = battles[battles["judge_hash"].isin(members)]
        per = b.groupby("judge_hash")
        tie_r = per["winner"].apply(lambda w: (~w.isin(["model_a", "model_b"])).mean())
        hist = b.assign(h=b["tstamp"].astype(float) < cutoff).groupby("judge_hash")["h"].agg(["any", "all"])
        lang = b["language"].value_counts(normalize=True)
        rows.append({
            "subgroup": int(g), "raters": len(members), "votes_median": float(per.size().median()),
            "votes_iqr": (float(per.size().quantile(0.25)), float(per.size().quantile(0.75))),
            "tie": float((b["winner"] == "tie").mean()), "bothbad": float((b["winner"] == "tie (bothbad)").mean()),
            "rater_tie_share_min": float(tie_r.min()), "rater_tie_share_median": float(tie_r.median()),
            "english": float(lang.get("English", 0.0)), "top_languages": ", ".join(
                f"{k} {100 * v:.1f}%" for k, v in lang.head(5).items()),
            "unknown": float(lang.get("unknown", 0.0)),
            "tokens_median": float(b["user_tokens"].median()), "multi_turn": float((b["turns"] > 1).mean()),
            "high_freq": float(b["high_freq"].mean()), "hard": float(b["hard"].mean()),
            "both_periods": float((hist["any"] & ~hist["all"]).mean()),
            "density": float(inst["observed"][labels == g].mean()),
            "votes": b,
        })
    big, small = sorted(rows, key=lambda r: -r["raters"])[:2]
    for r in (big, small):
        lines.append(f"- Subgroup of {r['raters']} raters: votes per rater median {r['votes_median']:.0f} "
                     f"(IQR {r['votes_iqr'][0]:.0f}-{r['votes_iqr'][1]:.0f}); ties {100 * (r['tie'] + r['bothbad']):.1f}% "
                     f"(tie {100 * r['tie']:.1f}%, bothbad {100 * r['bothbad']:.1f}%); rater tie share min "
                     f"{100 * r['rater_tie_share_min']:.0f}%, median {100 * r['rater_tie_share_median']:.0f}%; "
                     f"languages {r['top_languages']}; votes in both periods {100 * r['both_periods']:.0f}% of raters; "
                     f"profile density {100 * r['density']:.1f}%; prompts: median {r['tokens_median']:.0f} user tokens, "
                     f"multi-turn {100 * r['multi_turn']:.1f}%, high-frequency {100 * r['high_freq']:.1f}%, "
                     f"hard {100 * r['hard']:.1f}%")
    # language and activity: large vs small subgroup
    eng = [[int((r["votes"]["language"] == "English").sum()), int((r["votes"]["language"] != "English").sum())]
           for r in (big, small)]
    chi = stats.chi2_contingency(eng)
    lines.append(f"- English share of votes: large {100 * big['english']:.1f}% vs small {100 * small['english']:.1f}% "
                 f"(chi-square p = {chi[1]:.3g})")
    vb = big["votes"].groupby("judge_hash").size()
    vs = small["votes"].groupby("judge_hash").size()
    lines.append(f"- Votes per rater, large vs small: Mann-Whitney p = {stats.mannwhitneyu(vb, vs)[1]:.3g}")
    tb, ts = big["votes"]["user_tokens"], small["votes"]["user_tokens"]
    lines.append(f"- User tokens per prompt, large vs small: medians {tb.median():.0f} vs {ts.median():.0f}, "
                 f"Mann-Whitney p = {stats.mannwhitneyu(tb, ts)[1]:.3g}")
    for col in ("high_freq", "hard"):
        tab = [[int(r["votes"][col].sum()), int((~r["votes"][col]).sum())] for r in (big, small)]
        lines.append(f"- {col} prompts, large vs small: {100 * big[col]:.1f}% vs {100 * small[col]:.1f}% "
                     f"(chi-square p = {stats.chi2_contingency(tab)[1]:.3g})")
    # tie share by prompt type over all retained raters: are ties a property of prompts or of raters?
    ret = battles[battles["judge_hash"].isin(set(raters))]
    tie = ~ret["winner"].isin(["model_a", "model_b"])
    lines.append(f"- Ties among all retained raters' votes: high-frequency prompts {100 * tie[ret['high_freq']].mean():.1f}%, "
                 f"others {100 * tie[~ret['high_freq']].mean():.1f}%; prompts of at most 5 user tokens "
                 f"{100 * tie[ret['user_tokens'] <= 5].mean():.1f}%, longer {100 * tie[ret['user_tokens'] > 5].mean():.1f}%")
    return lines, big, small


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inst = load_instance(config.DATA_PROCESSED / "instance.npz")
    battles = load_battles()
    pipe = pipeline_from_instance(inst)
    recs = read(1, "SOBO_E")
    x = recs[int(np.argmin([scalarize(r["f_norm"], config.WEIGHTS["E"]) for r in recs]))]["x"]
    dec = pipe.evaluate(x)
    md = ["# Small subgroup and weight coverage (revision)", "", "## Part 1: the two subgroups of the SOBO decision"]
    lines, big, small = subgroup_profile(inst, battles, dec.labels)
    md += lines

    # subgroup rankings on real votes and the bootstrap of tau
    obs = inst["observed"]
    F0 = np.where(obs, inst["F"], 0.0)
    g_small = int(np.argmin(np.bincount(dec.labels)))
    idx_small, idx_big = np.where(dec.labels == g_small)[0], np.where(dec.labels != g_small)[0]

    def scores(idx):
        cnt = obs[idx].sum(0)
        return np.where(cnt > 0, F0[idx].sum(0) / np.maximum(cnt, 1), np.nan)

    s_big, s_small = scores(idx_big), scores(idx_small)
    ok = np.isfinite(s_small)
    tau0 = stats.kendalltau(s_big[ok], s_small[ok])[0]
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(N_BOOT):
        s = scores(rng.choice(idx_small, size=len(idx_small), replace=True))
        m = np.isfinite(s)
        boot.append(stats.kendalltau(s_big[m], s[m])[0])
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    md.append(f"- Kendall tau between the two subgroup rankings: {tau0:.2f}, 95% bootstrap interval over the "
              f"small subgroup's raters [{lo:.2f}, {hi:.2f}]")
    # decisive votes that agree with the large subgroup's ranking
    models = list(inst["models"])
    m_idx = {m: j for j, m in enumerate(models)}
    for name, r in (("large", big), ("small", small)):
        v = r["votes"]
        v = v[v["model_a"].isin(m_idx) & v["model_b"].isin(m_idx) & v["winner"].isin(["model_a", "model_b"])]
        a, b = v["model_a"].map(m_idx).to_numpy(), v["model_b"].map(m_idx).to_numpy()
        win_a = (v["winner"] == "model_a").to_numpy()
        diff = s_big[a] - s_big[b]
        agree = ((diff > 0) == win_a)[diff != 0]
        md.append(f"- Decisive votes of the {name} subgroup between two of the 30 models: {len(v)}; agreeing with "
                  f"the large subgroup's ranking: {100 * agree.mean():.1f}%")

    # Part 2: weight coverage
    md += ["", "## Part 2: weight-coverage gaps (Eq. 18) against the run-to-run spread of SOBO"]
    for key in ("V", "C", "T"):
        w = config.WEIGHTS[key]
        sobo = np.array([min(scalarize(r["f_norm"], w) for r in read(s, f"SOBO_{key}")) for s in config.SEEDS])
        mobo = np.array([min(scalarize(r["f_norm"], w) for r in read(s, "MOBO")) for s in config.SEEDS])
        gap = mobo - sobo
        spread = np.array([abs(a - b) for i, a in enumerate(sobo) for b in sobo[i + 1:]])
        bm = [np.median(rng.choice(gap, size=len(gap), replace=True)) for _ in range(N_BOOT)]
        within = np.mean(gap <= np.median(spread))
        md.append(f"- w_{key}: gap median {np.median(gap):+.4f} [95% CI {np.percentile(bm, 2.5):+.4f}, "
                  f"{np.percentile(bm, 97.5):+.4f}], upper quartile {np.quantile(gap, 0.75):+.4f}, max {gap.max():+.4f}; "
                  f"relative to the SOBO optimum median {100 * np.median(gap / sobo):+.1f}% (max {100 * np.max(gap / sobo):+.1f}%); "
                  f"MOBO at least as good in {int((gap <= 1e-9).sum())}/20; run-to-run spread of SOBO median "
                  f"{np.median(spread):.4f} (upper quartile {np.quantile(spread, 0.75):.4f}); gap within the median "
                  f"spread in {100 * within:.0f}% of seeds")
    (OUT / "minority_coverage.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
