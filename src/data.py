"""Section 2.2: data loading, rater filtering, P-HFE scoring, and rater relations."""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import config

WIN_A, WIN_B = "model_a", "model_b"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
class _HTTPRangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file, fetched with HTTP range requests.

    Lets pyarrow read only the needed parquet columns instead of the full
    380 MB file (which also stores the conversation texts).
    """

    def __init__(self, url: str):
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers["Content-Length"])
            self.url = r.url
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def load_battles(local_parquet: Path | None = None) -> pd.DataFrame:
    """Load the vote columns of lmarena-ai/arena-human-preference-100k.

    Uses a local copy of the parquet file if given, otherwise reads only the
    required columns remotely. The result is cached in data/raw/.
    """
    cache = config.DATA_RAW / "battles_slim.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    if local_parquet is not None:
        table = pq.read_table(local_parquet, columns=config.DATASET_COLUMNS)
    else:
        pf = pq.ParquetFile(_HTTPRangeFile(config.DATASET_URL), pre_buffer=True)
        table = pf.read(columns=config.DATASET_COLUMNS)
    pq.write_table(table, cache)
    return table.to_pandas()


# ---------------------------------------------------------------------------
# P-HFE score function (Eq. 1)
# ---------------------------------------------------------------------------
def phfe_score(gammas: np.ndarray) -> float:
    """Score of the P-HFE built from the outcomes of one rater on one model.

    gammas: outcomes coded 1 (win), 0.5 (tie), 0 (loss).
    f(h(p)) = sum_l gamma_l p_l - (max_l gamma_l p_l - min_l gamma_l p_l) / |h(p)|
    """
    values, counts = np.unique(gammas, return_counts=True)
    p = counts / counts.sum()
    gp = values * p
    fluctuation = (gp.max() - gp.min()) / len(values)
    return float(gp.sum() - fluctuation)


# ---------------------------------------------------------------------------
# Building the LSGDM instance
# ---------------------------------------------------------------------------
def _long_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (battle, side): rater, model, gamma, timestamp."""
    tie = ~df["winner"].isin([WIN_A, WIN_B])
    side_a = pd.DataFrame({
        "rater": df["judge_hash"].values,
        "model": df["model_a"].values,
        "gamma": np.where(df["winner"] == WIN_A, 1.0, np.where(tie, config.TIE_VALUE, 0.0)),
        "won": (df["winner"] == WIN_A).values,
        "tstamp": df["tstamp"].values,
    })
    side_b = pd.DataFrame({
        "rater": df["judge_hash"].values,
        "model": df["model_b"].values,
        "gamma": np.where(df["winner"] == WIN_B, 1.0, np.where(tie, config.TIE_VALUE, 0.0)),
        "won": (df["winner"] == WIN_B).values,
        "tstamp": df["tstamp"].values,
    })
    return pd.concat([side_a, side_b], ignore_index=True)


def _jaccard(B: np.ndarray) -> np.ndarray:
    """Pairwise Jaccard similarity between the rows of a binary matrix (Eq. 2)."""
    B = B.astype(np.float64)
    inter = B @ B.T
    size = B.sum(axis=1)
    union = size[:, None] + size[None, :] - inter
    J = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    np.fill_diagonal(J, 1.0)          # a rater is fully reachable from itself
    return J


def _pagerank(W: np.ndarray) -> np.ndarray:
    """PageRank centralities on a weighted, undirected rater network (no self-loops)."""
    A = W.copy()
    np.fill_diagonal(A, 0.0)
    G = nx.from_numpy_array(A)
    pr = nx.pagerank(G, alpha=config.PAGERANK_DAMPING, tol=config.PAGERANK_TOL,
                     max_iter=config.PAGERANK_MAX_ITER, weight="weight")
    return np.array([pr[i] for i in range(len(A))])


def build_instance(df: pd.DataFrame) -> dict:
    """Apply the selection rules of Section 2.2.1 and build all fixed inputs of the pipeline."""
    stats = {
        "total_battles": int(len(df)),
        "total_raters": int(df["judge_hash"].nunique()),
        "vote_outcomes_pct": (df["winner"].value_counts(normalize=True) * 100).round(2).to_dict(),
    }
    if "language" in df:
        stats["indonesian_votes"] = int((df["language"] == "Indonesian").sum())

    # Rule 1: raters with at least MIN_VOTES_PER_RATER votes
    votes = df.groupby("judge_hash").size()
    keep = votes[votes >= config.MIN_VOTES_PER_RATER].index
    d = df[df["judge_hash"].isin(keep)]
    stats["retained_raters"] = int(len(keep))
    stats["battles_retained_raters"] = int(len(d))

    # Rule 2: the N_MODELS models with the widest rater coverage (ties broken by name)
    long = _long_outcomes(d)
    coverage = long.drop_duplicates(["rater", "model"]).groupby("model").size()
    coverage = coverage.reset_index(name="n").sort_values(["n", "model"], ascending=[False, True])
    models = coverage["model"].head(config.N_MODELS).tolist()
    long = long[long["model"].isin(models)]

    raters = sorted(long["rater"].unique())
    r_idx = {r: i for i, r in enumerate(raters)}
    m_idx = {m: j for j, m in enumerate(models)}
    n, m = len(raters), len(models)

    # P-HFE scores (Eq. 1); NaN where a rater never evaluated a model
    F = np.full((n, m), np.nan)
    for (r, mod), g in long.groupby(["rater", "model"])["gamma"]:
        F[r_idx[r], m_idx[mod]] = phfe_score(g.to_numpy())
    observed = ~np.isnan(F)
    stats["raters_in_profile_matrix"] = int(n)
    stats["coverage_per_model_min_max"] = (int(observed.sum(0).min()), int(observed.sum(0).max()))
    stats["density_pct"] = round(100 * observed.mean(), 1)

    # Missing entries: model-wise mean over the raters who evaluated the model
    col_mean = np.nanmean(F, axis=0)
    F = np.where(observed, F, col_mean[None, :])

    # Historical / current relations (Section 2.2.3)
    cutoff = pd.Timestamp(config.CUTOFF_DATE, tz="UTC").timestamp()
    long = long.assign(hist=long["tstamp"].astype(float) < cutoff)
    has_hist = long.groupby("rater")["hist"].any()
    has_curr = (~long["hist"]).groupby(long["rater"]).any()

    def winner_sets(sub: pd.DataFrame) -> np.ndarray:
        B = np.zeros((n, m), dtype=bool)
        w = sub[sub["won"]]
        B[w["rater"].map(r_idx).to_numpy(), w["model"].map(m_idx).to_numpy()] = True
        return B

    B_full = winner_sets(long)
    B_hist = winner_sets(long[long["hist"]])
    B_curr = winner_sets(long[~long["hist"]])
    no_hist = np.array([not has_hist.get(r, False) for r in raters])
    no_curr = np.array([not has_curr.get(r, False) for r in raters])
    B_hist[no_hist] = B_full[no_hist]    # backup: full observation window
    B_curr[no_curr] = B_full[no_curr]
    stats["raters_without_historical_votes"] = int(no_hist.sum())
    stats["raters_without_current_votes"] = int(no_curr.sum())

    alpha = _jaccard(B_hist)
    beta = _jaccard(B_curr)
    pr_h = _pagerank(alpha)
    pr_c = _pagerank(beta)

    return {
        "F": F, "observed": observed, "alpha": alpha, "beta": beta, "pr_h": pr_h, "pr_c": pr_c,
        "raters": np.array(raters), "models": np.array(models), "stats": stats,
    }


def pilot_bounds(pipeline, size: int = config.PILOT_SIZE, seed: int = config.PILOT_SEED):
    """Raw-objective bounds (lo, hi) from a scrambled Sobol pilot sample (Eq. 12)."""
    import torch

    X = torch.quasirandom.SobolEngine(config.DIM, scramble=True, seed=seed).draw(size, dtype=torch.double)
    F = np.array([pipeline.evaluate(x).f_raw for x in X.numpy()])
    F = F[np.all(np.isfinite(F), axis=1)]
    return F.min(axis=0), F.max(axis=0), F


def save_instance(inst: dict, path: Path | None = None) -> Path:
    path = path or config.DATA_PROCESSED / "instance.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: v for k, v in inst.items() if k != "stats"})
    return path


def load_instance(path: Path | None = None) -> dict:
    path = path or config.DATA_PROCESSED / "instance.npz"
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
