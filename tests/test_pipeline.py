"""Sanity checks for the data and pipeline code. Run: python tests/test_pipeline.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import _jaccard, phfe_score  # noqa: E402
from src.scofcm import SCOFCMPipeline, normalize, scalarize  # noqa: E402


def test_score_function():
    # {1(0.75), 0(0.25)}: 0.75 - (0.75 - 0) / 2 = 0.375
    assert abs(phfe_score(np.array([1, 1, 1, 0])) - 0.375) < 1e-12
    # {0.5(1)}: a consistent tie scores 0.5
    assert abs(phfe_score(np.array([0.5, 0.5])) - 0.5) < 1e-12
    # {1(1)} and {0(1)}
    assert phfe_score(np.array([1.0])) == 1.0 and phfe_score(np.array([0.0])) == 0.0


def test_jaccard():
    B = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=bool)
    J = _jaccard(B)
    assert abs(J[0, 1] - 0.5) < 1e-12 and J[0, 2] == 0 and J[2, 2] == 1


def synthetic_instance(n_per=40, m=6, seed=0):
    """Three well-separated rater groups with matching reachability."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(0.2, 0.8, size=(3, m))
    F = np.vstack([np.clip(c + 0.03 * rng.standard_normal((n_per, m)), 0, 1) for c in centres])
    g = np.repeat(np.arange(3), n_per)
    same = (g[:, None] == g[None, :]).astype(float)
    alpha = np.clip(0.7 * same + 0.1 * rng.random((len(g), len(g))), 0, 1)
    beta = np.clip(0.6 * same + 0.1 * rng.random((len(g), len(g))), 0, 1)
    alpha = (alpha + alpha.T) / 2
    beta = (beta + beta.T) / 2
    np.fill_diagonal(alpha, 1)
    np.fill_diagonal(beta, 1)
    pr = np.full(len(g), 1 / len(g))
    return F, alpha, beta, pr, pr, g


def test_pipeline_outputs():
    F, alpha, beta, pr_h, pr_c, g = synthetic_instance()
    pipe = SCOFCMPipeline(F, alpha, beta, pr_h, pr_c, c_candidates=range(2, 6))
    res = pipe.evaluate((0.4, 0.6, 0.5))
    assert res.c_star in range(2, 6)
    assert np.all((res.f_norm >= 0) & (res.f_norm <= 1)), res.f_norm
    assert abs(res.weights.sum() - 1) < 1e-9
    assert len(res.ranking) == F.shape[1]
    # deterministic
    res2 = pipe.evaluate((0.4, 0.6, 0.5))
    assert np.allclose(res.f_raw, res2.f_raw)
    print("  synthetic: c* =", res.c_star, "| f_raw =", np.round(res.f_raw, 4),
          "| iters =", res.iters_by_c, "| converged =", res.converged_by_c)
    return res


def test_cost_modes():
    """With the literal 'sum' cost, FCM cycles for c != true number of groups;
    the 'mean' cost converges for every c (reason for config.CC_MODE = 'mean')."""
    F, alpha, beta, pr_h, pr_c, g = synthetic_instance()
    conv = {}
    for mode in ("sum", "mean"):
        pipe = SCOFCMPipeline(F, alpha, beta, pr_h, pr_c, c_candidates=range(2, 6), cc_mode=mode)
        conv[mode] = pipe.evaluate((0.4, 0.6, 0.5)).converged_by_c
    print("  converged (sum): ", conv["sum"])
    print("  converged (mean):", conv["mean"])
    assert all(conv["mean"].values())


def test_observed_subgroup_scores():
    """Subgroup scores use real votes only; with a full mask they equal the plain means."""
    F, alpha, beta, pr_h, pr_c, g = synthetic_instance()
    rng = np.random.default_rng(1)
    obs = rng.random(F.shape) < 0.5
    obs[:, 0] = True
    pipe = SCOFCMPipeline(F, alpha, beta, pr_h, pr_c, c_candidates=range(2, 6), observed=obs)
    U = np.eye(3)[g]
    _, f_group, _, has = pipe.weights_and_ranking(U, g, 0.5)
    for r in range(3):
        for i in range(F.shape[1]):
            rows = (g == r) & obs[:, i]
            if rows.any():
                assert abs(f_group[r, i] - F[rows, i].mean()) < 1e-12
            else:
                assert not has[r, i]
    full = SCOFCMPipeline(F, alpha, beta, pr_h, pr_c, c_candidates=range(2, 6), observed=np.ones_like(obs))
    plain = SCOFCMPipeline(F, alpha, beta, pr_h, pr_c, c_candidates=range(2, 6))
    assert np.allclose(full.evaluate((0.4, 0.6, 0.5)).f_raw, plain.evaluate((0.4, 0.6, 0.5)).f_raw)


def test_normalize_and_scalarize():
    fn = normalize(np.array([1.0, 0.0, 7.0]), n_c=7, t_max=300)
    assert fn[0] == 0.0 and fn[1] == 0.0 and abs(fn[2] - 7 / 2100) < 1e-12
    assert abs(scalarize([0.3, 0.3, 0.3], (1 / 3, 1 / 3, 1 / 3)) - 0.3) < 1e-12


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
