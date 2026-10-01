"""Section 2.6: evaluation metrics and statistical helpers (all objectives minimized)."""
from __future__ import annotations

import numpy as np
from pymoo.indicators.hv import HV

import config
from src.scofcm import scalarize


def nondominated_mask(Y: np.ndarray) -> np.ndarray:
    """True for points not dominated by any other point (minimization)."""
    Y = np.asarray(Y, dtype=float)
    n = len(Y)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        dominated = np.all(Y <= Y[i], axis=1) & np.any(Y < Y[i], axis=1)
        if dominated.any():
            mask[i] = False
    return mask


def hypervolume(Y: np.ndarray, ref=config.REF_POINT) -> float:
    """Eq. 13: volume dominated by the non-dominated set of Y and bounded by ref."""
    Y = np.asarray(Y, dtype=float)
    Y = Y[np.all(Y < np.asarray(ref), axis=1)]
    if len(Y) == 0:
        return 0.0
    return float(HV(ref_point=np.asarray(ref, dtype=float))(Y[nondominated_mask(Y)]))


def anytime_hv(Y: np.ndarray, ref=config.REF_POINT) -> np.ndarray:
    return np.array([hypervolume(Y[: t + 1], ref) for t in range(len(Y))])


def anytime_best_g(Y: np.ndarray, weights) -> np.ndarray:
    g = np.array([scalarize(y, weights) for y in Y])
    return np.minimum.accumulate(g)


def compromise_index(Y: np.ndarray) -> int:
    """Index of the non-dominated point closest to the ideal point (0, 0, 0)."""
    Y = np.asarray(Y, dtype=float)
    idx = np.where(nondominated_mask(Y))[0]
    return int(idx[np.argmin(np.linalg.norm(Y[idx], axis=1))])


def a12(x, y) -> float:
    """Vargha-Delaney A12: probability that a value from x is larger than one from y."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    gt = (x[:, None] > y[None, :]).mean()
    eq = (x[:, None] == y[None, :]).mean()
    return float(gt + 0.5 * eq)


def a12_magnitude(a: float) -> str:
    d = abs(a - 0.5) + 0.5
    if d >= 0.71:
        return "large"
    if d >= 0.64:
        return "medium"
    if d >= 0.56:
        return "small"
    return "negligible"


def holm(pvalues) -> np.ndarray:
    """Holm-adjusted p-values."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj
