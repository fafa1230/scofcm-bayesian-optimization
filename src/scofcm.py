"""Sections 2.3-2.4: the SCO-FCM evaluation pipeline Phi and the three objectives.

Phi maps a configuration x = (zeta, kappa, mu) to the objective vector
(f1 cluster validity, f2 consensus-correction cost, f3 convergence effort).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import config


@dataclass
class PipelineResult:
    x: tuple
    f_raw: np.ndarray                  # (f1, f2, f3)
    f_norm: np.ndarray                 # normalized, all minimized (Eq. 12)
    c_star: int = -1
    labels: np.ndarray | None = None   # hard subgroup of every rater at c*
    weights: np.ndarray | None = None  # subgroup weights w_r (Eq. 8)
    collective: np.ndarray | None = None  # collective score of every model
    ranking: np.ndarray | None = None  # model indices, best first
    iters_by_c: dict = field(default_factory=dict)
    converged_by_c: dict = field(default_factory=dict)
    silhouette_by_c: dict = field(default_factory=dict)


def _sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    d = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
    return np.maximum(d, 0.0)


class SCOFCMPipeline:
    """Holds the fixed inputs (Section 2.2) and evaluates Phi(x)."""

    def __init__(self, F, alpha, beta, pr_h, pr_c,
                 c_candidates=config.C_CANDIDATES, fuzzifier=config.FUZZIFIER,
                 tol=config.FCM_TOL, t_max=config.T_MAX, cc_floor=config.CC_FLOOR,
                 cc_mode=config.CC_MODE, bounds=None, observed=None):
        """bounds: (lo, hi) raw-objective bounds for Eq. 12, or None for theoretical bounds.
        observed: boolean n x m mask of real votes; if given, subgroup scores use real votes only."""
        self.F = np.asarray(F, dtype=np.float64)
        self.alpha = np.asarray(alpha, dtype=np.float64)
        self.beta = np.asarray(beta, dtype=np.float64)
        self.pr_h = np.asarray(pr_h, dtype=np.float64)
        self.pr_c = np.asarray(pr_c, dtype=np.float64)
        self.C = tuple(c_candidates)
        self.a = float(fuzzifier)
        self.tol = float(tol)
        self.t_max = int(t_max)
        self.cc_floor = float(cc_floor)
        if cc_mode not in ("mean", "sum"):
            raise ValueError("cc_mode must be 'mean' or 'sum'")
        self.cc_mode = cc_mode
        self.bounds = None if bounds is None else (np.asarray(bounds[0], float), np.asarray(bounds[1], float))
        self.n, self.m = self.F.shape
        self.observed = None if observed is None else np.asarray(observed, dtype=bool)
        if self.observed is not None:
            self.F_obs0 = np.where(self.observed, self.F, 0.0)
            self.obs_f = self.observed.astype(np.float64)
        # Fixed quantities, independent of x
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            self.dist = np.sqrt(_sqdist(self.F, self.F))
        if not np.all(np.isfinite(self.dist)):
            raise FloatingPointError("non-finite rater distances")
        np.fill_diagonal(self.dist, 0.0)
        self.R_eval = config.ZETA_EVAL * self.alpha + (1 - config.ZETA_EVAL) * self.beta

    # -- Stage 1 (Eq. 3) ------------------------------------------------------
    def priority_and_reachability(self, zeta):
        pi = zeta * self.pr_h + (1 - zeta) * self.pr_c
        R = zeta * self.alpha + (1 - zeta) * self.beta
        return pi, R

    # -- Stage 2 (Eq. 4) ------------------------------------------------------
    def centre_order(self, pi, R, c_max):
        """Greedy order of initial centres; the first c entries are the centres for c subgroups."""
        order = [int(np.argmax(pi))]
        mask = np.ones(self.n, dtype=bool)
        mask[order[0]] = False
        sum_dist = self.dist[:, order[0]].copy()
        sum_reach = R[:, order[0]].copy()
        eps = 1e-12
        while len(order) < c_max:
            cand = np.where(mask)[0]
            p = pi[cand] / max(pi[cand].sum(), eps) + sum_dist[cand] / max(sum_dist[cand].sum(), eps)
            q = sum_reach[cand] / max(sum_reach[cand].sum(), eps)
            k = int(cand[np.argmin(q / np.maximum(p, eps))])
            order.append(k)
            mask[k] = False
            sum_dist += self.dist[:, k]
            sum_reach += R[:, k]
        return order

    # -- Stage 3 (Eqs. 5-6) ---------------------------------------------------
    def sco_fcm(self, centres, one_minus_R):
        """Soft-constrained FCM. Returns U (n x c), centres V, hard labels, iterations, converged."""
        F, a = self.F, self.a
        V = F[centres].copy()
        c = len(centres)
        labels = np.argmin(_sqdist(F, V), axis=1)
        eye = np.eye(c)
        U_prev = None
        cc_labels, cc = None, None
        for t in range(1, self.t_max + 1):
            if cc_labels is None or not np.array_equal(labels, cc_labels):
                onehot = eye[labels]
                cc = one_minus_R @ onehot                        # sum_{s in g_r} (1 - R_ks)
                if self.cc_mode == "mean":
                    cc = cc / np.maximum(onehot.sum(axis=0), 1.0)[None, :]
                cc = cc + self.cc_floor
                cc_labels = labels.copy()
            W = cc * (_sqdist(F, V) + 1e-12)
            if a == 2.0:
                inv = 1.0 / W
                U = inv / inv.sum(axis=1, keepdims=True)
            else:
                ratio = (W[:, :, None] / W[:, None, :]) ** (1.0 / (a - 1.0))
                U = 1.0 / ratio.sum(axis=2)
            Ua = (U ** a) * cc
            V = (Ua.T @ F) / Ua.sum(axis=0)[:, None]
            labels = np.argmax(U, axis=1)
            if U_prev is not None and np.max(np.abs(U - U_prev)) < self.tol:
                return U, V, labels, t, True
            U_prev = U
        return U, V, labels, self.t_max, False

    # -- Stage 4 (Eq. 7) ------------------------------------------------------
    def modified_silhouette(self, labels, V, kappa, R):
        """kappa * s1 (distance) + (1 - kappa) * s2 (reachability), averaged over raters.

        The nearest other subgroup of a rater is the one whose centre is closest to it
        (Yang et al., 2026). Singletons get s1 = s2 = 0.
        """
        n, c = self.n, V.shape[0]
        onehot = np.eye(c)[labels]
        counts = onehot.sum(axis=0)
        own = labels
        cdist = _sqdist(self.F, V)
        cdist[np.arange(n), own] = np.inf
        near = np.argmin(cdist, axis=1)

        sum_d = self.dist @ onehot
        sum_r = R @ onehot
        n_own = counts[own] - 1
        single = n_own <= 0
        safe_own = np.maximum(n_own, 1)

        A_in = sum_d[np.arange(n), own] / safe_own
        a_out = sum_d[np.arange(n), near] / counts[near]
        B_in = (sum_r[np.arange(n), own] - np.diag(R)) / safe_own
        b_out = sum_r[np.arange(n), near] / counts[near]

        def ratio(num, den):
            return np.divide(num, den, out=np.zeros_like(num), where=den > 0)

        s1 = ratio(a_out - A_in, np.maximum(A_in, a_out))
        s2 = ratio(B_in - b_out, np.maximum(B_in, b_out))
        s1[single] = 0.0
        s2[single] = 0.0
        return float(np.mean(kappa * s1 + (1 - kappa) * s2))

    # -- Stage 5 (Eq. 8) ------------------------------------------------------
    def weights_and_ranking(self, U, labels, mu):
        """Subgroup weights (Eq. 8), subgroup scores and collective scores.

        With an observed mask, the score of subgroup g_r on model a_i is the mean over the
        members of g_r who actually rated a_i, and the collective score averages the
        subgroups that have such members, with their weights renormalized.
        """
        c = U.shape[1]
        onehot = np.eye(c)[labels]
        counts = onehot.sum(axis=0)
        psi = (U * onehot).sum(axis=0) / U.sum(axis=0)
        w_mem = psi / psi.sum()
        w_size = counts / self.n
        w = mu * w_mem + (1 - mu) * w_size
        if self.observed is None:
            f_group = (onehot.T @ self.F) / counts[:, None]      # c x m
            has = np.ones_like(f_group, dtype=bool)
        else:
            num = onehot.T @ self.F_obs0                         # sums of real scores
            cnt = onehot.T @ self.obs_f                          # members who rated each model
            has = cnt > 0
            f_group = np.divide(num, cnt, out=np.zeros_like(num), where=has)
        w_eff = w[:, None] * has
        f_coll = (w_eff * f_group).sum(axis=0) / w_eff.sum(axis=0)
        return w, f_group, f_coll, has

    # -- Phi(x) (Section 2.4) -------------------------------------------------
    def evaluate(self, x) -> PipelineResult:
        # numpy 2.0 with Apple Accelerate emits spurious floating-point warnings from
        # matmul; they are silenced here and the outputs are checked for finiteness.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            res = self._evaluate(x)
        if res.c_star > 0 and not np.all(np.isfinite(res.f_raw)):
            raise FloatingPointError(f"non-finite objectives at x={res.x}: {res.f_raw}")
        return res

    def _evaluate(self, x) -> PipelineResult:
        zeta, kappa, mu = (float(v) for v in x)
        pi, R = self.priority_and_reachability(zeta)
        one_minus_R = 1.0 - R
        order = self.centre_order(pi, R, max(self.C))

        best = None
        res = PipelineResult(x=(zeta, kappa, mu), f_raw=np.full(3, np.nan), f_norm=np.ones(3))
        total_iters = 0
        for c in self.C:
            U, V, labels, iters, conv = self.sco_fcm(order[:c], one_minus_R)
            total_iters += iters
            res.iters_by_c[c] = iters
            res.converged_by_c[c] = conv
            if np.bincount(labels, minlength=c).min() == 0:
                continue                                     # empty subgroup -> invalid partition
            S = self.modified_silhouette(labels, V, kappa, R)
            res.silhouette_by_c[c] = S
            if best is None or S > best[0]:
                best = (S, c, U, V, labels)

        f3 = float(total_iters)
        if best is None:                                     # no valid partition
            res.f_raw = np.array([np.nan, np.nan, f3])
            return res

        _, c_star, U, V, labels = best
        f1 = self.modified_silhouette(labels, V, config.KAPPA_EVAL, self.R_eval)   # Eq. 9
        w, f_group, f_coll, has = self.weights_and_ranking(U, labels, mu)
        f2 = float(np.mean(np.abs(f_group - f_coll[None, :])[has]))                  # Eq. 10
        res.c_star = int(c_star)
        res.labels = labels
        res.weights = w
        res.collective = f_coll
        res.ranking = np.argsort(-f_coll, kind="stable")
        res.f_raw = np.array([f1, f2, f3])
        res.f_norm = normalize(res.f_raw, len(self.C), self.t_max, self.bounds)
        return res


def normalize(f_raw, n_c=len(config.C_CANDIDATES), t_max=config.T_MAX, bounds=None) -> np.ndarray:
    """Eq. 12: all objectives to minimization, rescaled with min-max bounds.

    bounds = (lo, hi) of the raw objectives (f1, f2, f3) from the pilot sample; if None,
    the theoretical bounds f1 in [-1, 1], f2 in [0, 1], f3 in [0, |C| * T_max] are used.
    """
    f1, f2, f3 = f_raw
    if bounds is None:
        return np.array([(1.0 - f1) / 2.0, f2, f3 / (n_c * t_max)])
    lo, hi = bounds
    span = np.maximum(hi - lo, 1e-12)
    return np.array([(hi[0] - f1) / span[0], (f2 - lo[1]) / span[1], (f3 - lo[2]) / span[2]])


def pipeline_from_instance(inst: dict, **kwargs) -> "SCOFCMPipeline":
    """Build Phi from a saved instance, using the pilot bounds when config.NORMALIZATION == 'pilot'."""
    bounds = None
    if config.NORMALIZATION == "pilot":
        if "bounds_lo" not in inst:
            raise RuntimeError("Pilot bounds missing: run 01_prepare_data.py first.")
        bounds = (inst["bounds_lo"], inst["bounds_hi"])
    observed = None
    if config.SUBGROUP_SCORES == "observed":
        if "observed" not in inst:
            raise RuntimeError("Observed mask missing: run 01_prepare_data.py first.")
        observed = inst["observed"]
    return SCOFCMPipeline(inst["F"], inst["alpha"], inst["beta"], inst["pr_h"], inst["pr_c"],
                          bounds=bounds, observed=observed, **kwargs)


def scalarize(f_norm, weights) -> float:
    """Weighted sum g_w(x) of the normalized objectives."""
    return float(np.dot(np.asarray(weights, dtype=float), np.asarray(f_norm, dtype=float)))
