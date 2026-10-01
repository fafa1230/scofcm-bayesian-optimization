"""Section 2.5 / Algorithm 1: single-objective (SOBO) and multi-objective (MOBO) Bayesian optimization."""
from __future__ import annotations

import time
import warnings
from typing import Callable

import numpy as np
import torch
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.exceptions.errors import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood

import config
from src.scofcm import scalarize

TKW = {"dtype": torch.double, "device": torch.device("cpu")}
BOUNDS = torch.stack([torch.zeros(config.DIM, **TKW), torch.ones(config.DIM, **TKW)])

# BoTorch recommends the log-variants of EI/EHVI; the paper uses the original
# formulations (Garnett, 2023; Daulton et al., 2020), so these hints are silenced.
warnings.filterwarnings("ignore", message=".*has known numerical issues.*")
warnings.filterwarnings("ignore", category=UserWarning, module="botorch")


def sobol_design(seed: int, n: int = config.N_INIT) -> np.ndarray:
    """Scrambled Sobol initial design in [0, 1]^3, shared by all runs of a seed."""
    engine = torch.quasirandom.SobolEngine(dimension=config.DIM, scramble=True, seed=seed)
    return engine.draw(n, dtype=torch.double).numpy()


def _gp(X: torch.Tensor, y: torch.Tensor) -> SingleTaskGP:
    """GP with constant mean, Matern-5/2 ARD kernel and an inferred homoskedastic noise term.

    Phi is deterministic but not smooth (c* and iteration counts jump), so the noise term
    is estimated from the data instead of being fixed near zero.
    """
    covar = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=X.shape[-1]))
    return SingleTaskGP(X, y, covar_module=covar, outcome_transform=Standardize(m=1))


def _optimize(acq) -> np.ndarray:
    cand, _ = optimize_acqf(acq, bounds=BOUNDS, q=1,
                            num_restarts=config.NUM_RESTARTS, raw_samples=config.RAW_SAMPLES)
    return cand.detach().squeeze(0).clamp(0.0, 1.0).numpy()


def propose_sobo(X: np.ndarray, Fn: np.ndarray, weights) -> np.ndarray:
    """Fit a GP to the weighted sum g_w and maximize EI (BoTorch maximizes, so y = -g_w)."""
    Xt = torch.tensor(X, **TKW)
    y = -torch.tensor([scalarize(f, weights) for f in Fn], **TKW).unsqueeze(-1)
    model = _gp(Xt, y)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    acq = ExpectedImprovement(model, best_f=y.max())
    return _optimize(acq)


def propose_mobo(X: np.ndarray, Fn: np.ndarray) -> np.ndarray:
    """Fit one GP per normalized objective and maximize EHVI w.r.t. the reference point."""
    Xt = torch.tensor(X, **TKW)
    Y = -torch.tensor(Fn, **TKW)                                   # maximize -f
    model = ModelListGP(*[_gp(Xt, Y[:, i:i + 1]) for i in range(Y.shape[1])])
    fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
    ref = -torch.tensor(config.REF_POINT, **TKW)
    partitioning = FastNondominatedPartitioning(ref_point=ref, Y=Y)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([config.MC_SAMPLES]))
    acq = qExpectedHypervolumeImprovement(model=model, ref_point=ref,
                                          partitioning=partitioning, sampler=sampler)
    return _optimize(acq)


def run_bo(evaluate: Callable, mode: str, X_init: np.ndarray, seed: int,
           weights=None, budget: int = config.BUDGET) -> list:
    """Algorithm 1. `evaluate(x)` returns a PipelineResult. Returns one record per evaluation."""
    if mode not in ("SOBO", "MOBO"):
        raise ValueError(mode)
    torch.manual_seed(seed)
    fallback = torch.quasirandom.SobolEngine(config.DIM, scramble=True, seed=10_000 + seed)
    records, X, Fn = [], [], []

    def add(x, phase, t_acq):
        t0 = time.perf_counter()
        res = evaluate(x)
        t_eval = time.perf_counter() - t0
        X.append(np.asarray(x, dtype=float))
        Fn.append(res.f_norm)
        records.append({
            "iter": len(X), "phase": phase, "x": [float(v) for v in x],
            "f_raw": [None if not np.isfinite(v) else float(v) for v in res.f_raw],
            "f_norm": [float(v) for v in res.f_norm],
            "c_star": int(res.c_star), "t_eval": t_eval, "t_acq": t_acq,
        })

    for x in X_init:                                              # line 1
        add(x, "init", 0.0)
    while len(X) < budget:                                         # lines 2-6
        t0 = time.perf_counter()
        phase = "bo"
        try:
            if mode == "SOBO":
                x_next = propose_sobo(np.array(X), np.array(Fn), weights)
            else:
                x_next = propose_mobo(np.array(X), np.array(Fn))
        except (ModelFittingError, RuntimeError, ValueError):
            # Rare numerical failure of the GP fit: evaluate a quasi-random point instead,
            # and record it so the number of fallbacks can be reported.
            x_next = fallback.draw(1, dtype=torch.double).numpy()[0]
            phase = "bo_fallback"
        add(x_next, phase, time.perf_counter() - t0)
    return records
