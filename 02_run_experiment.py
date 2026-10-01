"""Step 2 (Section 2.5): run SOBO and MOBO for every seed and log every evaluation.

For each seed the five runs of config.RUNS share one Sobol initial design.
Logs are written to results/logs/seed{S}_{RUN}.jsonl; finished runs are skipped,
so the script can be stopped and restarted.

Usage:
    python 02_run_experiment.py --jobs 6                 # full experiment (20 seeds)
    python 02_run_experiment.py --quick                  # smoke test: 1 seed, budget 12
    python 02_run_experiment.py --seeds 1 2 3 --jobs 3
"""
import warnings

# PyTorch prints a deprecation notice for torch.jit.script when BoTorch/GPyTorch load;
# it does not affect the results.
warnings.filterwarnings("ignore", message=".*torch.jit.script.*", category=FutureWarning)
import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

import config

WORKER = {}


def _init_worker(threads: int):
    import torch
    from src.data import load_instance
    from src.scofcm import pipeline_from_instance

    torch.set_num_threads(threads)
    WORKER["pipe"] = pipeline_from_instance(load_instance())
    WORKER["cache"] = {}


def _evaluate(x):
    key = tuple(np.round(np.asarray(x, dtype=float), 12))
    cache = WORKER["cache"]
    if key not in cache:
        cache[key] = WORKER["pipe"].evaluate(key)
    return cache[key]


def run_seed(task):
    from src.optimizers import run_bo, sobol_design

    seed, log_dir, budget, n_init, runs = task
    X_init = sobol_design(seed, n_init)
    done = []
    for name, mode, wkey in runs:
        path = Path(log_dir) / f"seed{seed:02d}_{name}.jsonl"
        if path.exists() and sum(1 for _ in open(path)) >= budget:
            continue
        t0 = time.perf_counter()
        weights = config.WEIGHTS[wkey] if wkey else None
        records = run_bo(_evaluate, mode, X_init, seed, weights=weights, budget=budget)
        with open(path, "w") as fh:
            for r in records:
                r.update({"seed": seed, "run": name, "mode": mode, "weights": wkey})
                fh.write(json.dumps(r) + "\n")
        done.append(f"seed {seed} {name}: {time.perf_counter() - t0:.0f} s")
    WORKER["cache"].clear()
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=list(config.SEEDS))
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    ap.add_argument("--quick", action="store_true", help="smoke test: seed 1, budget 12")
    args = ap.parse_args()

    budget, n_init, seeds = config.BUDGET, config.N_INIT, args.seeds
    log_dir = config.LOGS
    if args.quick:
        budget, seeds, log_dir = 12, [1], config.RESULTS / "logs_quick"
    log_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(s, str(log_dir), budget, n_init, config.RUNS) for s in seeds]
    threads = max(1, (mp.cpu_count() or 1) // max(args.jobs, 1))
    t0 = time.perf_counter()
    if args.jobs == 1:
        _init_worker(threads)
        for t in tasks:
            for line in run_seed(t):
                print(line, flush=True)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.jobs, initializer=_init_worker, initargs=(threads,)) as pool:
            for lines in pool.imap_unordered(run_seed, tasks):
                for line in lines:
                    print(line, flush=True)
    print(f"Finished in {(time.perf_counter() - t0) / 60:.1f} min. Logs in {log_dir}")


if __name__ == "__main__":
    main()
