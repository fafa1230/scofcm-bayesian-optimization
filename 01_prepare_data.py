"""Step 1 (Sections 2.2 and 2.4): build the LSGDM instance from LMArena, report Table 1,
and estimate the normalization bounds of Eq. 12 from a separate pilot sample.

Usage:
    python 01_prepare_data.py                       # reads only the needed columns remotely
    python 01_prepare_data.py --local path.parquet  # use a downloaded copy of the dataset
"""
import warnings

# PyTorch prints a deprecation notice for torch.jit.script when BoTorch/GPyTorch load;
# it does not affect the results.
warnings.filterwarnings("ignore", message=".*torch.jit.script.*", category=FutureWarning)
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import config
from src.data import build_instance, load_battles, pilot_bounds, save_instance
from src.scofcm import SCOFCMPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", type=Path, default=None, help="local copy of the parquet file")
    args = ap.parse_args()

    t0 = time.perf_counter()
    df = load_battles(args.local)
    print(f"Loaded {len(df):,} battles in {time.perf_counter() - t0:.1f} s")

    inst = build_instance(df)
    stats = inst.pop("stats")
    print("\nTable 1 values")
    for k, v in stats.items():
        print(f"  {k:32s} {v}")

    # Pilot sample for the normalization bounds (Eq. 12); never given to the optimizers
    observed = inst["observed"] if config.SUBGROUP_SCORES == "observed" else None
    pipe = SCOFCMPipeline(inst["F"], inst["alpha"], inst["beta"], inst["pr_h"], inst["pr_c"],
                          observed=observed)
    t0 = time.perf_counter()
    lo, hi, F_pilot = pilot_bounds(pipe)
    dt = time.perf_counter() - t0
    inst["bounds_lo"], inst["bounds_hi"] = lo, hi
    path = save_instance(inst)

    rho = spearmanr(F_pilot).statistic
    pilot = {
        "size": config.PILOT_SIZE, "seed": config.PILOT_SEED, "valid": int(len(F_pilot)),
        "seconds_per_evaluation": dt / config.PILOT_SIZE,
        "raw_min": lo.tolist(), "raw_max": hi.tolist(),
        "spearman_raw_objectives": np.round(rho, 3).tolist(),
    }
    stats["pilot"] = pilot
    (config.DATA_PROCESSED / "table1_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    print(f"\nPilot ({config.PILOT_SIZE} Sobol configurations, {dt / config.PILOT_SIZE * 1000:.0f} ms each)")
    for name, a, b in zip(("f1 validity", "f2 consensus cost", "f3 iterations"), lo, hi):
        print(f"  {name:18s} min {a:.4f}  max {b:.4f}")
    print(f"Saved instance and bounds to {path}")


if __name__ == "__main__":
    main()
