"""Experimental settings (Table 2 of the paper). Single source of truth for all scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
LOGS = RESULTS / "logs"

# --- Data (Section 2.2) -------------------------------------------------------
DATASET_URL = (
    "https://huggingface.co/datasets/lmarena-ai/arena-human-preference-100k/"
    "resolve/main/data/arena-explorer-preference-100k.parquet"
)
DATASET_COLUMNS = ["model_a", "model_b", "winner", "judge_hash", "tstamp", "language"]
MIN_VOTES_PER_RATER = 10
N_MODELS = 30
TIE_VALUE = 0.5                      # gamma for "tie" and "tie (bothbad)"
CUTOFF_DATE = "2024-07-15"           # historical / current split (UTC)

# --- PageRank (Section 2.3, Stage 1) ------------------------------------------
PAGERANK_DAMPING = 0.85
PAGERANK_TOL = 1e-6
PAGERANK_MAX_ITER = 100

# --- SCO-FCM (Section 2.3, Stages 2-5) ----------------------------------------
FUZZIFIER = 2.0                      # a
FCM_TOL = 1e-5                       # epsilon on max |delta u|
T_MAX = 300
C_CANDIDATES = tuple(range(2, 9))    # C = {2, ..., 8}
CC_FLOOR = 1e-6                      # constant added to cc to avoid division by zero
# Coordination cost of rater k in subgroup g_r (Eq. 5):
#   "mean": average unreachability, (1/|g_r|) * sum_{s in g_r} (1 - R_ks)   <- default
#   "sum" : sum_{s in g_r} (1 - R_ks), the literal form in Yang et al. (2026).
# With "sum", an emptied subgroup has zero cost and absorbs every rater on the next
# iteration, so FCM cycles and never converges (see tests/test_pipeline.py).
CC_MODE = "mean"
# Subgroup scores in Stage 5 (Eq. 8-10):
#   "observed": mean over the members of g_r who actually rated model a_i   <- default
#   "imputed" : mean over all members, including model-mean imputed entries (first run;
#               imputation pulls every subgroup toward the global model mean)
SUBGROUP_SCORES = "observed"

# --- Fixed evaluation index for f1 (Section 2.4, Eq. 9) -----------------------
KAPPA_EVAL = 0.5
ZETA_EVAL = 0.5

# --- Normalization of the objectives (Section 2.4, Eq. 12) --------------------
# "pilot": min-max bounds from a separate pilot sample of Sobol configurations, evaluated
#          once in 01_prepare_data.py and never given to the optimizers.   <- default
# "theoretical": f1 in [-1, 1], f2 in [0, 1], f3 in [|C|, |C| * T_MAX]. The observed
#          ranges of f2 and f3 are tiny fractions of these, so an equally weighted sum
#          would be driven almost entirely by f1.
NORMALIZATION = "pilot"
PILOT_SIZE = 512
PILOT_SEED = 0

# --- Optimization (Section 2.5) -----------------------------------------------
DIM = 3                              # x = (zeta, kappa, mu) in [0, 1]^3
N_INIT = 8                           # scrambled Sobol points shared by all runs of a seed
BUDGET = 60                          # total evaluations per run (including N_INIT)
SEEDS = tuple(range(1, 21))          # R = 20 replications
WEIGHTS = {                          # scalarization weights g_w (normalized objectives)
    "E": (1 / 3, 1 / 3, 1 / 3),      # main SOBO run
    "V": (0.6, 0.2, 0.2),            # validity-oriented   (RQ2)
    "C": (0.2, 0.6, 0.2),            # consensus-oriented  (RQ2)
    "T": (0.2, 0.2, 0.6),            # convergence-oriented (RQ2)
}
REF_POINT = (1.1, 1.1, 1.1)          # hypervolume reference point (normalized, minimization)
NUM_RESTARTS = 20                    # multi-start L-BFGS-B for acquisition optimization
RAW_SAMPLES = 512
MC_SAMPLES = 128                     # quasi-MC samples for qEHVI

# Runs executed for every seed: (run name, mode, weight key)
RUNS = (
    ("SOBO_E", "SOBO", "E"),
    ("MOBO", "MOBO", None),
    ("SOBO_V", "SOBO", "V"),
    ("SOBO_C", "SOBO", "C"),
    ("SOBO_T", "SOBO", "T"),
)

# --- Statistics (Section 2.6) -------------------------------------------------
ALPHA = 0.05
