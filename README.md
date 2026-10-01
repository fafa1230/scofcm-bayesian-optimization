# SOBO vs MOBO for the balancing parameters of SCO-FCM

Code, processed data, and experimental logs for the paper

> I. Lazulfa, T. Widiyaningtyas, and D. D. Prasetya, "Comparing Single- and Multi-Objective Bayesian Optimization for
> Soft-Constrained Fuzzy Clustering in Large-Scale Group Decisions," manuscript, 2026.

The study tunes the three balancing parameters (ζ, κ, μ) of the soft-constrained fuzzy clustering
method SCO-FCM for large-scale group decision making (LSGDM), using 1,149 LMArena raters and
30 models, and compares single-objective Bayesian optimization (SOBO, GP + EI) with
multi-objective Bayesian optimization (MOBO, GP + EHVI) over 20 paired replications.

## Setup (once)

```bash
git clone https://github.com/fafa1230/scofcm-bayesian-optimization.git
cd scofcm-bayesian-optimization
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

In VS Code: *Python: Select Interpreter* → `.venv/bin/python`.

## Run

| Step | Command | What it does | Paper |
|---|---|---|---|
| 0 | `python tests/test_pipeline.py` | Sanity checks on synthetic data | – |
| 1 | `python 01_prepare_data.py` | Downloads the LMArena votes, builds F, HBC, CRC, PageRank and the pilot bounds; prints Table 1 | II-B, II-D |
| 2 | `python 02_run_experiment.py --quick` | Smoke test: seed 1, budget 12 | – |
| 2 | `python 02_run_experiment.py --jobs 4` | Full experiment: 20 seeds × 5 runs × 60 evaluations (about 23 min with 4 workers on an M4 Pro) | II-E |
| 3 | `python 03_analyze.py` | RQ1–RQ3 metrics, tests, Figure 2 → `results/analysis/` | II-F, III-A |
| 4 | `python 05_diagnostics.py` | Pareto-front size, (κ, μ) grid, ζ profile (Figure 3), time break-even → `results/analysis/diagnostics.md` | III-A, III-B |
| 5 | `python 06_sanity_benchmark.py` | SOBO vs MOBO on DTLZ2 and on a problem with aligned objectives → `results/analysis/tables/sanity_benchmark.csv` | II-G, III-A |
| 6 | `python 07_decision_analysis.py` | The selected decisions: subgroup sizes and weights, agreement with the flat ranking, effect of μ, ζ band, and how many MOBO Pareto configurations change the partition or ranking (main and imputed-score runs) → `results/analysis/decisions.md` | III-A-3, III-B |
| – | `python 04_verify_baseline.py` | Optional: reproduces the case study of Yang et al. (needs their matrices in `data/yang_case/*.csv`, not included) | – |

`02_run_experiment.py` skips finished runs, so it can be stopped and restarted. Steps 3–6 can be run
directly on the logs in this repository without repeating step 2 (step 5 does not use the LMArena
data and takes about 10 minutes).

## Where each part of the method lives

| Paper | Code |
|---|---|
| Table 2 (all settings) | `config.py` |
| II-B data, P-HFE score (1), Jaccard relations (2), PageRank | `src/data.py` |
| II-C SCO-FCM Stages 1–5, (3)–(8) | `src/scofcm.py` (`SCOFCMPipeline`) |
| II-D objectives f1–f3 and normalization, (9)–(12) | `src/scofcm.py` (`evaluate`, `normalize`) |
| II-E SOBO (GP + EI) and MOBO (GP + EHVI) | `src/optimizers.py` |
| II-F hypervolume (13), weight-coverage gap (14), A12, Holm | `src/metrics.py`, `03_analyze.py` |
| Figure style (Times New Roman 8 pt, printed size) | `src/plotstyle.py` |

## Data and results in this repository

- `data/processed/instance.npz`: processed rater-profile matrix *F* (1,149 × 30), observed-entry mask,
  HBC and CRC relations, PageRank centralities, and the pilot normalization bounds of (12).
- `data/processed/table1_stats.json`: the numbers in Table 1 and a summary of the pilot sample.
- `results/logs/seed{S}_{RUN}.jsonl`: every evaluation of the main experiment (x, objectives, c*, timings).
- `results/analysis/`: `summary.md` (RQ1–RQ3), `diagnostics.md`, `decisions.md`, `tables/*.csv`, and the figures.
- `results/archive_imputed_scores/` and `data/processed/archive_imputed_scores/`: the sensitivity run in
  which subgroup scores were averaged over all members, including imputed entries (Section III-A-4).

The raw votes are not redistributed here; `01_prepare_data.py` downloads only the needed columns of
[lmarena-ai/arena-human-preference-100k](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-100k)
(June–August 2024) into `data/raw/`. Only vote metadata (models, outcome, anonymized rater identifier,
timestamp, language) is used, not prompts or model responses. Please cite the dataset and
W.-L. Chiang *et al.*, "Chatbot Arena: An open platform for evaluating LLMs by human preference,"
ICML 2024, when using the data.

## Implementation notes

1. **Coordination cost (5).** `CC_MODE = "mean"` uses the average unreachability to the
   members of a subgroup. The literal sum of the baseline makes an emptied subgroup cost-free,
   so it absorbs every rater on the next iteration and FCM cycles without converging
   (`tests/test_pipeline.py`, `test_cost_modes`). Set `CC_MODE = "sum"` to use the literal form.
2. **Normalization (12).** `NORMALIZATION = "pilot"`: min-max bounds from 512 scrambled Sobol
   configurations (seed 0), computed once in `01_prepare_data.py`. With the theoretical bounds,
   the observed ranges of f2 (0.011-0.028) and f3 (130-272) are tiny, and f1 explains 81% of the
   variance of the equally weighted sum; with the pilot bounds the shares are 38%, 43% and 18%.
3. **GP noise.** Phi is deterministic but jumps when c* or iteration counts change, so each GP
   estimates a homoskedastic noise term. If a fit still fails, a quasi-random point is evaluated
   (logged as `bo_fallback` and counted in the summary).
4. **Nearest subgroup in the silhouette (7)** is the one whose centre is closest to the rater,
   as in the baseline.
5. **RQ2 correlations** are computed on the normalized objectives (all minimized), so a negative
   Spearman's rho means a genuine conflict.
6. Evaluations of the shared initial design are cached within a seed, so wall-clock comparisons
   use the BO iterations only.
7. **Subgroup scores (Stage 5, (10)).** `SUBGROUP_SCORES = "observed"`: a subgroup's score on a
   model is the mean over the members who actually rated it; imputed values are used only for the
   clustering distances. The first run used `"imputed"` (all members, including imputed entries);
   its logs and analysis are kept in `results/archive_imputed_scores/` as a sensitivity check.

Tested with Python 3.12, torch 2.14, BoTorch 0.12.0, GPyTorch 1.13, pymoo 0.6.2 on macOS 26.5 (Apple M4 Pro).

## License

Code: MIT (see `LICENSE`). The processed data are derived from the LMArena dataset; follow its terms
when reusing them.
