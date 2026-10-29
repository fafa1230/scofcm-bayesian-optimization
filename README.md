# SOBO vs MOBO for the balancing parameters of SCO-FCM

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23725652.svg)](https://doi.org/10.5281/zenodo.23725652)

Code, processed data, and experimental logs for the paper

> I. Lazulfa, T. Widiyaningtyas, and D. D. Prasetya, "Comparing Single- and Multi-Objective Bayesian Optimization for
> Soft-Constrained Fuzzy Clustering in Large-Scale Group Decisions," manuscript, 2026.

Archived version used in the paper: v1.0.0, doi:[10.5281/zenodo.23725653](https://doi.org/10.5281/zenodo.23725653)
(the badge above always points to the latest version).

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
| 1 | `python 01_prepare_data.py` | Downloads the LMArena votes, builds F, HBC, CRC, PageRank and the pilot bounds; prints Table 1 | 2.2, 2.4 |
| 2 | `python 02_run_experiment.py --quick` | Smoke test: seed 1, budget 12 | – |
| 2 | `python 02_run_experiment.py --jobs 4` | Full experiment: 20 seeds × 5 runs × 60 evaluations (about 23 min with 4 workers on an M4 Pro) | 2.5 |
| 3 | `python 03_analyze.py` | RQ1–RQ3 metrics, tests, Figure 2 → `results/analysis/` | 2.6, 3.1 |
| 4 | `python 05_diagnostics.py` | Pareto-front size, (κ, μ) grid, ζ profile in steps of 0.01, time break-even → `results/analysis/diagnostics.md` | 3.1, 3.2 |
| 5 | `python 06_sanity_benchmark.py` | SOBO vs MOBO on DTLZ2 and on a problem with aligned objectives → `results/analysis/tables/sanity_benchmark.csv` | 2.7, 3.1 |
| 6 | `python 07_decision_analysis.py` | The selected decisions: subgroup sizes and weights, agreement with the flat ranking, effect of μ, ζ band, and how many MOBO Pareto configurations change the partition or ranking (main and imputed-score runs) → `results/analysis/decisions.md` | 3.1.3, 3.2 |
| 7 | `python 08_baseline_comparison.py` | Gain of the SOBO and MOBO settings over the baseline setting (ζ, κ, μ) = (0.4, 0.6, 0.5): one-sample Wilcoxon (Holm), objectives and decisions → `results/analysis/baseline_comparison.md` | 2.6, 3.1.1 |
| 8 | `python 09_random_search.py` | Random search with the same initial design and budget; 95% bootstrap intervals of the MOBO − SOBO differences (Table 4) → `results/analysis/random_search.md` | 2.6, 3.1.1 |
| 9 | `python 10_landscape.py` | ζ profile in steps of 0.001 with the SOBO selections (Figure 3) and Sobol sensitivity indices of the objectives (about 3 min) → `results/analysis/landscape.md` | 3.1.3, 3.2.1 |
| 10 | `python 11_scaling.py` | Evaluation time of Φ for 1,149 to 9,192 resampled raters (needs about 6 GB of memory) → `results/analysis/scaling.md` | 3.1.4, 3.2.2 |
| 11 | `python 12_bradley_terry.py` | Consensus ranking vs Bradley–Terry rankings of the same battles (retained and all raters), Kendall's τ with bootstrap intervals → `results/analysis/bradley_terry.md` | 2.6, 3.1.3 |
| 12 | `python 13_sensitivity.py --jobs 6` | Nine data variants (5 or 20 votes per rater, two tie codings, five 80% rater subsamples), five paired SOBO/MOBO replications each (about 40 min) → `results/sensitivity/summary.md` (Table 7) | 2.7, 3.1.4 |
| – | `python 04_verify_baseline.py` | Optional: reproduces the case study of Yang et al. (needs their matrices in `data/yang_case/*.csv`, not included) | – |

`02_run_experiment.py` skips finished runs, so it can be stopped and restarted. Steps 3–11 can be run
directly on the logs in this repository without repeating step 2 (step 5 does not use the LMArena
data and takes about 10 minutes).

## Where each part of the method lives

| Paper | Code |
|---|---|
| Table 2 (all settings) | `config.py` |
| 2.2 data, P-HFE score (1), Jaccard relations (2), PageRank | `src/data.py` |
| 2.3 SCO-FCM pipeline (objective function), Stages 1–5, (3)–(8) | `src/scofcm.py` (`SCOFCMPipeline`) |
| 2.4 objectives f1–f3 and normalization, (9)–(12); constraints (K1)–(K4) and the MOBO/SOBO problems (13)–(14) | `src/scofcm.py` (`evaluate`, `normalize`), `config.py` |
| 2.5 SOBO (GP + EI, (15)) and MOBO (GP + EHVI, (16)); decisions x_SOBO and x_MOBO | `src/optimizers.py`, `src/metrics.py` (`compromise_index`) |
| 2.6 hypervolume (17), weight-coverage gap (18), A12, Holm | `src/metrics.py`, `03_analyze.py` |
| Figure style (Times New Roman 8 pt, printed size) | `src/plotstyle.py` |

## Data and results in this repository

- `data/processed/instance.npz`: processed rater-profile matrix *F* (1,149 × 30), observed-entry mask,
  HBC and CRC relations, PageRank centralities, and the pilot normalization bounds of (12).
- `data/processed/table1_stats.json`: the numbers in Table 1 and a summary of the pilot sample.
- `results/logs/seed{S}_{RUN}.jsonl`: every evaluation of the main experiment (x, objectives, c*, timings).
- `results/analysis/`: `summary.md` (RQ1–RQ3), `diagnostics.md`, `decisions.md`, `tables/*.csv`, and the figures.
- `results/archive_imputed_scores/` and `data/processed/archive_imputed_scores/`: the sensitivity run in
  which subgroup scores were averaged over all members, including imputed entries (Section 3.1.4).

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
4. **Nearest subgroup in the silhouette (7)** is the one whose center is closest to the rater,
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
