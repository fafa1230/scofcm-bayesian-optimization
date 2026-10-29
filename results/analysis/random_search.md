# Random search (same 8 initial points + 52 uniform points) and RQ1 confidence intervals
- Random search: HV median 1.050 [1.035, 1.065], best g median 0.0913 [0.0854, 0.1018]
- SOBO vs random search, HV: median 1.1046 vs 1.0504; better in 18 of 20 seeds; p = 8.20e-05, p_Holm = 2.46e-04; A12 (SOBO better) = 0.91 (large)
- SOBO vs random search, bestg: median 0.0737 vs 0.0913; better in 18 of 20 seeds; p = 9.54e-06, p_Holm = 3.81e-05; A12 (SOBO better) = 0.96 (large)
- MOBO vs random search, HV: median 1.1232 vs 1.0504; better in 16 of 20 seeds; p = 3.62e-02, p_Holm = 7.25e-02; A12 (MOBO better) = 0.79 (large)
- MOBO vs random search, bestg: median 0.0737 vs 0.0913; better in 15 of 20 seeds; p = 1.07e-01, p_Holm = 1.07e-01; A12 (MOBO better) = 0.77 (large)

## Median paired difference MOBO - SOBO with 95% bootstrap interval
- hypervolume: +0.0011 [-0.0348, +0.0617]
- best weighted sum: +0.0059 [-0.0035, +0.0106]
