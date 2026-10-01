# Decisions (Section III-A-3 and the Discussion)

## SOBO and MOBO decisions (20 seeds)
- c*: SOBO [2], MOBO [2]
- Same partition and ranking: 17 of 20 seeds; minimum ARI 0.69, minimum Kendall tau 0.995
- zeta: SOBO median 0.484 (0.471 to 0.486), MOBO median 0.484 (0.391 to 0.841)
- kappa: SOBO 0.00-0.64, MOBO 0.00-0.62
- mu: SOBO 0.00-1.00, MOBO 0.00-1.00
- SOBO subgroup sizes: ['1074/75']; weights: ['0.930/0.070', '0.931/0.069', '0.932/0.068', '0.933/0.067', '0.934/0.066', '0.935/0.065']
- Kendall tau between the two SOBO subgroups (real votes): -0.195 to -0.195
- Kendall tau between the SOBO consensus and the flat ranking: 1.000 to 1.000

## Effect of mu at zeta = 0.484, kappa = 0.5
- mu = 0: c* = 2, weights 0.935/0.065, f2 = 0.02882
- mu = 1: c* = 2, weights 0.930/0.070, f2 = 0.02882

## Zeta profile (kappa = mu = 0.5)
- g_E < 0.15 only for zeta in [0.40, 0.50]
- lowest g_E = 0.092 at zeta = 0.40; g_E(0.40) = 0.092, g_E(0.39) = 0.443

## Configurations in the MOBO Pareto sets compared with the SOBO decision of the same seed
- main: 174 configurations; different partition 74% (c* = 3 in 21); different ranking 51% (minimum Kendall tau 0.83)
- imputed: 204 configurations; different partition 81% (c* = 3 in 36); different ranking 0% (minimum Kendall tau 1.00)
