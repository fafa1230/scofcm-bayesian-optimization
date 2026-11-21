# Synthetic problems with controlled conflict (10 paired replications, 40 evaluations)

| lambda | pilot min rho | HV SOBO median [IQR] | HV MOBO median [IQR] | MOBO larger HV | p (Wilcoxon) | best g SOBO / MOBO | Pareto points SOBO / MOBO | acquisition s SOBO / MOBO |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 1.00 | 1.316 [1.306, 1.319] | 1.328 [1.327, 1.328] | 10/10 | 0.00195 | 0.004 / 0.001 | 1 / 1 | 6 / 40 |
| 0.2 | 0.63 | 1.047 [1.043, 1.048] | 1.062 [1.061, 1.066] | 10/10 | 0.00195 | 0.096 / 0.098 | 13 / 20 | 5 / 47 |
| 0.3 | 0.25 | 0.943 [0.938, 0.946] | 0.966 [0.965, 0.968] | 10/10 | 0.00195 | 0.142 / 0.145 | 14 / 21 | 4 / 32 |
| 0.4 | -0.08 | 0.854 [0.850, 0.867] | 0.888 [0.887, 0.893] | 10/10 | 0.00195 | 0.187 / 0.194 | 18 / 26 | 4 / 27 |
| 0.5 | -0.32 | 0.789 [0.788, 0.794] | 0.831 [0.830, 0.832] | 10/10 | 0.00195 | 0.232 / 0.242 | 19 / 28 | 4 / 23 |
| 0.7 | -0.55 | 0.788 [0.758, 0.798] | 0.873 [0.872, 0.874] | 10/10 | 0.00195 | 0.293 / 0.294 | 12 / 30 | 4 / 16 |
| 1.0 | -0.61 | 0.832 [0.784, 0.884] | 0.988 [0.986, 0.994] | 10/10 | 0.00195 | 0.267 / 0.267 | 24 / 24 | 6 / 16 |

- Spearman, pilot rho vs HV difference over 70 replications: -0.90 (p = 3.96e-26)
- Negative pilot rho: MOBO larger HV in 40 of 40; non-negative: 30 of 30; one-sided Fisher exact p = 1
