# Objective landscape

## Zeta profile (step 0.001, kappa = mu = 0.5)
- Band with the selected two-subgroup partition (contiguous, around the SOBO selections): zeta in [0.391, 0.506]; 657 of 1001 grid values give two subgroups overall
- SOBO selections: zeta 0.471-0.486, g 0.0619-0.0854; distance to the band edges at least 0.080 (lower) and 0.020 (upper)
- Baseline setting: g = 0.0925, f3 = 145; zeta = 0.4 lies 0.009 above the lower edge
- Inside the band: g from 0.0619 to 0.2310, f3 from 132 to 204 iterations; distinct values of f1: 3, of f2: 3
- Within 0.005 of a SOBO selection: g from 0.0619 to 0.1934, f3 from 132 to 188

## Sobol indices (2560 evaluations; 95% bootstrap intervals)
- g_wE: zeta S1 = 0.42 [0.18, 0.66], ST = 0.53 [0.46, 0.59]; kappa S1 = 0.49 [0.19, 0.78], ST = 0.64 [0.55, 0.74]; mu S1 = 0.00 [-0.00, 0.00], ST = 0.00 [0.00, 0.00]
- f1_norm: zeta S1 = 0.38 [0.22, 0.55], ST = 0.47 [0.41, 0.53]; kappa S1 = 0.60 [0.37, 0.84], ST = 0.68 [0.59, 0.77]; mu S1 = 0.00 [0.00, 0.00], ST = 0.00 [0.00, 0.00]
- f2_norm: zeta S1 = 0.07 [-0.10, 0.26], ST = 0.57 [0.48, 0.67]; kappa S1 = 0.49 [0.23, 0.75], ST = 0.95 [0.85, 1.05]; mu S1 = 0.00 [-0.00, 0.00], ST = 0.00 [0.00, 0.00]
- f3_norm: zeta S1 = 1.02 [0.67, 1.37], ST = 1.04 [0.95, 1.12]; kappa S1 = 0.00 [0.00, 0.00], ST = 0.00 [0.00, 0.00]; mu S1 = 0.00 [0.00, 0.00], ST = 0.00 [0.00, 0.00]
