# Diagnostics
- Non-dominated points per MOBO run: median 8 (range 3-20).
- Distinct objective vectors: 2222 of 6000 evaluations.
- Selected zeta (SOBO, 20 seeds): median 0.484, range 0.471-0.486; kappa range 0.00-0.64; mu range 0.00-1.00.

## Objectives over 121 (kappa, mu) pairs at zeta = 0.484
Rows per c*: c*=2: 88, c*=4: 22, c*=8: 11
```
           f1n             f2n             f3n             g_E        
           min     max     min     max     min     max     min     max
c_star                                                                
2       0.0000  0.0000  0.1717  0.1717  0.0563  0.0563  0.0760  0.0760
4       0.5155  0.5155  0.5079  0.5088  0.0563  0.0563  0.3599  0.3602
8       0.7180  0.7180  0.8997  0.9008  0.0563  0.0563  0.5580  0.5584
```

## Cost
- Median acquisition time per iteration: SOBO 0.22 s, MOBO 2.42 s; median evaluation of Phi 58 ms.
- Median acquisition time per seed: MOBO (1 run) 186.6 s, SOBO (4 runs) 56.1 s.
- Covering the four weightings costs MOBO 52 BO evaluations vs 208 for SOBO; MOBO is faster overall once one evaluation of Phi takes more than 0.84 s.
