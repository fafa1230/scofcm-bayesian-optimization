# Diagnostics
- Non-dominated points per MOBO run: median 9 (range 5-23).
- Distinct objective vectors: 2399 of 6000 evaluations.
- Selected zeta (SOBO, 20 seeds): median 0.484, range 0.155-0.486; kappa range 0.00-0.67; mu range 0.00-1.00.

## Objectives over 121 (kappa, mu) pairs at zeta = 0.484
Rows per c*: c*=2: 88, c*=4: 22, c*=8: 11
```
           f1n             f2n             f3n             g_E        
           min     max     min     max     min     max     min     max
c_star                                                                
2       0.0000  0.0000  0.1816  0.1816  0.0563  0.0563  0.0793  0.0793
4       0.5155  0.5155  0.6104  0.6123  0.0563  0.0563  0.3941  0.3947
8       0.7185  0.7185  0.9814  0.9816  0.0563  0.0563  0.5854  0.5855
```

## Cost
- Median acquisition time per iteration: SOBO 0.21 s, MOBO 2.30 s; median evaluation of Phi 58 ms.
- Median acquisition time per seed: MOBO (1 run) 146.3 s, SOBO (4 runs) 52.5 s.
- Covering the four weightings costs MOBO 52 BO evaluations vs 208 for SOBO; MOBO is faster overall once one evaluation of Phi takes more than 0.60 s.
