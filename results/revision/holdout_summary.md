# Held-out validation of the objectives and decisions

## Five random 80/20 splits per rater (missing votes)

- Held-out battles between two of the 30 models: median 3026 (1685 decisive); raters in training 1149.000 to 1149.000
- SOBO re-run on the training instances: x ['0.427/0.558/0.915', '0.331/0.322/0.000', '0.429/0.139/1.000', '1.000/0.466/0.000', '0.346/0.017/0.000'], c* [2], sizes ['1052/97', '1056/93', '987/162', '999/150']
- Skill of its subgroups (1 - error/pooled error): median 0.0027 (0.002 to 0.003); beats all 200 random partitions of the same sizes in 3 of 5; random partitions: mean -0.0006, max 0.0032
- Secondary: configuration of the main SOBO decision (all votes): skill median 0.0028
- Skill on battles of the smallest subgroup (7.0% of held-out battles): 0.005; on the others: 0.002
- Spearman over the 512 pilot configurations, objective vs skill (negative = better objective, better skill): g -0.72 (-0.762 to -0.632), f1 -0.76, f2 -0.72, f3 0.05
- Median skill of the best 10% configurations by g: 0.0027; of the rest: 0.0002
- Accuracy on decisive held-out battles (battle-averaged / rater-averaged): consensus 0.618 / 0.614; flat 0.619 / 0.613; Bradley-Terry 0.626 / 0.622
- Kendall tau, consensus vs Bradley-Terry on the training battles: median 0.770

## Temporal split: each rater's latest 20% of battles (later votes)

- Held-out battles between two of the 30 models: median 2741 (1502 decisive); raters in training 1149.000 to 1149.000
- SOBO re-run on the training instances: x ['0.070/0.100/0.090'], c* [2], sizes ['1025/124']
- Skill of its subgroups (1 - error/pooled error): median 0.0024 (0.002 to 0.002); beats all 200 random partitions of the same sizes in 0 of 1; random partitions: mean -0.0008, max 0.0030
- Secondary: configuration of the main SOBO decision (all votes): skill median 0.0019
- Skill on battles of the smallest subgroup (9.0% of held-out battles): 0.014; on the others: 0.002
- Spearman over the 512 pilot configurations, objective vs skill (negative = better objective, better skill): g -0.76 (-0.757 to -0.757), f1 -0.86, f2 -0.78, f3 0.09
- Median skill of the best 10% configurations by g: 0.0024; of the rest: -0.0023
- Accuracy on decisive held-out battles (battle-averaged / rater-averaged): consensus 0.613 / 0.608; flat 0.615 / 0.610; Bradley-Terry 0.611 / 0.604
- Kendall tau, consensus vs Bradley-Terry on the training battles: median 0.784

