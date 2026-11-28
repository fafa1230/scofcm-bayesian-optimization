# Held-out validation of the objectives and decisions (5 random 80/20 splits per rater)

- Held-out battles between two of the 30 models: median 3026 (1685 decisive); raters in training 1149.000 to 1149.000
- Main SOBO decision on the training instances: c* [2, 3], sizes ['1002/147', '1028/121', '1040/109', '640/428/81', '979/170']
- Skill of its subgroups (1 - error/pooled error): median 0.003 (-0.005 to 0.005); random partitions of the same sizes: mean -0.0007, max 0.0044 over 200 per split
- Skill on battles of the smallest subgroup (7.6% of held-out battles): 0.001; on the others: 0.003
- Spearman over the 512 pilot configurations, objective vs skill (negative = better objective, better skill): g -0.72 (-0.762 to -0.632), f1 -0.76, f2 -0.72, f3 0.05
- Median skill of the best 10% configurations by g: 0.003; of the rest: 0.000
- Accuracy on decisive held-out battles (battle-averaged / rater-averaged): consensus 0.620 / 0.614; flat 0.619 / 0.613; Bradley-Terry 0.626 / 0.622
- Kendall tau, consensus vs Bradley-Terry on the training battles: median 0.775
