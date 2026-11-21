# Small subgroup and weight coverage (revision)

## Part 1: the two subgroups of the SOBO decision
- Subgroup of 1074 raters: votes per rater median 14 (IQR 11-21); ties 41.5% (tie 20.7%, bothbad 20.8%); rater tie share min 0%, median 40%; languages English 53.5%, Russian 12.6%, Chinese 8.9%, Japanese 5.7%, Korean 4.2%; votes in both periods 43% of raters; profile density 50.4%; prompts: median 55 user tokens, multi-turn 14.8%, high-frequency 10.6%, hard 25.7%
- Subgroup of 75 raters: votes per rater median 12 (IQR 11-18); ties 96.9% (tie 68.4%, bothbad 28.5%); rater tie share min 20%, median 100%; languages English 36.6%, unknown 21.6%, Russian 8.9%, Japanese 7.0%, Spanish 5.9%; votes in both periods 32% of raters; profile density 47.6%; prompts: median 38 user tokens, multi-turn 4.7%, high-frequency 46.4%, hard 12.4%
- English share of votes: large 53.5% vs small 36.6% (chi-square p = 1.73e-34)
- Votes per rater, large vs small: Mann-Whitney p = 0.0751
- User tokens per prompt, large vs small: medians 55 vs 38, Mann-Whitney p = 2.08e-62
- high_freq prompts, large vs small: 10.6% vs 46.4% (chi-square p = 0)
- hard prompts, large vs small: 25.7% vs 12.4% (chi-square p = 5.42e-29)
- Ties among all retained raters' votes: high-frequency prompts 67.0%, others 41.4%; prompts of at most 5 user tokens 71.0%, longer 42.9%
- Kendall tau between the two subgroup rankings: -0.20, 95% bootstrap interval over the small subgroup's raters [-0.28, -0.02]
- Decisive votes of the large subgroup between two of the 30 models: 8370; agreeing with the large subgroup's ranking: 62.8%
- Decisive votes of the small subgroup between two of the 30 models: 13; agreeing with the large subgroup's ranking: 30.8%

## Part 2: weight-coverage gaps (Eq. 18) against the run-to-run spread of SOBO
- w_V: gap median +0.0007 [95% CI -0.0077, +0.0092], upper quartile +0.0123, max +0.1651; relative to the SOBO optimum median +1.6% (max +362.0%); MOBO at least as good in 10/20; run-to-run spread of SOBO median 0.0099 (upper quartile 0.0750); gap within the median spread in 70% of seeds
- w_C: gap median -0.0012 [95% CI -0.0122, +0.0073], upper quartile +0.0078, max +0.0917; relative to the SOBO optimum median -1.0% (max +80.3%); MOBO at least as good in 10/20; run-to-run spread of SOBO median 0.0113 (upper quartile 0.0279); gap within the median spread in 80% of seeds
- w_T: gap median +0.0042 [95% CI -0.0047, +0.0232], upper quartile +0.0264, max +0.0967; relative to the SOBO optimum median +9.3% (max +177.7%); MOBO at least as good in 9/20; run-to-run spread of SOBO median 0.0169 (upper quartile 0.0296); gap within the median spread in 65% of seeds
