# Analysis v2 Phase 2: Population Shift vs Objective — Results

Date: 2026-04-05

## Method

For each of 147 experiment rows across 10 datasets, computed population shift properties between rows with extended features (ext_pop) and rows without (no_ext_pop):

- **Label rate gap** — |positive_rate_ext - positive_rate_no_ext|
- **Mean KS statistic** — average Kolmogorov-Smirnov distance on numeric base features
- **Mean Cohen's d** — average standardized mean difference on base features
- **% features shifted** — fraction of base features with KS p < 0.05
- **Chi-squared** — test of label × has_extended independence

## Results

### Cross-Dataset Correlations

**No significant correlations found.**

| Property | Pearson r | p-value | Spearman r | p-value |
|---|---|---|---|---|
| Label rate gap | +0.121 | 0.143 | -0.005 | 0.954 |
| Label rate ratio | -0.045 | 0.587 | -0.024 | 0.771 |
| Chi-squared | +0.062 | 0.454 | +0.045 | 0.588 |
| Mean KS statistic | +0.046 | 0.582 | +0.143 | 0.085 |
| Max KS statistic | +0.057 | 0.490 | +0.096 | 0.248 |
| Mean Cohen's d | +0.053 | 0.525 | +0.150 | 0.069 |
| % features shifted | -0.034 | 0.684 | +0.132 | 0.110 |
| Combined shift | +0.100 | 0.228 | +0.110 | 0.184 |

All |r| < 0.15, no p < 0.05.

### Binned Analysis

**By label rate gap:** No monotonic pattern. Gap 0.03-0.05 has the best mean objective (-0.0017) but gap 0.10+ has the worst (+0.010).

**By mean KS:** The 0.02-0.05 range has the highest win rate (58%) but the pattern is not consistent.

**By % features shifted:** No clear pattern. Both low-shift (0-20%, 47% negative) and high-shift (80-100%, 38% negative) groups have similar win rates.

### Per-Dataset Summary

| Dataset | Label Gap | Mean KS | % Shifted | Mean Objective |
|---|---|---|---|---|
| FlightDelay | 0.002 | 0.007 | 5.6% | +0.001 |
| ClientRecordV2 | 0.014 | 0.017 | 5.1% | -0.001 |
| MovieAugV2 | 0.004 | 0.019 | 18.7% | +0.002 |
| ClientRecordAug | 0.029 | 0.020 | 19.0% | -0.000 |
| BankLoanSta | 0.038 | 0.030 | 50.2% | -0.003 |
| DiabetesRecord | 0.028 | 0.076 | 96.7% | +0.002 |
| WeatherAUS | 0.024 | 0.078 | 100.0% | +0.003 |
| WIDS | 0.051 | 0.094 | 91.4% | +0.001 |
| HRAnalytics | 0.161 | 0.104 | 75.0% | +0.007 |
| Weather | 0.023 | 0.143 | 87.9% | +0.000 |

Interestingly, datasets with the LEAST covariate shift (FlightDelay, ClientRecordV2 — augmented with random nulls) tend to have better objectives. This is the **opposite** of our hypothesis. Possible explanation: augmented datasets inject nulls randomly (no covariate shift by design), and these datasets tend to produce cleaner incremental learning results because the populations are exchangeable.

## Conclusions

1. **Population shift (label or covariate) does not predict the objective.** Neither label rate differences nor base feature distribution differences between ext and no-ext populations correlate with incremental learning success.

2. **The initial hypothesis was wrong.** We hypothesized that more different populations would benefit more from specialization. Instead, the data suggests the opposite trend (though not significant): random/exchangeable populations (low shift, augmented datasets) tend toward better incremental results.

3. **This is consistent with Phase 1 findings.** The objective is not explained by surface-level properties of the data split (feature importance, population differences, null rates, dataset size).

4. **What we haven't measured:** The objective may depend on properties of the XGBoost optimization landscape — how well the warm-started model converges vs training from scratch, which is specific to the hyperparameter search trajectory and tree structure, not to population-level statistics.

## Data
Raw computed properties saved to `analysis/phase2_shift_properties.csv`.
