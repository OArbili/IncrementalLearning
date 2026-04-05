# Analysis v1: Correlation Between Objective and Dataset/Experiment Properties

Date: 2026-04-05

## Setup

- 153 experiment rows across 10 datasets (129 optuna + 12 no_pruning + 12 fixed_50)
- Objective = (comb+ext AUC - ext AUC) + (comb-no AUC - base AUC)
- Negative objective = incremental learning outperforms combined model

## Properties Tested

### Dataset Properties
- `ext_null_pct` — average null % of extended features
- `n_ext_features` — number of extended features
- `n_rows` — dataset size
- `label_1_pct` — positive class percentage
- `ext_pct` — % of test rows with extended features

### Model Properties
- `base_auc`, `ext_auc`, `comb_no_auc`, `comb_ext_auc`
- `inv_base_auc` (1 - base AUC, room for improvement)
- `ext_minus_base`, `ext_minus_comb`, `base_minus_comb`

### Experiment Properties
- `trials` — number of Optuna trials
- `pruning_mode` — optuna / no_pruning / fixed_50
- `is_augmented` — whether dataset uses artificial null injection

### Interaction Terms
- 16 pairwise interactions tested (e.g., label_1_pct × ext_pct, inv_base_auc × ext_null_pct)

---

## Results

### Cross-Dataset Correlations

No strong predictors found. The two weakly significant results:

| Predictor | Pearson r | p-value |
|---|---|---|
| ratio_ext_no (ext/no_ext rows) | -0.263 | 0.001 |
| ratio_ext_no × inv_base_auc | -0.248 | 0.002 |

All other single features and interactions: |r| < 0.18, p > 0.05.

Multivariate regression (9 non-AUC features): **R² = 0.019** — essentially zero explanatory power.

### Objective Decomposition

The objective has two approximately equal components:
- Term1 (comb+ext - ext AUC): r=0.895 with objective, 30% of variance
- Term2 (comb-no - base AUC): r=0.803 with objective, 36% of variance

These are mathematically related to the objective (they ARE the objective), so the high correlations are tautological.

### Binned Analysis

**By null %:**

| Null % Range | n | Mean Objective | % Negative |
|---|---|---|---|
| 0-20% | 23 | -0.001769 | 56.5% |
| 20-40% | 22 | +0.000892 | 59.1% |
| 40-60% | 59 | -0.000070 | 40.7% |
| 60-80% | 14 | +0.001558 | 21.4% |
| 80-100% | 35 | +0.000941 | 34.3% |

Weak trend: lower null rates tend toward better objectives, but not monotonic.

**By base AUC:**

| Base AUC Range | n | Mean Objective | % Negative |
|---|---|---|---|
| 0.60-0.70 | 29 | +0.000050 | 34.5% |
| 0.70-0.80 | 31 | -0.000093 | 67.7% |
| 0.80-0.85 | 20 | +0.001802 | 30.0% |
| 0.85-0.90 | 38 | -0.000305 | 28.9% |
| 0.90-1.00 | 35 | +0.000187 | 48.6% |

Mid-range base AUC (0.70-0.80) shows the highest win rate, but the pattern is not monotonic.

### Pruning Mode Comparison

| Mode | n | Mean Objective | % Negative |
|---|---|---|---|
| no_pruning | 12 | +0.000213 | 66.7% |
| optuna | 129 | +0.000122 | 41.9% |
| fixed_50 | 12 | +0.000934 | 25.0% |

Differences not statistically significant (Mann-Whitney p > 0.19).

### Augmented vs Natural

| Type | n | Mean Objective | % Negative |
|---|---|---|---|
| Augmented | 64 | -0.000444 | 51.6% |
| Natural | 89 | +0.000652 | 36.0% |

Marginally significant (Mann-Whitney p=0.074).

---

## Conclusions

1. **With the properties measured, we found no strong cross-dataset predictor of the objective.** The strongest correlation (ratio_ext_no, r=-0.263) explains only ~7% of variance. The full multivariate model explains <2%.

2. **Weak trends observed but not conclusive:**
   - Lower null rates (0-20%) tend to produce better incremental results
   - Mid-range base AUC (0.70-0.80) has the highest win rate
   - Augmented datasets trend toward better incremental performance (p=0.074)

3. **Pruning mode has no significant effect** in this sample, though no_pruning shows the highest win rate (66.7% vs 41.9% for optuna).

4. **The objective decomposition shows both terms (base vs combined, ext vs combined) contribute roughly equally** — neither the base population nor the extended population dominates the outcome.

## Limitations

- Only 153 rows across 10 datasets — low statistical power for cross-dataset analysis
- Objectives are small (mostly -0.01 to +0.01) — measurement noise may dominate
- Dataset heterogeneity (different domains, sizes, feature types) adds confounding variance
- Properties measured are all surface-level statistics — we did not measure the **informativeness** of the extended features relative to the task

## Next Steps

Deeper analysis needed to investigate properties that capture the **relationship between extended features and the prediction task**, not just surface-level dataset statistics. See `analysis_v2_plan.md`.
