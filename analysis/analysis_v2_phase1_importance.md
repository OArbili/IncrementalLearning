# Analysis v2 Phase 1: Feature Importance Ratios vs Objective

Date: 2026-04-05

## Method

Extracted feature importances from saved combined models (`results/ablation/<dataset>/combined_model.json`) for 9 datasets (FlightDelay excluded — no saved combined model). Computed:

- `ext_imp_ratio` — fraction of total combined-model importance from extended features
- `ext_imp_per_feature` — average importance per extended feature
- `ext_vs_base_ratio` — ratio of per-feature importance (extended / base)

Matched to 147 experiment rows (110 unique combos across 9 datasets).

## Cross-Dataset Results

**No significant correlation found.**

| Property | Pearson r | p-value |
|---|---|---|
| ext_imp_ratio | -0.035 | 0.677 |
| ext_imp_per_feature | -0.059 | 0.478 |
| ext_vs_base_ratio | -0.103 | 0.213 |
| has_extended importance | -0.086 | 0.306 |

## Within-Dataset Results

Mixed and mostly non-significant:

| Dataset | n | Spearman r | p | Direction |
|---|---|---|---|---|
| BankLoanSta | 14 | -0.246 | 0.396 | High importance → better (but n.s.) |
| WIDS | 21 | -0.386 | 0.084 | High importance → better (marginally) |
| WeatherAUS | 8 | +0.714 | 0.047* | High importance → WORSE |
| Others | — | — | >0.5 | No pattern |

## Median Split Test

Splitting combos within each dataset at their median ext_imp_ratio:

| Group | n | Mean Objective | % Negative |
|---|---|---|---|
| Below-median importance | 53 | +0.001309 | 34.0% |
| Above-median importance | 57 | -0.000084 | 35.1% |

**Mann-Whitney p = 0.957** — no difference.

## Key Observations from Per-Dataset Patterns

Looking at the detailed combo-level data, some interesting (but non-significant) patterns:

- **BankLoanSta:** The strongest incremental wins come from combos with HIGH ext importance (Curren+Annual+Credit: ext_ratio=0.558, obj=-0.074). But some high-importance combos also lose (Credit alone: ext_ratio=0.409, obj=+0.008). The pattern is driven by how many strong features are used together, not importance alone.

- **MovieAugV2:** The best incremental win (rating_mean+rating_std, obj=-0.004) has VERY LOW ext importance (0.027). The tag-heavy combos with HIGH importance (0.83) tend to lose. This suggests that when extended features dominate the combined model, the combined model is already exploiting them well and incremental learning adds little.

- **WIDS:** Marginal trend (p=0.084) — more important ext features tend toward better objectives. But the effect is weak.

## Conclusion

**Feature importance of extended features in the combined model does not predict whether incremental learning will outperform the combined model.** This is actually an informative null result:

1. The combined model already uses the extended features optimally (by definition — it's trained on all data). Whether those features are important or not, the combined model has access to them.

2. The incremental learning advantage comes from **specialization** — the extended model is trained only on the subpopulation that has the features, potentially learning different patterns. This advantage is not captured by overall feature importance.

3. The objective depends on the **relative ability** of the extended model to beat the combined model on the extended population, which is about model architecture (warm-starting, tree pruning) rather than feature informativeness.

## Next Steps

Phase 1 properties (feature importance) do not predict the objective. The v2 plan's Phase 2 (information-theoretic measures) and Phase 3 (covariate shift between populations) may be more promising, as they directly measure the structural differences between the extended and non-extended populations that incremental learning exploits.
