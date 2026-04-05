# Analysis v2 Phase 2 Plan: Population Differences & Covariate Shift

Date: 2026-04-05

## Motivation

Phase 1 showed that feature importance doesn't predict the objective. The incremental learning advantage comes from **specialization** — the extended model trains on a subpopulation (rows with extended features) that may have different patterns than the full population. Phase 2 investigates whether measurable differences between the extended and non-extended populations predict incremental learning success.

## Core Hypothesis

**If the extended population differs meaningfully from the non-extended population (in label distribution, feature distributions, or both), incremental learning benefits from specialization and should produce more negative objectives.**

## Properties to Compute

For each dataset × combo experiment, split the data into two populations:
- **ext_pop**: rows where `has_extended = 1` (have the extended features)
- **no_ext_pop**: rows where `has_extended = 0` (missing the extended features)

### Group 1: Label Distribution Shift
- `label_rate_ext` — positive class rate in the extended population
- `label_rate_no_ext` — positive class rate in the non-extended population
- `label_rate_gap` — absolute difference: |label_rate_ext - label_rate_no_ext|
- `label_rate_ratio` — label_rate_ext / label_rate_no_ext
- `chi2_label` — chi-squared test statistic for label × has_extended independence
- `chi2_label_p` — p-value of the chi-squared test

### Group 2: Base Feature Distribution Shift
For each base feature, compute statistical distance between ext_pop and no_ext_pop:
- `mean_ks_statistic` — average Kolmogorov-Smirnov statistic across all numeric base features
- `max_ks_statistic` — maximum KS statistic (the most shifted feature)
- `mean_feature_shift` — average |mean_ext - mean_no_ext| / std across base features (Cohen's d-like)
- `pct_features_shifted` — % of base features with significant KS test (p < 0.05)

### Group 3: Combined Shift Score
- `psi` — Population Stability Index between ext and no_ext populations on base features
- `total_shift` — label_rate_gap + mean_ks_statistic (combined measure)

## Implementation

### Script: `analysis/compute_phase2_properties.py`

For each dataset:
1. Load the data using the same loader functions from `run_ablation_pruning.py`
2. For each combo in the dataset's CSV, compute `has_extended` from the ext_features
3. Split into ext_pop and no_ext_pop
4. Compute all Group 1, 2, 3 properties
5. Merge with existing CSV data
6. Save augmented CSV

### Datasets to process
All 10 datasets. Use the loaders from `scripts/run_ablation_pruning.py` for consistency.

### Correlation Analysis
After computing properties:
1. Correlate each property with objective (Pearson + Spearman, across all experiments)
2. Test interactions between new properties and existing ones
3. Build multivariate model combining best Phase 1 + Phase 2 properties
4. Report findings in `analysis/analysis_v2_phase2_results.md`

## Key Files
- `scripts/run_ablation_pruning.py` — dataset loader functions to reuse
- `results/all_results_*.csv` — experiment data to augment
- `analysis/compute_phase2_properties.py` — new script to create
- `analysis/analysis_v2_phase2_results.md` — findings

## Expected Runtime
Each dataset needs to: load data, compute has_extended per combo, run KS tests.
Should be fast (< 5 min total) — no model training needed.
