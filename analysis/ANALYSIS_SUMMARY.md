# Analysis Summary — Incremental Learning Experiments

Date: 2026-04-05

## Overview

153 experiments across 10 datasets, testing whether an incremental learning approach (base + extended model) outperforms a single combined model. Objective = (comb+ext AUC - ext AUC) + (comb-no AUC - base AUC). Negative = incremental wins.

## Key Findings

### 1. Incremental learning wins are achievable but not guaranteed

- **42.5%** of all experiments show negative objectives (incremental wins)
- **All 10 datasets** achieve negative objectives with at least one feature combination
- The mean objective across all experiments is **not significantly different from zero** (p=0.83)
- But the **best per dataset** is consistently negative (Wilcoxon p=0.002, Cohen's d = -0.65)

### 2. The win comes from the base model, not the extended model

The objective decomposes into two paired comparisons:
- Base model vs combined on no-ext rows: **base wins 59.5%** of the time
- Extended model vs combined on ext rows: **ext wins only 31.4%** of the time

The incremental approach succeeds primarily because training on all rows with extended features set to NaN allows the base model to **specialize on the no-extended population** better than the combined model, which must handle both populations.

### 3. No measurable property predicts success

We tested correlations between the objective and:
- Dataset properties (size, null rates, label distribution) — no signal
- Feature importance ratios from trained models — no signal
- Population shift between ext and no-ext rows (KS, Cohen's d, label rate) — no signal

The multivariate regression R² was 0.019. **The optimal feature combination is dataset-specific and cannot be predicted from metadata.**

### 4. High run-to-run variance limits reliability

Within-combo std from different Optuna runs = **0.009**, which is **88% of total std** (0.011). Several combos flip sign between runs. Individual results should be replicated before drawing conclusions.

## Analyses Performed

| Analysis | File | Finding |
|---|---|---|
| v1: Surface correlations | `analysis_v1_correlations.md` | R²=0.019, no strong predictor |
| v2 Phase 1: Feature importance | `analysis_v2_phase1_importance.md` | r=-0.035, no signal |
| v2 Phase 2: Population shift | `analysis_v2_phase2_results.md` | All |r| < 0.15 |
| v2 Phase 4: Variance + effect size | `analysis_v2_phase4_results.md` | High noise, base drives wins |

## Data Files

| File | Description |
|---|---|
| `results/all_results_*.csv` | Per-dataset experiment results (10 files, 153 rows total) |
| `results/summary_best_per_dataset.csv` | Best objective per dataset |
| `results/summary_results_v1.md` | Research results summary |
| `analysis/phase2_shift_properties.csv` | Population shift computed properties |
