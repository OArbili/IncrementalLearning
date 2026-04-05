# Analysis v2 Plan: Deeper Investigation of Incremental Learning Success Factors

Date: 2026-04-05

## Motivation

Analysis v1 found no strong cross-dataset predictor of the objective using surface-level dataset statistics (null %, dataset size, label distribution, base AUC). The missing piece is likely the **informativeness of the extended features** — how much predictive signal they carry relative to the base features and the task.

## Proposed New Properties to Compute

### 1. Feature Importance Ratios (from trained models)

For each experiment, extract from the trained models:

- **ext_importance_sum** — sum of feature importances for extended features in the combined model
- **ext_importance_ratio** — ext_importance_sum / total importance (what fraction of the combined model's signal comes from extended features)
- **base_importance_sum** — importance of base features in the combined model
- **ext_importance_in_ext_model** — importance of extended features in the extended model specifically
- **has_extended_importance** — importance of the `has_extended` indicator feature

**Hypothesis:** Higher ext_importance_ratio → more negative objective (incremental wins), because the extended features carry real signal that the base model misses.

### 2. Information-Theoretic Measures (computed from raw data)

- **mutual_information(ext_features, label)** — how much information extended features carry about the target
- **conditional_MI(ext_features, label | base_features)** — additional information beyond what base features provide
- **correlation_ext_base** — how correlated extended features are with base features (redundancy measure)

**Hypothesis:** Higher conditional MI → more negative objective. If ext features are redundant with base, incremental learning adds little.

### 3. Data Distribution Properties

- **ext_population_label_rate** — positive class rate among rows WITH extended features
- **no_ext_population_label_rate** — positive class rate among rows WITHOUT extended features
- **label_rate_gap** — difference between the two rates above
- **covariate_shift** — statistical distance between ext and no-ext populations on base features (e.g., KL divergence or Wasserstein distance on base feature distributions)

**Hypothesis:** If the ext population has a different label distribution or covariate shift from the no-ext population, incremental learning should benefit more from specialization.

### 4. Model Capacity Measures

- **n_trees_base** — number of trees in the base model
- **n_trees_extended** — number of trees added by the extended model
- **tree_ratio** — extended trees / base trees
- **best_use_base_model** — whether Optuna chose to warm-start (True/False)
- **best_n_trees_keep** — how many base trees Optuna kept

**Hypothesis:** When Optuna keeps more base trees and adds fewer new ones, the task is simpler and incremental learning is effective.

### 5. Cross-Validation vs Test Gap

- **cv_auc_ext** — CV AUC of the extended model during training
- **test_auc_ext** — actual test AUC
- **cv_test_gap** — cv_auc - test_auc (overfitting measure)

**Hypothesis:** Larger CV-test gap indicates overfitting, which hurts incremental performance.

## Implementation Plan

### Phase 1: Feature Importance Extraction
- Load saved model JSONs from `results/ablation/<dataset>/` directories
- Parse XGBoost feature importances from each model
- Compute ext_importance_ratio for combined model
- Match to experiment rows in CSVs

### Phase 2: Data-Level Properties
- For each dataset × combo, load the data and compute:
  - Label rates per population (ext vs no-ext)
  - Mutual information of ext features with label
  - Base feature distribution shift between populations

### Phase 3: Model Metadata
- Extract n_trees, Optuna decisions from model metadata JSONs
- Parse from training logs if metadata not available

### Phase 4: Analysis
- Add all new properties as columns to the per-dataset CSVs
- Re-run correlation analysis
- Build multivariate model with new + old features
- Report which properties have the strongest predictive power

## Priority

Start with **Feature Importance Ratios** (Phase 1) — these are the easiest to extract from existing model files and most likely to show a strong signal.

## Files to Read/Modify

- `results/ablation/<dataset>/*.json` — saved XGBoost models
- `results/all_results_*.csv` — add new columns
- `core/XGBoostModel.py` — understand model save format
- `scripts/run_ablation_pruning.py` — understand training pipeline for each dataset
