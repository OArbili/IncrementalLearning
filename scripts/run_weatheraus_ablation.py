#!/usr/bin/env python3
"""WeatherAUS: Feature combo search + ablation study.

Phase 1: Find best feature combo using 20 Optuna trials per combo.
Phase 2: Ablation on best combo — optuna (50 trials), no_pruning (15), fixed_50 (15).

Usage:
    python run_weatheraus_ablation.py
    python run_weatheraus_ablation.py --test
"""
import sys
import os
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import time
import shutil
from datetime import datetime
from itertools import combinations
import pandas as pd
import numpy as np

from core.GenericDataPipeline import GenericDataPipeline
from core.RunData import RunPipeline
from core.XGBoostModel import XGBoostModel
from sklearn.metrics import roc_auc_score
from core.seed_utils import SEED, set_all_seeds

pd.set_option('future.infer_string', False)
set_all_seeds()

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPTS_DIR, '..', 'results', 'WeatherAUS')
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASET_NAME = 'WeatherAUS'

# Quick test mode: python run_weatheraus_ablation.py --test
TEST_MODE = '--test' in sys.argv

# Phase 1 config
PHASE1_TRIALS = 3 if TEST_MODE else 20

# Phase 2 config (ablation)
PHASE2_BASE_COMBINED_TRIALS = 3 if TEST_MODE else 30
ABLATION_TRIALS = {
    'optuna': 3 if TEST_MODE else 50,
    'no_pruning': 3 if TEST_MODE else 15,
    'fixed_50': 3 if TEST_MODE else 15,
}

pipeline = GenericDataPipeline()


# ============================================================================
# Data loading
# ============================================================================

def load_weatheraus():
    """WeatherAUS (Natural Nulls)."""
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'weatherAUS.csv')
    df = pd.read_csv(csv_path)
    label = "RainTomorrow"
    df.dropna(subset=[label], inplace=True)
    df.drop(columns=['Date', 'Location', 'RainToday'], errors='ignore', inplace=True)
    df = pipeline.preprocessing(df)
    df[label] = df[label].astype(int)
    return df, label


# ============================================================================
# Null group discovery and combo enumeration
# ============================================================================

def find_null_groups(df, label, min_null_rate=0.05, jaccard_threshold=0.95, max_groups=5):
    """Group features by null pattern similarity, return list of groups."""
    null_features = [c for c in df.columns if c != label and df[c].isnull().mean() > min_null_rate]
    if not null_features:
        return []

    null_masks = {}
    for col in null_features:
        null_masks[col] = set(df[df[col].isna()].index.tolist())

    groups = []
    assigned = set()
    for i, f1 in enumerate(null_features):
        if f1 in assigned:
            continue
        group = [f1]
        assigned.add(f1)
        for f2 in null_features[i+1:]:
            if f2 in assigned:
                continue
            overlap = len(null_masks[f1] & null_masks[f2])
            union = len(null_masks[f1] | null_masks[f2])
            jaccard = overlap / union if union > 0 else 0
            if jaccard > jaccard_threshold:
                group.append(f2)
                assigned.add(f2)
        groups.append(group)

    groups.sort(key=lambda g: df[g[0]].isnull().mean(), reverse=True)
    return groups[:max_groups]


def enumerate_combos(df, label, groups):
    """Enumerate all valid feature combos from groups, dedup by population."""
    all_combos = []
    for r in range(1, len(groups) + 1):
        for combo in combinations(range(len(groups)), r):
            ext_features = []
            for idx in combo:
                ext_features.extend(groups[idx])
            all_combos.append(ext_features)

    valid_combos = []
    seen_populations = set()
    for ext_features in all_combos:
        has_ext = df[ext_features].notna().all(axis=1)
        n_ext = has_ext.sum()
        n_no = (~has_ext).sum()

        if n_ext < 100 or n_no < 100:
            continue

        pop_key = (n_no, n_ext)
        if pop_key in seen_populations:
            continue
        seen_populations.add(pop_key)

        valid_combos.append({
            'ext_features': ext_features,
            'n_ext': n_ext,
            'n_no': n_no,
            'ext_pct': n_ext / len(df) * 100,
        })

    return valid_combos


# ============================================================================
# Phase 1: Run a single combo experiment
# ============================================================================

def run_combo(df, label, ext_features, n_trials):
    """Train base, extended, combined for one feature combo. Returns result dict."""
    all_features_list = [c for c in df.columns if c != label]
    base_features = [f for f in all_features_list if f not in ext_features]

    dm = RunPipeline()
    dm.load_data(base_features, ext_features, df.copy(), label)
    dm.set_has_extended()
    ret = dm.train_test_split()
    if ret == 999:
        return None
    dm.set_train_base_ext_datasets()

    ext_train = dm.ext_df[dm.ext_df['has_extended'] == 1].copy()
    test_with = dm.test_df[dm.test_df['has_extended'] == 1]
    test_without = dm.test_df[dm.test_df['has_extended'] == 0]

    if len(test_with) < 50 or len(test_without) < 50:
        return None

    feature_cols = sorted(dm.base_features + dm.ext_features)

    # Base model
    print("\n=== Training Base Model ===", flush=True)
    dm.base_model = XGBoostModel(name="base_model")
    dm.base_model.train(
        X=dm.base_df[feature_cols],
        y=dm.base_df[dm.label],
        n_trials=n_trials
    )
    dm.base_model.save_model()

    base_auc = roc_auc_score(test_without[label],
                              dm.base_model.predict(test_without[feature_cols]))

    # Extended model
    print("\n=== Training Extended Model ===", flush=True)
    dm.extended_model = XGBoostModel(name="extended_model")
    dm.extended_model.train(
        ext_train[feature_cols],
        ext_train[dm.label],
        base_model_path="base_model.json",
        n_trials=n_trials,
        pruning_mode='optuna'
    )

    ext_auc = roc_auc_score(test_with[label],
                             dm.extended_model.predict(test_with[feature_cols]))

    # Combined model
    print("\n=== Training Combined Model ===", flush=True)
    dm.combined_model = XGBoostModel(name="combined_model")
    dm.combined_model.train(
        dm.train_df[feature_cols],
        dm.train_df[dm.label],
        n_trials=n_trials
    )

    comb_no_auc = roc_auc_score(test_without[label],
                                 dm.combined_model.predict(test_without[feature_cols]))
    comb_ext_auc = roc_auc_score(test_with[label],
                                  dm.combined_model.predict(test_with[feature_cols]))

    objective = (comb_ext_auc - ext_auc) + (comb_no_auc - base_auc)

    return {
        'objective': objective,
        'base_auc': base_auc,
        'ext_auc': ext_auc,
        'comb_no_auc': comb_no_auc,
        'comb_ext_auc': comb_ext_auc,
        'n_test_no': len(test_without),
        'n_test_ext': len(test_with),
        'ext_features': ext_features,
        'dm': dm,
    }


# ============================================================================
# Main
# ============================================================================

print("=" * 100)
print(f"{DATASET_NAME}: FEATURE COMBO SEARCH + ABLATION STUDY")
print(f"Phase 1: {PHASE1_TRIALS} trials per combo | Phase 2: base/combined={PHASE2_BASE_COMBINED_TRIALS}, "
      f"optuna={ABLATION_TRIALS['optuna']}, no_pruning={ABLATION_TRIALS['no_pruning']}, fixed_50={ABLATION_TRIALS['fixed_50']}")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 100)

# Load data
df, label = load_weatheraus()
print(f"Shape: {df.shape}, Label: {label}")
print(f"Target distribution:\n{df[label].value_counts()}", flush=True)

# ============================================================================
# PHASE 1: Find best feature combo
# ============================================================================

print(f"\n{'#'*100}")
print(f"# PHASE 1: FEATURE COMBO SEARCH ({PHASE1_TRIALS} trials per combo)")
print(f"{'#'*100}", flush=True)

groups = find_null_groups(df, label)
if not groups:
    print("No null groups found, exiting")
    sys.exit(1)

print(f"\nFound {len(groups)} null groups:")
for i, g in enumerate(groups):
    null_rate = df[g[0]].isnull().mean()
    print(f"  Group {i+1}: {g} ({null_rate:.1%} null)")

combos = enumerate_combos(df, label, groups)
print(f"\n{len(combos)} valid combos to evaluate")

print(f"\n{'#':<4} {'Name':<50} {'N_no':>8} {'N_ext':>8} {'%ext':>8}")
print("-" * 82)
for i, c in enumerate(combos):
    name = "+".join([f[:6] for f in c['ext_features']])
    print(f"{i+1:<4} {name:<50} {c['n_no']:>8} {c['n_ext']:>8} {c['ext_pct']:>7.1f}%")
print(flush=True)

# Run all combos
phase1_start = time.time()
best_result = None
all_combo_results = []

for i, combo in enumerate(combos):
    ext_features = combo['ext_features']
    combo_name = "+".join([f[:6] for f in ext_features])

    print(f"\n{'='*80}")
    print(f"Combo {i+1}/{len(combos)}: {combo_name} ext")
    print(f"{'='*80}", flush=True)

    result = run_combo(df, label, ext_features, PHASE1_TRIALS)

    if result is None:
        print(f">>> RESULT: SKIPPED (invalid split)")
        continue

    print(f"\n>>> RESULT: objective = {result['objective']:.6f}")
    print(f"    Base AUC: {result['base_auc']:.6f} ({result['n_test_no']} rows)")
    print(f"    Extended AUC: {result['ext_auc']:.6f} ({result['n_test_ext']} rows)")
    print(f"    Combined (no ext) AUC: {result['comb_no_auc']:.6f}")
    print(f"    Combined (with ext) AUC: {result['comb_ext_auc']:.6f}", flush=True)

    all_combo_results.append({
        'combo_name': combo_name,
        'ext_features': ext_features,
        **{k: result[k] for k in ['objective', 'base_auc', 'ext_auc', 'comb_no_auc', 'comb_ext_auc', 'n_test_no', 'n_test_ext']},
    })

    if best_result is None or result['objective'] < best_result['objective']:
        best_result = result

phase1_elapsed = time.time() - phase1_start

if best_result is None:
    print("\nNo valid combos found, exiting")
    sys.exit(1)

best_name = "+".join([f[:6] for f in best_result['ext_features']])
print(f"\n{'*'*80}")
print(f"PHASE 1 COMPLETE — Best combo: {best_name}")
print(f"  Objective: {best_result['objective']:.6f}")
print(f"  Base AUC: {best_result['base_auc']:.4f}, Ext AUC: {best_result['ext_auc']:.4f}")
print(f"  Comb-no: {best_result['comb_no_auc']:.4f}, Comb+ext: {best_result['comb_ext_auc']:.4f}")
print(f"  Time: {phase1_elapsed:.0f}s")
print(f"{'*'*80}", flush=True)

# ============================================================================
# PHASE 2: Ablation study on best combo
# ============================================================================

print(f"\n{'#'*100}")
print(f"# PHASE 2: ABLATION STUDY on best combo ({best_name})")
print(f"# Trials: optuna={ABLATION_TRIALS['optuna']}, no_pruning={ABLATION_TRIALS['no_pruning']}, fixed_50={ABLATION_TRIALS['fixed_50']}")
print(f"{'#'*100}", flush=True)

# Reuse data split from Phase 1 best result
dm = best_result['dm']
ext_features = best_result['ext_features']
feature_cols = sorted(dm.base_features + dm.ext_features)

ext_train = dm.ext_df[dm.ext_df['has_extended'] == 1].copy()
test_with = dm.test_df[dm.test_df['has_extended'] == 1]
test_without = dm.test_df[dm.test_df['has_extended'] == 0]

# Re-train base and combined with 30 trials (Phase 1 used 20)
print(f"\n=== Training Base Model ({PHASE2_BASE_COMBINED_TRIALS} trials) ===", flush=True)
dm.base_model = XGBoostModel(name="ablation_base_model")
dm.base_model.train(
    X=dm.base_df[feature_cols],
    y=dm.base_df[dm.label],
    n_trials=PHASE2_BASE_COMBINED_TRIALS
)
dm.base_model.save_model()

base_auc = roc_auc_score(test_without[label],
                          dm.base_model.predict(test_without[feature_cols]))

print(f"\n=== Training Combined Model ({PHASE2_BASE_COMBINED_TRIALS} trials) ===", flush=True)
dm.combined_model = XGBoostModel(name="ablation_combined_model")
dm.combined_model.train(
    dm.train_df[feature_cols],
    dm.train_df[dm.label],
    n_trials=PHASE2_BASE_COMBINED_TRIALS
)

comb_no_auc = roc_auc_score(test_without[label],
                             dm.combined_model.predict(test_without[feature_cols]))
comb_ext_auc = roc_auc_score(test_with[label],
                              dm.combined_model.predict(test_with[feature_cols]))

print(f"\nBase AUC: {base_auc:.6f}")
print(f"Combined (no-ext): {comb_no_auc:.6f}, Combined (with-ext): {comb_ext_auc:.6f}", flush=True)

# Train extended model with 3 pruning modes
ablation_results = []
ablation_models = {}

for mode in ['optuna', 'no_pruning', 'fixed_50']:
    mode_trials = ABLATION_TRIALS[mode]
    print(f"\n{'='*80}")
    print(f"ABLATION — {mode} (n_trials={mode_trials})")
    print(f"{'='*80}", flush=True)

    dm.extended_model = XGBoostModel(name=f"extended_{mode}")
    dm.extended_model.train(
        ext_train[feature_cols],
        ext_train[dm.label],
        base_model_path="ablation_base_model.json",
        n_trials=mode_trials,
        pruning_mode=mode
    )

    ext_auc = roc_auc_score(test_with[label],
                             dm.extended_model.predict(test_with[feature_cols]))
    objective = (comb_ext_auc - ext_auc) + (comb_no_auc - base_auc)

    print(f"\n>>> {mode}: objective={objective:.6f}, ext_auc={ext_auc:.6f}", flush=True)

    ablation_models[mode] = dm.extended_model
    ablation_results.append({
        'mode': mode,
        'objective': objective,
        'ext_auc': ext_auc,
        'base_auc': base_auc,
        'comb_no_auc': comb_no_auc,
        'comb_ext_auc': comb_ext_auc,
    })

# ============================================================================
# Final Summary + Save All Artifacts
# ============================================================================

print(f"\n{'='*120}")
print("FINAL SUMMARY")
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*120}")

# Phase 1 results
print(f"\n--- Phase 1: Feature Combo Search ({PHASE1_TRIALS} trials) ---")
print(f"{'Combo':<50} {'Objective':>10} {'Base AUC':>10} {'Ext AUC':>10} {'Comb-no':>10} {'Comb+ext':>10}")
print("-" * 100)
for r in all_combo_results:
    marker = " <-- BEST" if r['combo_name'] == best_name else ""
    print(f"{r['combo_name']:<50} {r['objective']:>10.6f} {r['base_auc']:>10.4f} {r['ext_auc']:>10.4f} {r['comb_no_auc']:>10.4f} {r['comb_ext_auc']:>10.4f}{marker}")

# Phase 2 results
print(f"\n--- Phase 2: Ablation on {best_name} (base/combined={PHASE2_BASE_COMBINED_TRIALS} trials) ---")
print(f"{'Mode':<14} {'Trials':>8} {'Objective':>12} {'Ext AUC':>10} {'Base AUC':>10} {'Comb-no':>10} {'Comb+ext':>10}")
print("-" * 80)
best_ablation = min(ablation_results, key=lambda x: x['objective'])
for r in ablation_results:
    marker = " <-- BEST" if r['mode'] == best_ablation['mode'] else ""
    print(f"{r['mode']:<14} {ABLATION_TRIALS[r['mode']]:>8} {r['objective']:>12.6f} {r['ext_auc']:>10.4f} {r['base_auc']:>10.4f} {r['comb_no_auc']:>10.4f} {r['comb_ext_auc']:>10.4f}{marker}")

# --- Save all artifacts ---
print(f"\nSaving artifacts to {RESULTS_DIR} ...", flush=True)

# 1. Summary text
summary_path = os.path.join(RESULTS_DIR, 'ablation_summary.txt')
with open(summary_path, 'w') as f:
    f.write("=" * 120 + "\n")
    f.write(f"{DATASET_NAME}: FEATURE COMBO SEARCH + ABLATION STUDY\n")
    f.write(f"Phase 1: {PHASE1_TRIALS} trials per combo\n")
    f.write(f"Phase 2: optuna={ABLATION_TRIALS['optuna']}, no_pruning={ABLATION_TRIALS['no_pruning']}, fixed_50={ABLATION_TRIALS['fixed_50']}\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 120 + "\n\n")

    f.write("Phase 1: Feature Combo Search\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Combo':<50} {'Objective':>10} {'Base AUC':>10} {'Ext AUC':>10} {'Comb-no':>10} {'Comb+ext':>10}\n")
    for r in all_combo_results:
        marker = " <-- BEST" if r['combo_name'] == best_name else ""
        f.write(f"{r['combo_name']:<50} {r['objective']:>10.6f} {r['base_auc']:>10.4f} {r['ext_auc']:>10.4f} {r['comb_no_auc']:>10.4f} {r['comb_ext_auc']:>10.4f}{marker}\n")

    f.write(f"\nPhase 2: Ablation on {best_name}\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Mode':<14} {'Trials':>8} {'Objective':>12} {'Ext AUC':>10} {'Base AUC':>10} {'Comb-no':>10} {'Comb+ext':>10}\n")
    for r in ablation_results:
        marker = " <-- BEST" if r['mode'] == best_ablation['mode'] else ""
        f.write(f"{r['mode']:<14} {ABLATION_TRIALS[r['mode']]:>8} {r['objective']:>12.6f} {r['ext_auc']:>10.4f} {r['base_auc']:>10.4f} {r['comb_no_auc']:>10.4f} {r['comb_ext_auc']:>10.4f}{marker}\n")

# 2. Phase 1 results CSV
pd.DataFrame(all_combo_results).drop(columns=['ext_features'], errors='ignore').to_csv(
    os.path.join(RESULTS_DIR, 'phase1_results.csv'), index=False)

# 3. Phase 2 results CSV
pd.DataFrame(ablation_results).to_csv(
    os.path.join(RESULTS_DIR, 'phase2_ablation_results.csv'), index=False)

# 4. Save Phase 2 model files
models_dir = os.path.join(RESULTS_DIR, 'models')
os.makedirs(models_dir, exist_ok=True)

dm.base_model.model.save_model(os.path.join(models_dir, 'base_model.json'))
dm.combined_model.model.save_model(os.path.join(models_dir, 'combined_model.json'))
for mode, ext_model in ablation_models.items():
    ext_model.model.save_model(os.path.join(models_dir, f'extended_{mode}.json'))

# 5. Feature importance per model
fi_dir = os.path.join(RESULTS_DIR, 'feature_importance')
os.makedirs(fi_dir, exist_ok=True)

for name, model_obj in [('base', dm.base_model), ('combined', dm.combined_model)] + \
                         [(f'extended_{m}', ablation_models[m]) for m in ablation_models]:
    imp = model_obj.model.feature_importances_
    fi_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': imp,
    }).sort_values('importance', ascending=False)
    fi_df.to_csv(os.path.join(fi_dir, f'{name}_importance.csv'), index=False)

# 6. Optuna trial history per model
trials_dir = os.path.join(RESULTS_DIR, 'optuna_trials')
os.makedirs(trials_dir, exist_ok=True)

for name, model_obj in [('base', dm.base_model), ('combined', dm.combined_model)] + \
                         [(f'extended_{m}', ablation_models[m]) for m in ablation_models]:
    if hasattr(model_obj, 'study') and model_obj.study is not None:
        trials_data = []
        for t in model_obj.study.trials:
            trials_data.append({
                'number': t.number,
                'value': t.value,
                **t.params,
            })
        pd.DataFrame(trials_data).to_csv(
            os.path.join(trials_dir, f'{name}_trials.csv'), index=False)

# 7. Experiment config
config = {
    'dataset': DATASET_NAME,
    'phase1_trials': PHASE1_TRIALS,
    'phase2_base_combined_trials': PHASE2_BASE_COMBINED_TRIALS,
    'ablation_trials': ABLATION_TRIALS,
    'best_combo': best_result['ext_features'],
    'best_combo_name': best_name,
    'n_combos_evaluated': len(all_combo_results),
    'generated': datetime.now().isoformat(),
    'seed': SEED,
}
with open(os.path.join(RESULTS_DIR, 'experiment_config.json'), 'w') as f:
    json.dump(config, f, indent=2)

print(f"Artifacts saved: summary, CSVs, models, feature importance, optuna trials, config")
print(f"Results directory: {RESULTS_DIR}")
print("Done!")
