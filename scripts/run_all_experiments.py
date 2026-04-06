#!/usr/bin/env python3
"""Run all dataset experiments + ablation sequentially and generate a unified summary.

For each dataset:
  1. Run the original experiment (find best ext feature combo)
  2. Run ablation on best combo (no_pruning and fixed_50 only)
  3. Save results to per-dataset files and unified summary

Usage:
    python run_all_experiments.py [N_TRIALS] [DATASET_FILTER]

Examples:
    python run_all_experiments.py 30                    # all datasets, 30 trials
    python run_all_experiments.py 30 Weather,WIDS       # only Weather and WIDS
"""
import sys
import os
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import re
import time
import shutil
from datetime import datetime
import pandas as pd
import numpy as np
try:
    import kagglehub
except ImportError:
    kagglehub = None

from core.GenericDataPipeline import GenericDataPipeline
from core.RunData import RunPipeline
from core.XGBoostModel import XGBoostModel
from sklearn.metrics import roc_auc_score
from core.seed_utils import SEED, set_all_seeds

pd.set_option('future.infer_string', False)
set_all_seeds()

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPTS_DIR, '..', 'results')
ABLATION_DIR = os.path.join(RESULTS_DIR, 'ablation')

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
DATASET_FILTER = sys.argv[2].split(',') if len(sys.argv) > 2 else None
ABLATION_TRIALS = max(N_TRIALS // 2, 10)  # no_pruning and fixed_50 trials

pipeline = GenericDataPipeline()

# ============================================================================
# Dataset loading functions — each returns (df, label, null_groups_config)
#   null_groups_config: dict with keys:
#     'mode': 'natural' | 'augmented'
#     For 'natural': uses pipeline to find null groups and enumerate combos
#     For 'augmented': 'inject_features' list, 'inject_rate' float
# ============================================================================

def load_bankloansta():
    """BankLoanSta (Augmented)."""
    path = kagglehub.dataset_download("zaurbegiev/my-dataset")
    csv_files = sorted([f for f in os.listdir(path) if f.endswith('.csv')])
    csv_path = os.path.join(path, csv_files[1])
    df = pd.read_csv(csv_path)
    df.dropna(subset=['Loan Status'], inplace=True)
    df.drop(['Loan ID', 'Customer ID'], axis=1, inplace=True)
    df = pipeline.preprocessing(df)
    label = "Loan Status"
    df[label] = df[label].astype(int)

    # Augmentation: inject nulls into selected features (independent masks)
    inject_features = ['Current Loan Amount', 'Annual Income', 'Credit Score']
    inject_rate = 0.20
    np.random.seed(SEED)
    for col in inject_features:
        mask = np.random.rand(len(df)) < inject_rate
        df.loc[mask, col] = np.nan

    return df, label, {
        'mode': 'augmented',
        'inject_features': inject_features,
    }


def load_weather():
    """Weather (Natural Nulls)."""
    path = kagglehub.dataset_download("rever3nd/weather-data")
    csv_files = sorted([f for f in os.listdir(path) if f.endswith('.csv')])
    csv_path = os.path.join(path, csv_files[0])
    df = pd.read_csv(csv_path)
    label = "RainTomorrow"
    df.dropna(subset=[label], inplace=True)
    df = pipeline.preprocessing(df)
    df[label] = df[label].astype(int)
    return df, label, {'mode': 'natural'}


def load_diabetes():
    """DiabetesRecord (Natural Nulls)."""
    path = kagglehub.dataset_download("priyamchoksi/100000-diabetes-clinical-dataset")
    csv_path = os.path.join(path, "diabetes_prediction_dataset.csv")
    df = pd.read_csv(csv_path)
    label = "diabetes"
    df = pipeline.preprocessing(df)
    df[label] = df[label].astype(int)
    return df, label, {'mode': 'natural'}


def load_hr_analytics():
    """HRAnalytics (Natural Nulls)."""
    path = kagglehub.dataset_download("arashnic/hr-analytics-job-change-of-data-scientists")
    csv_path = os.path.join(path, "aug_train.csv")
    df = pd.read_csv(csv_path)
    df = df.drop(columns=['enrollee_id'], errors='ignore')
    df = pipeline.preprocessing(df)
    label = "target"
    df[label] = df[label].astype(int)
    return df, label, {'mode': 'natural'}


def load_client_record_aug():
    """ClientRecord (Augmented) — Telecom customer churn."""
    path = kagglehub.dataset_download("shilongzhuang/telecom-customer-churn-by-maven-analytics")
    csv_files = sorted([f for f in os.listdir(path) if f.endswith('.csv')])
    data1 = pd.read_csv(os.path.join(path, csv_files[0]), encoding='latin1')
    data2 = pd.read_csv(os.path.join(path, csv_files[2]), encoding='latin1')
    df = pd.merge(data2, data1, on='Zip Code')
    df['Customer Status'] = df['Customer Status'].apply(lambda x: 1 if x == 'Stayed' else 0)
    columns_to_drop = ['Customer ID', 'Churn Category', 'Churn Reason',
                        'Total Charges', 'Total Revenue', 'Total Refunds',
                        'Total Long Distance Charges', 'Zip Code', 'City', 'Latitude', 'Longitude']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    df = pipeline.preprocessing(df)
    label = "Customer Status"
    df[label] = df[label].astype(int)

    # This dataset already has natural nulls (Offer, Internet Type, etc.)
    # The original script also augmented Contract/Monthly Charge/Payment Method
    # but the best combo was just 'Offer' ext, so we use natural nulls
    return df, label, {'mode': 'natural'}


def load_movie_aug_v2():
    """Movie (Augmented v2 — Heavy). MovieLens 20M with user-level aggregation."""
    print("Downloading MovieLens 20M dataset...")
    path = kagglehub.dataset_download("grouplens/movielens-20m-dataset")
    ratings = pd.read_csv(os.path.join(path, "rating.csv"))
    tags = pd.read_csv(os.path.join(path, "tag.csv"))

    # Build user-level features
    user_stats = ratings.groupby('userId')['rating'].agg(['count', 'mean', 'std']).reset_index()
    user_stats.columns = ['userId', 'rating_count', 'rating_mean', 'rating_std']

    # First/last rating timestamps
    user_time = ratings.groupby('userId')['timestamp'].agg(['min', 'max']).reset_index()
    user_time.columns = ['userId', 'first_rating', 'last_rating']
    user_stats = user_stats.merge(user_time, on='userId')

    # Days active and frequency
    user_stats['days_active'] = (user_stats['last_rating'] - user_stats['first_rating']) / 86400
    user_stats['rating_frequency'] = user_stats['rating_count'] / (user_stats['days_active'] + 1)

    # Tag count per user
    tag_count = tags.groupby('userId').size().reset_index(name='tag_count')
    user_stats = user_stats.merge(tag_count, on='userId', how='left')
    user_stats['tag_count'] = user_stats['tag_count'].fillna(0)

    # Binary label: high rater (mean > 3.5)
    label = "high_rater"
    user_stats[label] = (user_stats['rating_mean'] > 3.5).astype(int)

    # Drop userId and timestamps
    df = user_stats.drop(columns=['userId', 'first_rating', 'last_rating'])
    df = pipeline.preprocessing(df)
    df[label] = df[label].astype(int)

    # Heavy augmentation: inject 50% nulls independently per feature
    augment_features = ['rating_count', 'rating_mean', 'rating_std', 'days_active', 'rating_frequency']
    rng = np.random.RandomState(SEED)
    n_null = int(0.50 * len(df))
    for feat in augment_features:
        if feat in df.columns:
            null_idx = rng.choice(df.index, size=n_null, replace=False)
            df.loc[null_idx, feat] = np.nan

    return df, label, {
        'mode': 'augmented',
        'inject_features': [c for c in augment_features if c in df.columns],
    }


def load_weatheraus():
    """WeatherAUS (Natural Nulls)."""
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'weatherAUS.csv')
    df = pd.read_csv(csv_path)
    label = "RainTomorrow"
    df.dropna(subset=[label], inplace=True)
    df.drop(columns=['Date', 'Location', 'RainToday', 'RISK_MM'], errors='ignore', inplace=True)
    df = pipeline.preprocessing(df)
    df[label] = df[label].astype(int)
    return df, label, {'mode': 'natural'}


def load_wids():
    """WIDS (Natural Nulls)."""
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'WIDS.csv')
    df = pd.read_csv(csv_path)
    label = "hospital_death"
    drop_cols = ['encounter_id', 'patient_id', 'hospital_id', 'icu_id',
                 'Unnamed: 0', 'readmission_status']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    df = pipeline.preprocessing(df)
    df[label] = df[label].astype(int)
    return df, label, {'mode': 'natural'}


def load_flight_delay():
    """FlightDelay (Augmented)."""
    path = kagglehub.dataset_download("divyansh22/flight-delay-prediction")
    csv_files = sorted([f for f in os.listdir(path) if f.endswith('.csv')])
    csv_path = os.path.join(path, csv_files[0])
    df = pd.read_csv(csv_path)
    label = "DEP_DEL15"
    df.dropna(subset=[label], inplace=True)

    drop_cols = ['Unnamed: 0', 'FL_DATE', 'TAIL_NUM', 'DEP_TIME',
                 'DEP_DELAY', 'ARR_TIME', 'ARR_DELAY', 'ARR_DEL15',
                 'CANCELLED', 'DIVERTED', 'ACTUAL_ELAPSED_TIME',
                 'DISTANCE_GROUP']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    df = pipeline.preprocessing(df)
    df[label] = df[label].astype(int)

    # Sample to 100K for performance
    if len(df) > 100000:
        df = df.sample(n=100000, random_state=SEED).reset_index(drop=True)

    inject_features = ['OP_CARRIER_FL_NUM']
    inject_rate = 0.30
    np.random.seed(SEED)
    mask = np.random.rand(len(df)) < inject_rate
    for col in inject_features:
        if col in df.columns:
            df.loc[mask, col] = np.nan

    return df, label, {
        'mode': 'augmented',
        'inject_features': [c for c in inject_features if c in df.columns],
    }


# ============================================================================
# Dataset registry
# ============================================================================

DATASETS = [
    ('BankLoanSta',     load_bankloansta),
    ('Weather',         load_weather),
    ('DiabetesRecord',  load_diabetes),
    ('HRAnalytics',     load_hr_analytics),
    ('ClientRecordAug', load_client_record_aug),
    ('MovieAugV2',      load_movie_aug_v2),
    ('WeatherAUS',      load_weatheraus),
    ('WIDS',            load_wids),
    ('FlightDelay',     load_flight_delay),
]


# ============================================================================
# Helper: find null groups and enumerate valid combos
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

    # Sort by null rate descending, limit
    groups.sort(key=lambda g: df[g[0]].isnull().mean(), reverse=True)
    return groups[:max_groups]


def enumerate_combos(df, label, groups):
    """Enumerate all valid feature combos from groups, dedup by population."""
    from itertools import combinations

    all_combos = []
    for r in range(1, len(groups) + 1):
        for combo in combinations(range(len(groups)), r):
            ext_features = []
            for idx in combo:
                ext_features.extend(groups[idx])
            all_combos.append(ext_features)

    # Validate and dedup
    valid_combos = []
    seen_populations = set()
    for ext_features in all_combos:
        has_ext = df[ext_features].notna().all(axis=1)
        n_ext = has_ext.sum()
        n_no = (~has_ext).sum()

        # Minimum group sizes
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
# Helper: run a single combo experiment
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

    # Use consistent feature column order everywhere
    # Must be sorted to ensure same order in base model, extended, combined, and ablation
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

    n_total = len(test_with) + len(test_without)
    objective = (len(test_without) * (comb_no_auc - base_auc) + len(test_with) * (comb_ext_auc - ext_auc)) / n_total

    return {
        'objective': objective,
        'base_auc': base_auc,
        'ext_auc': ext_auc,
        'comb_no_auc': comb_no_auc,
        'comb_ext_auc': comb_ext_auc,
        'n_test_no': len(test_without),
        'n_test_ext': len(test_with),
        'ext_features': ext_features,
        'dm': dm,  # keep for ablation reuse
    }


# ============================================================================
# Helper: run ablation on best combo (no_pruning + fixed_50)
# ============================================================================

def run_ablation(dm, ext_train, test_with, test_without, label,
                 base_auc, comb_no_auc, comb_ext_auc,
                 ds_name, n_trials_ablation):
    """Run no_pruning and fixed_50 ablation using already-trained base+combined."""
    ablation_results = []
    # Use consistent sorted feature order (must match run_combo)
    feature_cols = sorted(dm.base_features + dm.ext_features)

    for mode in ['no_pruning', 'fixed_50']:
        print(f"\n{'='*80}", flush=True)
        print(f"ABLATION — {ds_name} | {mode} (n_trials={n_trials_ablation})", flush=True)
        print(f"{'='*80}", flush=True)

        dm.extended_model = XGBoostModel(name=f"extended_{mode}")
        dm.extended_model.train(
            ext_train[feature_cols],
            ext_train[dm.label],
            base_model_path="base_model.json",
            n_trials=n_trials_ablation,
            pruning_mode=mode
        )

        ext_auc = roc_auc_score(test_with[label],
                                 dm.extended_model.predict(test_with[feature_cols]))
        n_total = len(test_with) + len(test_without)
        objective = (len(test_without) * (comb_no_auc - base_auc) + len(test_with) * (comb_ext_auc - ext_auc)) / n_total

        print(f"\n>>> {ds_name} | {mode}: objective={objective:.6f}", flush=True)
        print(f"    Ext AUC: {ext_auc:.6f}", flush=True)

        ablation_results.append({
            'dataset': ds_name,
            'mode': mode,
            'objective': objective,
            'ext_auc': ext_auc,
            'base_auc': base_auc,
            'comb_no_auc': comb_no_auc,
            'comb_ext_auc': comb_ext_auc,
        })

    return ablation_results


# ============================================================================
# Main loop
# ============================================================================

print("=" * 100)
print(f"RUNNING ALL EXPERIMENTS + ABLATION — N_TRIALS={N_TRIALS}, ABLATION_TRIALS={ABLATION_TRIALS}")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
if DATASET_FILTER:
    print(f"Filter: {DATASET_FILTER}")
print("=" * 100)

all_best = []       # best combo per dataset
all_ablation = []   # ablation results

for ds_name, load_fn in DATASETS:
    if DATASET_FILTER and ds_name not in DATASET_FILTER:
        print(f"\nSkipping {ds_name} (not in filter)")
        continue

    print(f"\n{'#'*100}")
    print(f"# DATASET: {ds_name}")
    print(f"# Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'#'*100}", flush=True)

    start_time = time.time()
    ds_results_dir = os.path.join(RESULTS_DIR, ds_name)
    os.makedirs(ds_results_dir, exist_ok=True)

    # Load data
    try:
        df, label, config = load_fn()
    except Exception as e:
        print(f"ERROR loading {ds_name}: {e}")
        continue

    print(f"Shape: {df.shape}, Label: {label}")
    print(f"Target distribution:\n{df[label].value_counts()}", flush=True)

    # ---- Phase 1: Find all valid combos ----
    if config['mode'] == 'natural':
        groups = find_null_groups(df, label)
        if not groups:
            print(f"No null groups found for {ds_name}, skipping")
            continue
        print(f"\nFound {len(groups)} null groups")
        for i, g in enumerate(groups):
            null_rate = df[g[0]].isnull().mean()
            print(f"  Group {i+1}: {g[0][:40]}... ({len(g)} feats, {null_rate:.1%} null)")

        combos = enumerate_combos(df, label, groups)
    else:
        # Augmented: use inject_features as a single combo list
        inject_features = config['inject_features']
        # Generate combos from subsets of injected features
        from itertools import combinations
        combos = []
        seen_populations = set()
        for r in range(1, len(inject_features) + 1):
            for combo in combinations(inject_features, r):
                ext_features = list(combo)
                has_ext = df[ext_features].notna().all(axis=1)
                n_ext = has_ext.sum()
                n_no = (~has_ext).sum()
                if n_ext < 100 or n_no < 100:
                    continue
                pop_key = (n_no, n_ext)
                if pop_key in seen_populations:
                    continue
                seen_populations.add(pop_key)
                combos.append({
                    'ext_features': ext_features,
                    'n_ext': n_ext,
                    'n_no': n_no,
                    'ext_pct': n_ext / len(df) * 100,
                })

    print(f"\n{len(combos)} valid combos to evaluate")

    # Print combo table
    print(f"\n{'#':<4} {'Name':<50} {'N_no':>8} {'N_ext':>8} {'%ext':>8}")
    print("-" * 82)
    for i, c in enumerate(combos):
        name = "+".join([f[:6] for f in c['ext_features']])
        print(f"{i+1:<4} {name:<50} {c['n_no']:>8} {c['n_ext']:>8} {c['ext_pct']:>7.1f}%")
    print(flush=True)

    # ---- Phase 2: Run all combos ----
    best_result = None
    best_combo_idx = -1

    for i, combo in enumerate(combos):
        ext_features = combo['ext_features']
        combo_name = "+".join([f[:6] for f in ext_features])

        print(f"\n{'='*80}")
        print(f"Combo {i+1}/{len(combos)}: {combo_name} ext")
        print(f"{'='*80}", flush=True)

        result = run_combo(df, label, ext_features, N_TRIALS)

        if result is None:
            print(f">>> RESULT: SKIPPED (invalid split)")
            continue

        print(f"\n>>> RESULT: objective = {result['objective']:.6f}")
        print(f"    Base AUC: {result['base_auc']:.6f} ({result['n_test_no']} rows)")
        print(f"    Extended AUC: {result['ext_auc']:.6f} ({result['n_test_ext']} rows)")
        print(f"    Combined (no ext) AUC: {result['comb_no_auc']:.6f}")
        print(f"    Combined (with ext) AUC: {result['comb_ext_auc']:.6f}", flush=True)

        if best_result is None or result['objective'] < best_result['objective']:
            best_result = result
            best_combo_idx = i

    if best_result is None:
        print(f"\nNo valid combos for {ds_name}")
        continue

    best_ext = best_result['ext_features']
    best_name = "+".join([f[:6] for f in best_ext])
    elapsed = time.time() - start_time

    print(f"\n{'*'*80}")
    print(f"BEST COMBO for {ds_name}: {best_name}")
    print(f"  Objective: {best_result['objective']:.6f}")
    print(f"  Base AUC: {best_result['base_auc']:.4f}, Ext AUC: {best_result['ext_auc']:.4f}")
    print(f"  Comb-no: {best_result['comb_no_auc']:.4f}, Comb+ext: {best_result['comb_ext_auc']:.4f}")
    print(f"  Time: {elapsed:.0f}s")
    print(f"{'*'*80}", flush=True)

    all_best.append({
        'dataset': ds_name,
        'combo_name': best_name,
        'ext_features': best_ext,
        'objective': best_result['objective'],
        'base_auc': best_result['base_auc'],
        'ext_auc': best_result['ext_auc'],
        'comb_no_auc': best_result['comb_no_auc'],
        'comb_ext_auc': best_result['comb_ext_auc'],
        'n_test_no': best_result['n_test_no'],
        'n_test_ext': best_result['n_test_ext'],
        'n_combos': len(combos),
        'elapsed': elapsed,
    })

    # ---- Phase 3: Ablation on best combo ----
    print(f"\n{'#'*80}")
    print(f"# ABLATION for {ds_name} — best combo: {best_name}")
    print(f"{'#'*80}", flush=True)

    dm = best_result['dm']
    ext_train = dm.ext_df[dm.ext_df['has_extended'] == 1].copy()
    test_with = dm.test_df[dm.test_df['has_extended'] == 1]
    test_without = dm.test_df[dm.test_df['has_extended'] == 0]

    ablation_res = run_ablation(
        dm=dm,
        ext_train=ext_train,
        test_with=test_with,
        test_without=test_without,
        label=label,
        base_auc=best_result['base_auc'],
        comb_no_auc=best_result['comb_no_auc'],
        comb_ext_auc=best_result['comb_ext_auc'],
        ds_name=ds_name,
        n_trials_ablation=ABLATION_TRIALS,
    )
    all_ablation.extend(ablation_res)

    # Also add optuna result to ablation for comparison
    all_ablation.append({
        'dataset': ds_name,
        'mode': 'optuna',
        'objective': best_result['objective'],
        'ext_auc': best_result['ext_auc'],
        'base_auc': best_result['base_auc'],
        'comb_no_auc': best_result['comb_no_auc'],
        'comb_ext_auc': best_result['comb_ext_auc'],
    })


# ============================================================================
# Final Summary
# ============================================================================

print(f"\n{'='*120}")
print(f"FINAL SUMMARY — N_TRIALS={N_TRIALS}")
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*120}")

# --- Best combo per dataset ---
print(f"\n{'Dataset':<20} {'Best Combo':<30} {'Objective':>10} {'Base AUC':>10} {'Ext AUC':>10} {'Comb-no':>10} {'Comb+ext':>10} {'N_no':>8} {'N_ext':>8} {'Time':>8}")
print("-" * 140)
for b in all_best:
    winner = "Increm" if b['objective'] < 0 else "Comb"
    print(f"{b['dataset']:<20} {b['combo_name']:<30} {b['objective']:>10.6f} {b['base_auc']:>10.4f} {b['ext_auc']:>10.4f} {b['comb_no_auc']:>10.4f} {b['comb_ext_auc']:>10.4f} {b['n_test_no']:>8} {b['n_test_ext']:>8} {b['elapsed']:>7.0f}s")

n_increm = sum(1 for b in all_best if b['objective'] < 0)
print(f"\nIncremental wins: {n_increm}/{len(all_best)} datasets")

# --- Ablation results ---
print(f"\n{'='*120}")
print("ABLATION: PRUNING MODE COMPARISON (no_pruning, fixed_50 vs optuna)")
print(f"{'='*120}")
print(f"{'Dataset':<20} {'Mode':<14} {'Objective':>12} {'Ext AUC':>10} {'Base AUC':>10} {'Comb+ext':>10}")
print("-" * 80)
for r in sorted(all_ablation, key=lambda x: (x['dataset'], x['mode'])):
    print(f"{r['dataset']:<20} {r['mode']:<14} {r['objective']:>12.6f} {r['ext_auc']:>10.4f} {r['base_auc']:>10.4f} {r['comb_ext_auc']:>10.4f}")

# --- Save summary file ---
summary_path = os.path.join(RESULTS_DIR, f'summary_t{N_TRIALS}.txt')
with open(summary_path, 'w') as f:
    f.write(f"{'='*120}\n")
    f.write(f"INCREMENTAL LEARNING EXPERIMENT + ABLATION SUMMARY\n")
    f.write(f"N_TRIALS={N_TRIALS}, ABLATION_TRIALS={ABLATION_TRIALS}\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"{'='*120}\n\n")
    f.write("Objective = (combined_with_ext_auc - extended_auc) + (combined_no_ext_auc - base_auc)\n")
    f.write("Negative objective = incremental learning approach outperforms combined model.\n\n")

    f.write(f"{'='*80}\n")
    f.write("BEST COMBO PER DATASET\n")
    f.write(f"{'='*80}\n\n")
    f.write(f"{'Dataset':<20} {'Best Combo':<30} {'Objective':>10} {'Base AUC':>10} {'Ext AUC':>10} {'Comb-no':>10} {'Comb+ext':>10} {'N_no':>8} {'N_ext':>8}\n")
    f.write("-" * 130 + "\n")
    for b in all_best:
        f.write(f"{b['dataset']:<20} {b['combo_name']:<30} {b['objective']:>10.6f} {b['base_auc']:>10.4f} {b['ext_auc']:>10.4f} {b['comb_no_auc']:>10.4f} {b['comb_ext_auc']:>10.4f} {b['n_test_no']:>8} {b['n_test_ext']:>8}\n")
    f.write(f"\nIncremental wins: {n_increm}/{len(all_best)} datasets\n")

    f.write(f"\n{'='*80}\n")
    f.write("ABLATION: PRUNING MODE COMPARISON\n")
    f.write(f"{'='*80}\n\n")
    f.write(f"{'Dataset':<20} {'Mode':<14} {'Objective':>12} {'Ext AUC':>10} {'Base AUC':>10} {'Comb+ext':>10}\n")
    f.write("-" * 80 + "\n")
    for r in sorted(all_ablation, key=lambda x: (x['dataset'], x['mode'])):
        f.write(f"{r['dataset']:<20} {r['mode']:<14} {r['objective']:>12.6f} {r['ext_auc']:>10.4f} {r['base_auc']:>10.4f} {r['comb_ext_auc']:>10.4f}\n")

    # Per-dataset comparison
    f.write(f"\n{'='*80}\n")
    f.write("PER-DATASET ABLATION COMPARISON\n")
    f.write(f"{'='*80}\n")
    datasets_seen = []
    for r in all_ablation:
        if r['dataset'] not in datasets_seen:
            datasets_seen.append(r['dataset'])
    for ds in datasets_seen:
        ds_results = [r for r in all_ablation if r['dataset'] == ds]
        f.write(f"\n{ds}:\n")
        best = min(ds_results, key=lambda x: x['objective'])
        for r in sorted(ds_results, key=lambda x: x['mode']):
            marker = " <-- BEST" if r['mode'] == best['mode'] else ""
            f.write(f"  {r['mode']:<14} objective={r['objective']:.6f}  ext_auc={r['ext_auc']:.4f}{marker}\n")

print(f"\nSummary saved to: {summary_path}")
print("Done!")
