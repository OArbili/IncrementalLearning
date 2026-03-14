#!/usr/bin/env python3
"""Dry run: enumerate all 64 combinations, show population sizes, filter invalid."""
import itertools
import pandas as pd
import numpy as np
import kagglehub
import os
from GenericDataPipeline import GenericDataPipeline
from seed_utils import SEED, set_all_seeds
from sklearn.model_selection import train_test_split

pd.set_option('future.infer_string', False)
set_all_seeds()

pipeline = GenericDataPipeline()

path = kagglehub.dataset_download("zaurbegiev/my-dataset")
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
csv_path = os.path.join(path, csv_files[1])
df = pd.read_csv(csv_path)
df = df.dropna(subset=['Loan Status']).reset_index(drop=True)
df = df.drop(columns=['Loan ID', 'Customer ID'])
df = pipeline.preprocessing(df)

label = "Loan Status"

# --- Augment: inject nulls into Home Ownership & Purpose ---
rng = np.random.RandomState(SEED)
null_features = ['Credit Score', 'Annual Income', 'Months since last delinquent', 'Years in current job']
has_any_null = df[null_features].isna().any(axis=1)
candidate_indices = df[has_any_null].index.tolist()
n_sample = min(int(0.20 * len(df)), len(candidate_indices))
sampled_indices = rng.choice(candidate_indices, size=n_sample, replace=False)
choices = rng.choice(['home', 'purpose', 'both'], size=len(sampled_indices))
home_null_idx = sampled_indices[choices != 'purpose']
purpose_null_idx = sampled_indices[choices != 'home']
df.loc[home_null_idx, 'Home Ownership'] = np.nan
df.loc[purpose_null_idx, 'Purpose'] = np.nan

# --- Get all features and null features ---
X = df.drop(label, axis=1)
all_features = list(X.columns)

extended_null_features = []
for col in all_features:
    if df[col].isna().mean() > 0.01:
        extended_null_features.append(col)

print(f"Features eligible for extended (null_ratio > 0.01): {len(extended_null_features)}")
for f in extended_null_features:
    print(f"  {f}: {df[f].isna().mean():.2%}")

print(f"\nTotal combinations: {2**len(extended_null_features)}")
print()

# --- Enumerate all combinations ---
results = []

for combo_idx, bits in enumerate(itertools.product([0, 1], repeat=len(extended_null_features))):
    assignment = {}
    for feat, bit in zip(extended_null_features, bits):
        assignment[feat] = bit  # 1=base, 0=extended

    for feat in all_features:
        if feat not in assignment:
            assignment[feat] = 1

    base_feats = [f for f, a in assignment.items() if a == 1]
    ext_feats = [f for f, a in assignment.items() if a == 0]

    # Check: all features extended → no base features
    vals = sum(assignment.values())
    if vals == 0:
        results.append({
            'combo_idx': combo_idx + 1,
            'assignment': {f: assignment[f] for f in extended_null_features},
            'status': 'SKIP (all extended)',
            'n_total': 100000,
            'n_with_ext': None,
            'n_no_ext': None,
            'pct_ext': None,
            'n_test_with_ext': None,
            'n_test_no_ext': None,
            'min_group': None,
        })
        continue

    # Check: no extended features
    if len(ext_feats) == 0:
        results.append({
            'combo_idx': combo_idx + 1,
            'assignment': {f: assignment[f] for f in extended_null_features},
            'status': 'SKIP (no extended)',
            'n_total': 100000,
            'n_with_ext': 0,
            'n_no_ext': 100000,
            'pct_ext': 0,
            'n_test_with_ext': None,
            'n_test_no_ext': None,
            'min_group': None,
        })
        continue

    # Compute has_extended
    has_extended = df[ext_feats].notnull().any(axis=1).astype(int)
    n_with_ext = has_extended.sum()
    n_no_ext = len(df) - n_with_ext

    # Try stratified split to check group sizes
    strat_col = df[label].astype(str) + '_' + has_extended.astype(str)

    if has_extended.nunique() < 2:
        results.append({
            'combo_idx': combo_idx + 1,
            'assignment': {f: assignment[f] for f in extended_null_features},
            'status': 'SKIP (no variation)',
            'n_total': 100000,
            'n_with_ext': int(n_with_ext),
            'n_no_ext': int(n_no_ext),
            'pct_ext': round(n_with_ext / len(df) * 100, 2),
            'n_test_with_ext': None,
            'n_test_no_ext': None,
            'min_group': 0,
        })
        continue

    if strat_col.value_counts().min() < 2:
        results.append({
            'combo_idx': combo_idx + 1,
            'assignment': {f: assignment[f] for f in extended_null_features},
            'status': 'SKIP (strat < 2)',
            'n_total': 100000,
            'n_with_ext': int(n_with_ext),
            'n_no_ext': int(n_no_ext),
            'pct_ext': round(n_with_ext / len(df) * 100, 2),
            'n_test_with_ext': None,
            'n_test_no_ext': None,
            'min_group': int(strat_col.value_counts().min()),
        })
        continue

    # Do the split
    temp_df = df.copy()
    temp_df['has_extended'] = has_extended
    temp_df['strat_col'] = strat_col

    _, X_test, _, y_test = train_test_split(
        temp_df.drop(label, axis=1),
        temp_df[label],
        test_size=0.2,
        random_state=SEED,
        stratify=temp_df['strat_col']
    )
    test_df = pd.concat([X_test, y_test], axis=1)
    test_with = test_df[test_df['has_extended'] == 1]
    test_without = test_df[test_df['has_extended'] == 0]

    train_df = temp_df.drop(test_df.index)
    train_counts = train_df.groupby([label, 'has_extended']).size()
    test_counts = test_df.groupby([label, 'has_extended']).size()
    min_group = min(train_counts.min(), test_counts.min())

    if min_group < 100:
        status = 'SKIP (group < 100)'
    else:
        status = 'VALID'

    results.append({
        'combo_idx': combo_idx + 1,
        'assignment': {f: assignment[f] for f in extended_null_features},
        'status': status,
        'n_total': 100000,
        'n_with_ext': int(n_with_ext),
        'n_no_ext': int(n_no_ext),
        'pct_ext': round(n_with_ext / len(df) * 100, 2),
        'n_test_with_ext': len(test_with),
        'n_test_no_ext': len(test_without),
        'min_group': int(min_group),
    })

# --- Print results ---
valid = [r for r in results if r['status'] == 'VALID']
skipped = [r for r in results if r['status'] != 'VALID']

print(f"{'='*120}")
print(f"VALID COMBINATIONS: {len(valid)} out of {len(results)}")
print(f"{'='*120}")
header = f"{'#':<4} {'CS':<5} {'AI':<5} {'MLD':<5} {'YCJ':<5} {'HO':<5} {'Purp':<5} {'N_no_ext':<10} {'N_ext':<10} {'%ext':<8} {'Test_no':<10} {'Test_ext':<10} {'MinGrp':<8}"
print(header)
print("-" * 120)

for r in valid:
    a = r['assignment']
    cs = 'ext' if a.get('Credit Score', 1) == 0 else 'base'
    ai = 'ext' if a.get('Annual Income', 1) == 0 else 'base'
    mld = 'ext' if a.get('Months since last delinquent', 1) == 0 else 'base'
    ycj = 'ext' if a.get('Years in current job', 1) == 0 else 'base'
    ho = 'ext' if a.get('Home Ownership', 1) == 0 else 'base'
    pur = 'ext' if a.get('Purpose', 1) == 0 else 'base'
    print(f"{r['combo_idx']:<4} {cs:<5} {ai:<5} {mld:<5} {ycj:<5} {ho:<5} {pur:<5} {r['n_no_ext']:<10} {r['n_with_ext']:<10} {r['pct_ext']:<8} {r['n_test_no_ext']:<10} {r['n_test_with_ext']:<10} {r['min_group']:<8}")

print(f"\n{'='*120}")
print(f"SKIPPED COMBINATIONS: {len(skipped)}")
print(f"{'='*120}")
for r in skipped:
    a = r['assignment']
    cs = 'ext' if a.get('Credit Score', 1) == 0 else 'base'
    ai = 'ext' if a.get('Annual Income', 1) == 0 else 'base'
    mld = 'ext' if a.get('Months since last delinquent', 1) == 0 else 'base'
    ycj = 'ext' if a.get('Years in current job', 1) == 0 else 'base'
    ho = 'ext' if a.get('Home Ownership', 1) == 0 else 'base'
    pur = 'ext' if a.get('Purpose', 1) == 0 else 'base'
    n_no = r['n_no_ext'] if r['n_no_ext'] is not None else 'N/A'
    n_ext = r['n_with_ext'] if r['n_with_ext'] is not None else 'N/A'
    print(f"{r['combo_idx']:<4} {cs:<5} {ai:<5} {mld:<5} {ycj:<5} {ho:<5} {pur:<5} {str(n_no):<10} {str(n_ext):<10} {r['status']}")
