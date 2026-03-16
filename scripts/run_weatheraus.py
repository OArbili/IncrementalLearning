#!/usr/bin/env python3
"""Run all valid combinations on WeatherAUS dataset (local CSV)."""
import itertools
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
from core.GenericDataPipeline import GenericDataPipeline
from core.RunData import RunPipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from core.seed_utils import SEED, set_all_seeds

pd.set_option('future.infer_string', False)
set_all_seeds()

# --- Config ---
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
NULL_THRESHOLD = 0.05  # Features with >5% nulls are candidates for extended
MIN_GROUP_PCT = 0.02   # Min 2% of total population per test group

pipeline = GenericDataPipeline()

# --- Load data ---
csv_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'weatherAUS.csv')
df = pd.read_csv(csv_path, na_values=['NA'])

columns_to_drop = ['Date', 'Location', 'RISK_MM']
df = df.drop(columns=columns_to_drop)
df = pipeline.preprocessing(df)

label = "RainTomorrow"
print(f"Dataset shape: {df.shape}")
print(f"N_TRIALS per model: {N_TRIALS}")

# --- Identify candidate extended features ---
X = df.drop(label, axis=1)
all_features = list(X.columns)

null_features = []
for col in all_features:
    nr = df[col].isna().mean()
    if nr > NULL_THRESHOLD:
        null_features.append(col)
        print(f"  Candidate: {col} (null: {nr:.2%})")

print(f"\n{len(null_features)} candidate features -> {2**len(null_features)} total combinations")

# --- Check for duplicate null patterns ---
null_masks = {}
for col in null_features:
    null_masks[col] = frozenset(df[df[col].isna()].index.tolist())

print("\nNull pattern overlap:")
for i, f1 in enumerate(null_features):
    for f2 in null_features[i+1:]:
        overlap = len(null_masks[f1] & null_masks[f2])
        union = len(null_masks[f1] | null_masks[f2])
        jaccard = overlap / union if union > 0 else 0
        if jaccard > 0.95:
            print(f"  {f1} ~ {f2}: Jaccard={jaccard:.4f} (nearly identical!)")

# --- Dry run: enumerate all valid combinations ---
print(f"\n{'='*100}")
print("DRY RUN: Finding valid combinations")
print(f"{'='*100}")

min_group_size = int(MIN_GROUP_PCT * len(df))
valid_combos = []

for bits in itertools.product([0, 1], repeat=len(null_features)):
    ext_feats = [f for f, b in zip(null_features, bits) if b == 0]
    base_feats_check = [f for f in all_features if f not in ext_feats]

    # Skip: no extended features
    if len(ext_feats) == 0:
        continue
    # Skip: no base features
    if len(base_feats_check) == 0:
        continue

    # Compute has_extended
    has_extended = df[ext_feats].notnull().any(axis=1).astype(int)
    n_with = has_extended.sum()
    n_without = len(df) - n_with

    if has_extended.nunique() < 2:
        continue

    # Stratified split check
    strat_col = df[label].astype(str) + '_' + has_extended.astype(str)
    if strat_col.value_counts().min() < 2:
        continue

    _, X_test, _, y_test = train_test_split(
        df.drop(label, axis=1), df[label],
        test_size=0.2, random_state=SEED, stratify=strat_col
    )
    test_df = pd.concat([X_test, y_test], axis=1)
    test_df['has_extended'] = has_extended[test_df.index]

    train_df_temp = df.drop(test_df.index)
    train_df_temp['has_extended'] = has_extended[train_df_temp.index]

    train_counts = train_df_temp.groupby([label, 'has_extended']).size()
    test_counts = test_df.groupby([label, 'has_extended']).size()
    min_group = min(train_counts.min(), test_counts.min())

    if min_group < 100:
        continue

    test_with = len(test_df[test_df['has_extended'] == 1])
    test_without = len(test_df[test_df['has_extended'] == 0])

    if test_with < min_group_size or test_without < min_group_size:
        continue

    ext_names = '+'.join([f.replace(' ', '')[:6] for f in ext_feats])
    valid_combos.append({
        'ext_feats': ext_feats,
        'name': f"{ext_names} ext",
        'n_with': int(n_with),
        'n_without': int(n_without),
        'pct_ext': round(n_with / len(df) * 100, 1),
        'test_with': test_with,
        'test_without': test_without,
        'min_group': int(min_group),
    })

# --- Deduplicate equivalent combos (identical has_extended populations) ---
seen_populations = {}
unique_combos = []
for combo in valid_combos:
    ext_feats = combo['ext_feats']
    has_ext = frozenset(df[df[ext_feats].notnull().any(axis=1)].index.tolist())
    if has_ext not in seen_populations:
        seen_populations[has_ext] = combo['name']
        unique_combos.append(combo)
    else:
        print(f"  Dedup: '{combo['name']}' same population as '{seen_populations[has_ext]}'")

print(f"\nValid: {len(valid_combos)} -> Unique: {len(unique_combos)}")
print(f"\n{'#':<4} {'Name':<45} {'N_no_ext':<10} {'N_ext':<10} {'%ext':<8} {'Test_no':<10} {'Test_ext':<10} {'MinGrp':<8}")
print("-" * 100)
for i, c in enumerate(unique_combos):
    print(f"{i+1:<4} {c['name']:<45} {c['n_without']:<10} {c['n_with']:<10} {c['pct_ext']:<8} {c['test_without']:<10} {c['test_with']:<10} {c['min_group']:<8}")

sys.stdout.flush()

# --- Run each valid unique combination ---
print(f"\n{'='*100}")
print(f"RUNNING {len(unique_combos)} UNIQUE COMBINATIONS (n_trials={N_TRIALS})")
print(f"{'='*100}")

results = []

for i, combo in enumerate(unique_combos):
    ext_feats = combo['ext_feats']
    base_feats = [f for f in all_features if f not in ext_feats]

    print(f"\n{'='*80}")
    print(f"Combo {i+1}/{len(unique_combos)}: {combo['name']}")
    print(f"Extended: {ext_feats}")
    print(f"{'='*80}")
    sys.stdout.flush()

    dm = RunPipeline()
    objective_val = dm.full_run(df.copy(), base_feats, ext_feats, label,
                                 f"weatheraus_combo_{i+1}.csv", n_trials=N_TRIALS)

    if objective_val in [999, 99999]:
        print(f">>> FAILED: {objective_val}")
        results.append({
            'combo': combo['name'],
            'objective': objective_val,
            'base_auc': None, 'ext_auc': None,
            'comb_no_auc': None, 'comb_ext_auc': None,
            'n_test_no': None, 'n_test_ext': None,
        })
    else:
        test_with = dm.test_df[dm.test_df['has_extended'] == 1]
        test_without = dm.test_df[dm.test_df['has_extended'] == 0]

        base_auc = roc_auc_score(test_without[label],
                                  dm.base_model.predict(test_without[dm.all_features]))
        ext_auc = roc_auc_score(test_with[label],
                                 dm.extended_model.predict(test_with[dm.all_features]))
        comb_no_auc = roc_auc_score(test_without[label],
                                     dm.combined_model.predict(test_without[dm.all_features]))
        comb_ext_auc = roc_auc_score(test_with[label],
                                      dm.combined_model.predict(test_with[dm.all_features]))

        print(f"\n>>> RESULT: objective = {objective_val:.6f}")
        print(f"    Base AUC: {base_auc:.4f} ({len(test_without)} rows)")
        print(f"    Extended AUC: {ext_auc:.4f} ({len(test_with)} rows)")
        print(f"    Combined (no ext) AUC: {comb_no_auc:.4f}")
        print(f"    Combined (with ext) AUC: {comb_ext_auc:.4f}")

        results.append({
            'combo': combo['name'],
            'objective': objective_val,
            'base_auc': base_auc, 'ext_auc': ext_auc,
            'comb_no_auc': comb_no_auc, 'comb_ext_auc': comb_ext_auc,
            'n_test_no': len(test_without), 'n_test_ext': len(test_with),
        })

    sys.stdout.flush()

# --- Final summary ---
print(f"\n{'='*120}")
print("FINAL SUMMARY")
print(f"{'='*120}")
print(f"{'Combo':<45} {'Objective':<12} {'Base AUC':<10} {'Ext AUC':<10} {'Comb-no':<10} {'Comb+ext':<10} {'N_no_ext':<10} {'N_ext':<10}")
print("-" * 117)
for r in results:
    obj = f"{r['objective']:.6f}" if r['objective'] not in [999, 99999] else str(int(r['objective']))
    base = f"{r['base_auc']:.4f}" if r['base_auc'] is not None else "N/A"
    ext = f"{r['ext_auc']:.4f}" if r['ext_auc'] is not None else "N/A"
    cno = f"{r['comb_no_auc']:.4f}" if r['comb_no_auc'] is not None else "N/A"
    cex = f"{r['comb_ext_auc']:.4f}" if r['comb_ext_auc'] is not None else "N/A"
    nno = str(r['n_test_no']) if r['n_test_no'] is not None else "N/A"
    nex = str(r['n_test_ext']) if r['n_test_ext'] is not None else "N/A"
    print(f"{r['combo']:<45} {obj:<12} {base:<10} {ext:<10} {cno:<10} {cex:<10} {nno:<10} {nex:<10}")

# Count negative objectives
neg = [r for r in results if isinstance(r['objective'], float) and r['objective'] < 0]
total_valid = [r for r in results if r['objective'] not in [999, 99999]]
print(f"\nNegative objectives (incremental wins): {len(neg)}/{len(total_valid)}")
print("\nDone!")
