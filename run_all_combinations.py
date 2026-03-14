#!/usr/bin/env python3
"""Run all 2^4 feature assignment combinations and print objective values."""
import itertools
import pandas as pd
import numpy as np
import kagglehub
import os
import sys
from GenericDataPipeline import GenericDataPipeline
from RunData import RunPipeline
from seed_utils import SEED, set_all_seeds

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
print(f"Dataset shape: {df.shape}")

X = df.drop(label, axis=1)
y = df[label]

feature_scores = pipeline.rank_features(X, y)
features_with_nulls = feature_scores[feature_scores['null_ratio'] > 0.01]
all_features = feature_scores['feature_name'].to_list()

null_features = features_with_nulls['feature_name'].to_list()
print(f"\nFeatures with nulls: {null_features}")
print(f"Total combinations: {2**len(null_features)}")

# Store results
results = []

# Enumerate all 2^N combinations
for combo_idx, bits in enumerate(itertools.product([0, 1], repeat=len(null_features))):
    assignment = {}
    for feat, bit in zip(null_features, bits):
        assignment[feat] = bit  # 1=base, 0=extended

    # All non-null features go to base
    for feat in all_features:
        if feat not in assignment:
            assignment[feat] = 1

    base_features = [f for f, a in assignment.items() if a == 1]
    ext_features = [f for f, a in assignment.items() if a == 0]

    combo_str = ", ".join([f"{f}={'base' if assignment[f]==1 else 'ext'}" for f in null_features])
    print(f"\n{'='*80}")
    print(f"Combination {combo_idx+1}/{2**len(null_features)}: {combo_str}")
    print(f"Base features ({len(base_features)}): {base_features}")
    print(f"Extended features ({len(ext_features)}): {ext_features}")
    print(f"{'='*80}")

    # Check if all features are extended (no base features from null set, but non-null features are still base)
    vals = sum(assignment.values())
    if vals == 0:
        print("SKIP: All features assigned to extended (no base features)")
        results.append({
            'combo': combo_str,
            'base_features': str(base_features),
            'ext_features': str(ext_features),
            'objective': 99999,
            'base_auc': None,
            'extended_auc': None,
            'combined_no_ext_auc': None,
            'combined_with_ext_auc': None,
            'n_test_with_ext': None,
            'n_test_without_ext': None,
            'status': 'all_extended'
        })
        continue

    dm = RunPipeline()
    dm.load_data(base_features, ext_features, df.copy(), label)
    dm.set_has_extended()
    split_result = dm.train_test_split()

    if split_result == 999:
        print("SKIP: Not enough samples in groups")
        results.append({
            'combo': combo_str,
            'base_features': str(base_features),
            'ext_features': str(ext_features),
            'objective': 999,
            'base_auc': None,
            'extended_auc': None,
            'combined_no_ext_auc': None,
            'combined_with_ext_auc': None,
            'n_test_with_ext': None,
            'n_test_without_ext': None,
            'status': 'insufficient_samples'
        })
        continue

    dm.set_train_base_ext_datasets()
    train_ok = dm.train_all()

    if not train_ok:
        print("SKIP: Training failed")
        results.append({
            'combo': combo_str,
            'base_features': str(base_features),
            'ext_features': str(ext_features),
            'objective': 999,
            'base_auc': None,
            'extended_auc': None,
            'combined_no_ext_auc': None,
            'combined_with_ext_auc': None,
            'n_test_with_ext': None,
            'n_test_without_ext': None,
            'status': 'training_failed'
        })
        continue

    test_with_ext = dm.test_df[dm.test_df['has_extended'] == 1]
    test_without_ext = dm.test_df[dm.test_df['has_extended'] == 0]

    objective_val = dm.test_all()

    # Extract individual AUCs from the models
    from sklearn.metrics import roc_auc_score
    y_true_no = test_without_ext[label]
    y_true_ext = test_with_ext[label]

    base_auc = roc_auc_score(y_true_no, dm.base_model.predict(test_without_ext[dm.all_features]))
    ext_auc = roc_auc_score(y_true_ext, dm.extended_model.predict(test_with_ext[dm.all_features]))
    comb_no_auc = roc_auc_score(y_true_no, dm.combined_model.predict(test_without_ext[dm.all_features]))
    comb_ext_auc = roc_auc_score(y_true_ext, dm.combined_model.predict(test_with_ext[dm.all_features]))

    results.append({
        'combo': combo_str,
        'base_features': str(base_features),
        'ext_features': str(ext_features),
        'objective': objective_val,
        'base_auc': base_auc,
        'extended_auc': ext_auc,
        'combined_no_ext_auc': comb_no_auc,
        'combined_with_ext_auc': comb_ext_auc,
        'n_test_with_ext': len(test_with_ext),
        'n_test_without_ext': len(test_without_ext),
        'status': 'success'
    })

    print(f"\n>>> RESULT: objective = {objective_val:.6f}")
    print(f"    Base AUC: {base_auc:.4f} ({len(test_without_ext)} rows)")
    print(f"    Extended AUC: {ext_auc:.4f} ({len(test_with_ext)} rows)")
    print(f"    Combined (no ext) AUC: {comb_no_auc:.4f}")
    print(f"    Combined (with ext) AUC: {comb_ext_auc:.4f}")
    sys.stdout.flush()

# Final summary
print("\n" + "="*100)
print("FINAL SUMMARY: All Combinations")
print("="*100)
results_df = pd.DataFrame(results)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

# Print clean summary table
print(f"\n{'Combo':<70} {'Status':<20} {'Objective':<12} {'Base AUC':<10} {'Ext AUC':<10} {'Comb-noExt':<12} {'Comb+Ext':<10} {'N no_ext':<10} {'N ext':<10}")
print("-"*164)
for _, row in results_df.iterrows():
    obj = f"{row['objective']:.6f}" if row['objective'] not in [999, 99999] else str(int(row['objective']))
    base = f"{row['base_auc']:.4f}" if row['base_auc'] is not None else "N/A"
    ext = f"{row['extended_auc']:.4f}" if row['extended_auc'] is not None else "N/A"
    cno = f"{row['combined_no_ext_auc']:.4f}" if row['combined_no_ext_auc'] is not None else "N/A"
    cex = f"{row['combined_with_ext_auc']:.4f}" if row['combined_with_ext_auc'] is not None else "N/A"
    nno = str(int(row['n_test_without_ext'])) if row['n_test_without_ext'] is not None else "N/A"
    nex = str(int(row['n_test_with_ext'])) if row['n_test_with_ext'] is not None else "N/A"
    print(f"{row['combo']:<70} {row['status']:<20} {obj:<12} {base:<10} {ext:<10} {cno:<12} {cex:<10} {nno:<10} {nex:<10}")

results_df.to_csv("all_combinations_results.csv", index=False)
print("\nResults saved to all_combinations_results.csv")
