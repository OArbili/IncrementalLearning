#!/usr/bin/env python3
"""Run 7 unique augmented combinations on BankLoanSta dataset."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
import kagglehub
from core.GenericDataPipeline import GenericDataPipeline
from core.RunData import RunPipeline
from sklearn.metrics import roc_auc_score
from core.seed_utils import SEED, set_all_seeds

pd.set_option('future.infer_string', False)
set_all_seeds()

# --- Config ---
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
SAMPLE_FRAC = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0  # 1.0 = full data, 0.1 = 10%

pipeline = GenericDataPipeline()

# --- Load data ---
path = kagglehub.dataset_download("zaurbegiev/my-dataset")
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
csv_path = os.path.join(path, csv_files[1])
df = pd.read_csv(csv_path)
df = df.dropna(subset=['Loan Status']).reset_index(drop=True)
df = df.drop(columns=['Loan ID', 'Customer ID'])
df = pipeline.preprocessing(df)

# --- Optional: subsample for testing ---
if SAMPLE_FRAC < 1.0:
    rng_sample = np.random.RandomState(SEED)
    n_rows = int(len(df) * SAMPLE_FRAC)
    df = df.sample(n=n_rows, random_state=SEED).reset_index(drop=True)
    print(f"Subsampled to {len(df)} rows ({SAMPLE_FRAC*100:.0f}%)")

label = "Loan Status"
print(f"Dataset shape: {df.shape}")
print(f"N_TRIALS per model: {N_TRIALS}")

# --- Augment: inject nulls into Home Ownership & Purpose ---
rng = np.random.RandomState(SEED)
null_features_original = ['Credit Score', 'Annual Income', 'Months since last delinquent', 'Years in current job']
has_any_null = df[null_features_original].isna().any(axis=1)
candidate_indices = df[has_any_null].index.tolist()
n_sample = min(int(0.20 * len(df)), len(candidate_indices))
sampled_indices = rng.choice(candidate_indices, size=n_sample, replace=False)
choices = rng.choice(['home', 'purpose', 'both'], size=len(sampled_indices))
home_null_idx = sampled_indices[choices != 'purpose']
purpose_null_idx = sampled_indices[choices != 'home']
df.loc[home_null_idx, 'Home Ownership'] = np.nan
df.loc[purpose_null_idx, 'Purpose'] = np.nan

print(f"\nAugmented null ratios:")
for col in df.columns:
    nr = df[col].isna().mean()
    if nr > 0.01:
        print(f"  {col}: {nr:.2%}")

# --- Feature ranking ---
X = df.drop(label, axis=1)
y = df[label]
feature_scores = pipeline.rank_features(X, y)
all_features = feature_scores['feature_name'].to_list()

# --- Define 7 unique combinations ---
# Format: dict of feature_name -> 0 (extended) or 1 (base)
# Only specifying the 6 null features; all others default to base (1)
null_feats = ['Credit Score', 'Annual Income', 'Months since last delinquent',
              'Years in current job', 'Home Ownership', 'Purpose']

COMBOS = [
    {"name": "CS+AI+MLD ext",
     "ext": ['Credit Score', 'Annual Income', 'Months since last delinquent']},
    {"name": "CS+AI ext",
     "ext": ['Credit Score', 'Annual Income']},
    {"name": "MLD+HO ext",
     "ext": ['Months since last delinquent', 'Home Ownership']},
    {"name": "HO ext only",
     "ext": ['Home Ownership']},
    {"name": "MLD+Purp ext",
     "ext": ['Months since last delinquent', 'Purpose']},
    {"name": "Purp ext only",
     "ext": ['Purpose']},
    {"name": "MLD ext only",
     "ext": ['Months since last delinquent']},
]

# --- Run each combination ---
results = []

for i, combo in enumerate(COMBOS):
    ext_feats = combo['ext']
    base_feats = [f for f in all_features if f not in ext_feats]

    print(f"\n{'='*80}")
    print(f"Combo {i+1}/{len(COMBOS)}: {combo['name']}")
    print(f"Extended: {ext_feats}")
    print(f"{'='*80}")
    sys.stdout.flush()

    dm = RunPipeline()
    objective_val = dm.full_run(df.copy(), base_feats, ext_feats, label,
                                 f"combo_{i+1}.csv", n_trials=N_TRIALS)

    if objective_val in [999, 99999]:
        print(f">>> FAILED: {objective_val}")
        results.append({
            'combo': combo['name'],
            'objective': objective_val,
            'base_auc': None,
            'ext_auc': None,
            'comb_no_auc': None,
            'comb_ext_auc': None,
            'n_test_no': None,
            'n_test_ext': None,
        })
    else:
        # Extract AUCs from the pipeline
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
            'base_auc': base_auc,
            'ext_auc': ext_auc,
            'comb_no_auc': comb_no_auc,
            'comb_ext_auc': comb_ext_auc,
            'n_test_no': len(test_without),
            'n_test_ext': len(test_with),
        })

    sys.stdout.flush()

# --- Final summary ---
print(f"\n{'='*120}")
print("FINAL SUMMARY")
print(f"{'='*120}")
print(f"{'Combo':<20} {'Objective':<12} {'Base AUC':<10} {'Ext AUC':<10} {'Comb-no':<10} {'Comb+ext':<10} {'N_no_ext':<10} {'N_ext':<10}")
print("-" * 92)
for r in results:
    obj = f"{r['objective']:.6f}" if r['objective'] not in [999, 99999] else str(int(r['objective']))
    base = f"{r['base_auc']:.4f}" if r['base_auc'] is not None else "N/A"
    ext = f"{r['ext_auc']:.4f}" if r['ext_auc'] is not None else "N/A"
    cno = f"{r['comb_no_auc']:.4f}" if r['comb_no_auc'] is not None else "N/A"
    cex = f"{r['comb_ext_auc']:.4f}" if r['comb_ext_auc'] is not None else "N/A"
    nno = str(r['n_test_no']) if r['n_test_no'] is not None else "N/A"
    nex = str(r['n_test_ext']) if r['n_test_ext'] is not None else "N/A"
    print(f"{r['combo']:<20} {obj:<12} {base:<10} {ext:<10} {cno:<10} {cex:<10} {nno:<10} {nex:<10}")

print("\nDone!")
