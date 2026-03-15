#!/usr/bin/env python3
"""Run all valid combinations on Movie dataset with augmented nulls."""
import itertools
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
import kagglehub
from core.GenericDataPipeline import GenericDataPipeline
from core.RunData import RunPipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from core.seed_utils import SEED, set_all_seeds

pd.set_option('future.infer_string', False)
set_all_seeds()

# --- Config ---
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
NULL_THRESHOLD = 0.05
MIN_GROUP_PCT = 0.02

pipeline = GenericDataPipeline()

# --- Load data ---
print("Downloading MovieLens 20M dataset...")
path = kagglehub.dataset_download("grouplens/movielens-20m-dataset")

ratings = pd.read_csv(os.path.join(path, "rating.csv"))
tags = pd.read_csv(os.path.join(path, "tag.csv"))
print(f"Ratings: {ratings.shape}, Tags: {tags.shape}")

# Convert datetime strings to unix timestamps
print("Converting timestamps...")
ratings['timestamp'] = pd.to_datetime(ratings['timestamp']).astype(np.int64) // 10**9
tags['timestamp'] = pd.to_datetime(tags['timestamp']).astype(np.int64) // 10**9

# --- Feature engineering ---
print("Engineering features...")
cutoff_date = ratings['timestamp'].quantile(0.8)
target_date = ratings['timestamp'].quantile(0.9)

user_stats = ratings[ratings['timestamp'] < cutoff_date].groupby('userId').agg({
    'rating': ['count', 'mean', 'std'],
    'timestamp': ['min', 'max']
}).round(3)
user_stats.columns = ['rating_count', 'rating_mean', 'rating_std', 'first_rating', 'last_rating']
df = user_stats.reset_index()
df['days_active'] = (df['last_rating'] - df['first_rating']) / (24 * 60 * 60)
df['rating_frequency'] = df['rating_count'] / df['days_active'].clip(lower=1)

future_activity = ratings[
    (ratings['timestamp'] >= cutoff_date) &
    (ratings['timestamp'] < target_date)
].groupby('userId')['rating'].count().reset_index()
future_activity.columns = ['userId', 'future_ratings']
future_activity['TARGET'] = (future_activity['future_ratings'] >
                              future_activity['future_ratings'].median()).astype(int)
df = df.merge(future_activity[['userId', 'TARGET']], on='userId', how='left')
df['TARGET'] = df['TARGET'].fillna(0)

tag_activity = tags[tags['timestamp'] < cutoff_date].groupby('userId').agg({
    'tag': ['count', 'nunique'],
    'timestamp': ['min', 'max']
})
tag_activity.columns = ['tag_count', 'unique_tags', 'first_tag', 'last_tag']
tag_activity = tag_activity.reset_index()
tag_activity['days_tagging'] = (tag_activity['last_tag'] - tag_activity['first_tag']) / (24 * 60 * 60)
tag_activity['tag_frequency'] = tag_activity['tag_count'] / tag_activity['days_tagging'].clip(lower=1)

tag_lengths = tags[tags['timestamp'] < cutoff_date].groupby('userId')['tag'].apply(
    lambda x: np.mean([len(str(t)) for t in x])
).reset_index()
tag_lengths.columns = ['userId', 'avg_tag_length']
tag_activity = tag_activity.merge(tag_lengths, on='userId', how='left')
df = df.merge(tag_activity, on='userId', how='left')

columns_to_keep = [
    'rating_count', 'rating_mean', 'rating_std',
    'days_active', 'rating_frequency',
    'tag_count', 'unique_tags', 'avg_tag_length',
    'tag_frequency', 'last_tag', 'TARGET'
]
df = df[columns_to_keep]
df = pipeline.preprocessing(df)

label = "TARGET"
df[label] = df[label].astype(int)

print(f"Dataset shape (before augmentation): {df.shape}")

# --- Augmentation: inject nulls into rating_count, rating_mean, days_active ---
# These base features have 0% nulls. We'll null them in ~20% of rows randomly.
rng = np.random.RandomState(SEED)

n_sample = int(0.20 * len(df))
sampled_indices = rng.choice(df.index, size=n_sample, replace=False)

augment_features = ['rating_count', 'rating_mean', 'days_active']
choices = rng.choice(['count', 'mean', 'active', 'count+mean', 'mean+active', 'all'],
                     size=len(sampled_indices))

count_idx = sampled_indices[np.isin(choices, ['count', 'count+mean', 'all'])]
mean_idx = sampled_indices[np.isin(choices, ['mean', 'count+mean', 'mean+active', 'all'])]
active_idx = sampled_indices[np.isin(choices, ['active', 'mean+active', 'all'])]

df.loc[count_idx, 'rating_count'] = np.nan
df.loc[mean_idx, 'rating_mean'] = np.nan
df.loc[active_idx, 'days_active'] = np.nan

print("\n=== After Augmentation ===")
print("Null ratios (features with nulls):")
for col in df.columns:
    nr = df[col].isna().mean()
    if nr > 0.01:
        print(f"  {col}: {nr:.2%}")

print(f"\nN_TRIALS per model: {N_TRIALS}")
print(f"Target distribution:\n{df[label].value_counts()}")

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

    if len(ext_feats) == 0:
        continue
    if len(base_feats_check) == 0:
        continue

    has_extended = df[ext_feats].notnull().any(axis=1).astype(int)
    n_with = has_extended.sum()
    n_without = len(df) - n_with

    if has_extended.nunique() < 2:
        continue

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

    ext_names = '+'.join([f.replace(' ', '').replace('_', '')[:6] for f in ext_feats])
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

# --- Deduplicate ---
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
                                 f"movie_aug_combo_{i+1}.csv", n_trials=N_TRIALS)

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

neg = [r for r in results if isinstance(r['objective'], float) and r['objective'] < 0]
total_valid = [r for r in results if r['objective'] not in [999, 99999]]
print(f"\nNegative objectives (incremental wins): {len(neg)}/{len(total_valid)}")
print("\nDone!")
