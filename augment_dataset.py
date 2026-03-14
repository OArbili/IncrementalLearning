#!/usr/bin/env python3
"""Augment BankLoanSta dataset with artificial nulls in Home Ownership & Purpose."""
import pandas as pd
import numpy as np
import kagglehub
import os
from GenericDataPipeline import GenericDataPipeline
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

# --- BEFORE augmentation ---
print("\n=== BEFORE Augmentation ===")
print("Null ratios:")
for col in df.columns:
    null_ratio = df[col].isna().mean()
    if null_ratio > 0:
        print(f"  {col}: {null_ratio:.4f} ({df[col].isna().sum()} rows)")

null_features = ['Credit Score', 'Annual Income', 'Months since last delinquent', 'Years in current job']
print(f"\nHome Ownership nulls: {df['Home Ownership'].isna().sum()}")
print(f"Purpose nulls: {df['Purpose'].isna().sum()}")

# --- Augmentation ---
rng = np.random.RandomState(SEED)

# Step 1: Rows that already have at least one null in the 4 natural null features
has_any_null = df[null_features].isna().any(axis=1)
candidate_indices = df[has_any_null].index.tolist()
print(f"\nCandidate rows (have at least 1 null in {null_features}): {len(candidate_indices)}")

# Step 2: Sample up to 20% of total rows
max_sample = int(0.20 * len(df))  # 20,000
n_sample = min(max_sample, len(candidate_indices))
sampled_indices = rng.choice(candidate_indices, size=n_sample, replace=False)
print(f"Sampled rows: {len(sampled_indices)} (max 20% = {max_sample})")

# Step 3: For each sampled row, randomly null Home Ownership, Purpose, or both
choices = rng.choice(['home', 'purpose', 'both'], size=len(sampled_indices))

home_null_idx = sampled_indices[choices != 'purpose']  # 'home' or 'both'
purpose_null_idx = sampled_indices[choices != 'home']   # 'purpose' or 'both'

df.loc[home_null_idx, 'Home Ownership'] = np.nan
df.loc[purpose_null_idx, 'Purpose'] = np.nan

# --- AFTER augmentation ---
print("\n=== AFTER Augmentation ===")
print("Null ratios (all features with nulls):")
for col in df.columns:
    null_ratio = df[col].isna().mean()
    if null_ratio > 0:
        print(f"  {col}: {null_ratio:.4f} ({df[col].isna().sum()} rows)")

print(f"\nHome Ownership nulls introduced: {df.loc[home_null_idx, 'Home Ownership'].isna().sum()}")
print(f"Purpose nulls introduced: {df.loc[purpose_null_idx, 'Purpose'].isna().sum()}")

# Show overlap with existing null features
print("\n=== Null Co-occurrence ===")
extended_null_features = null_features + ['Home Ownership', 'Purpose']
for feat in extended_null_features:
    null_mask = df[feat].isna()
    print(f"\n{feat} (null={null_mask.sum()}):")
    for other in extended_null_features:
        if other != feat:
            other_null = df[other].isna()
            both_null = (null_mask & other_null).sum()
            print(f"  also null in {other}: {both_null}")

# Features that would pass null_ratio > 0.01
print("\n=== Features eligible for extended (null_ratio > 0.01) ===")
for col in df.columns:
    null_ratio = df[col].isna().mean()
    if null_ratio > 0.01:
        print(f"  {col}: {null_ratio:.2%}")
