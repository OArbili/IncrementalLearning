#!/usr/bin/env python3
"""Run BankLoanSta pipeline."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
import optuna
import kagglehub
from core.GenericDataPipeline import GenericDataPipeline
from core.seed_utils import SEED, set_all_seeds

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
print(f"Target: {label}")
print(f"Target distribution:\n{df[label].value_counts()}")

X = df.drop(label, axis=1)
y = df[label]

feature_scores = pipeline.rank_features(X, y)
features_with_nulls = feature_scores[feature_scores['null_ratio'] > 0.01]

print(f"\nFeatures with nulls ({len(features_with_nulls)}):")
print(features_with_nulls[['feature_name', 'null_ratio']].to_string(index=False))

n_trials = 20
print("\nRunning Optuna optimization...")

study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=SEED)
)
study.optimize(
    lambda trial: pipeline.objective(trial, features_with_nulls, feature_scores, df, label),
    n_trials=n_trials
)

print(f"\nBest trial:")
print(f"Validation loss + penalty: {study.best_value}")
for key, value in study.best_params.items():
    print(f"    {key}: {value}")
