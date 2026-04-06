#!/usr/bin/env python3
"""Continue WIDS ablation — only run optuna extended (base+combined from previous run)."""
import sys
import os
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from core.GenericDataPipeline import GenericDataPipeline
from core.RunData import RunPipeline
from core.XGBoostModel import XGBoostModel
from sklearn.metrics import roc_auc_score
from core.seed_utils import SEED, set_all_seeds

pd.set_option('future.infer_string', False)
set_all_seeds()

ABLATION_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'ablation', 'WIDS')

# Load data
pipeline = GenericDataPipeline()
csv_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'WIDS.csv')
df = pd.read_csv(csv_path, na_values=['NA'])
df = df.drop(columns=['encounter_id', 'patient_id', 'hospital_id'])
df = pipeline.preprocessing(df)
label = "hospital_death"
df[label] = df[label].astype(int)

ext_features = [
    'h1_bilirubin_max', 'h1_bilirubin_min', 'h1_albumin_max', 'h1_albumin_min',
    'h1_lactate_max', 'h1_lactate_min',
    'h1_pao2fio2ratio_max', 'h1_pao2fio2ratio_min',
    'h1_arterial_ph_max', 'h1_arterial_ph_min', 'h1_arterial_pco2_max',
    'h1_arterial_pco2_min', 'h1_arterial_po2_max', 'h1_arterial_po2_min',
]

all_features_list = [c for c in df.columns if c != label]
base_features = [f for f in all_features_list if f not in ext_features]

# Setup pipeline (same split as before)
dm = RunPipeline()
dm.load_data(base_features, ext_features, df.copy(), label)
dm.set_has_extended()
dm.train_test_split()
dm.set_train_base_ext_datasets()

ext_train = dm.ext_df[dm.ext_df['has_extended'] == 1].copy()
test_with = dm.test_df[dm.test_df['has_extended'] == 1]
test_without = dm.test_df[dm.test_df['has_extended'] == 0]
print(f"Test: {len(test_without)} no-ext, {len(test_with)} with-ext")

# Train base model (needed for warm-starting)
print("\n=== Training Base Model ===", flush=True)
dm.base_model = XGBoostModel(name="base_model")
dm.base_model.train(
    X=dm.base_df[dm.base_features + dm.ext_features],
    y=dm.base_df[dm.label],
    n_trials=15
)
dm.base_model.save_model()

base_auc = roc_auc_score(test_without[label],
                          dm.base_model.predict(test_without[dm.all_features]))
print(f"Base AUC: {base_auc:.6f}")

# Use known combined AUCs from previous successful run
comb_no_auc = 0.899059
comb_ext_auc = 0.894995
print(f"Combined (no-ext): {comb_no_auc:.6f}, Combined (with-ext): {comb_ext_auc:.6f} [from prev run]")

# Run only optuna extended
print(f"\n{'='*80}")
print(f"WIDS | optuna (n_trials=30, with mandatory no_pruning trial)")
print(f"{'='*80}", flush=True)

dm.extended_model = XGBoostModel(name="extended_optuna")
dm.extended_model.train(
    ext_train[dm.base_features + dm.ext_features],
    ext_train[dm.label],
    base_model_path="base_model.json",
    n_trials=30,
    pruning_mode='optuna'
)

ext_auc = roc_auc_score(test_with[label],
                         dm.extended_model.predict(test_with[dm.all_features]))
n_total = len(test_with) + len(test_without)
objective = (len(test_without) * (comb_no_auc - base_auc) + len(test_with) * (comb_ext_auc - ext_auc)) / n_total

print(f"\n>>> WIDS | optuna: objective={objective:.6f}")
print(f"    Ext AUC: {ext_auc:.6f}")
print(f"    Base AUC: {base_auc:.6f}, Comb-no: {comb_no_auc:.6f}, Comb+ext: {comb_ext_auc:.6f}")
print("\nDone!")
