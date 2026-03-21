# Running Experiments on a Different Machine

## 1. Clone the repo

```bash
git clone https://github.com/OArbili/IncrementalLearning.git
cd IncrementalLearning
```

## 2. Create conda environment

```bash
conda create -n bgu python=3.11 -y
conda activate bgu
pip install -r requirements.txt
```

If `requirements.txt` doesn't exist, install manually:

```bash
pip install pandas numpy scikit-learn xgboost optuna kagglehub
```

## 3. Run all experiments

```bash
# All datasets with 30 Optuna trials per model
python scripts/run_all_experiments.py 30

# Specific datasets only
python scripts/run_all_experiments.py 30 Weather,WIDS

# Quick test with 5 trials
python scripts/run_all_experiments.py 5 Weather
```

### What it runs

| # | Dataset | Script | Type |
|---|---------|--------|------|
| 1 | BankLoanSta | run_augmented_combos.py | Augmented nulls |
| 2 | Weather | run_weather.py | Natural nulls |
| 3 | DiabetesRecord | run_diabetes.py | Natural nulls |
| 4 | HRAnalytics | run_hr_analytics.py | Natural nulls |
| 5 | ClientRecordAug | run_client_record_augmented.py | Augmented nulls |
| 6 | MovieAugV2 | run_movie_augmented_v2.py | Augmented nulls |
| 7 | WeatherAUS | run_weatheraus.py | Natural nulls |
| 8 | WIDS | run_wids.py | Natural nulls |
| 9 | FlightDelay | run_airline.py | Augmented nulls |

### Output

- Per-dataset results: `results/<DatasetName>/`
- Summary: `results/summary_t<N>.txt`
- Live progress: printed to stdout with per-trial AUC logging

## 4. Run ablation study (pruning mode comparison)

```bash
# N_TRIALS controls optuna mode; no_pruning and fixed_50 get 60% of N_TRIALS
python scripts/run_ablation_pruning.py 50 Weather
python scripts/run_ablation_pruning.py 30            # all datasets
```

Output: `results/ablation/`

## 5. Run individual datasets

Each script accepts N_TRIALS as first argument:

```bash
python scripts/run_weather.py 30
python scripts/run_wids.py 10
python scripts/run_augmented_combos.py 30   # BankLoanSta
```

## 6. Local datasets

Some datasets are downloaded automatically via `kagglehub`. Others require local CSV files:

- `datasets/weatherAUS.csv` — WeatherAUS dataset
- `datasets/WIDS.csv` — WIDS ICU dataset
- `datasets/CreditRisk/data_devsample.csv` + `data_to_score.csv` — Credit Risk

Place these files before running the corresponding experiments.

## 7. Run from a Jupyter Notebook (on a different machine)

### Cell 1: Clone repo and install dependencies

```python
!git clone https://github.com/OArbili/IncrementalLearning.git
%cd IncrementalLearning
!pip install pandas numpy scikit-learn xgboost optuna kagglehub
```

### Cell 2: Setup imports

```python
import sys, os
sys.path.insert(0, os.getcwd())

import pandas as pd
import numpy as np
from core.GenericDataPipeline import GenericDataPipeline
from core.RunData import RunPipeline
from core.seed_utils import SEED, set_all_seeds

pd.set_option('future.infer_string', False)
set_all_seeds()

pipeline = GenericDataPipeline()
```

### Option A: Run on your own dataset

```python
# 1. Load and preprocess
df = pd.read_csv('your_data.csv')
df = pipeline.preprocessing(df)

label = 'target_column'

# 2. Define which features are "extended" (partially available)
ext_features = ['feature_A', 'feature_B']  # features with missing values
base_features = [c for c in df.columns if c != label and c not in ext_features]

# 3. Run the full pipeline (trains base, extended, combined + evaluates)
dm = RunPipeline()
objective = dm.full_run(
    df.copy(),
    base_features,
    ext_features,
    label,
    csv_name='my_experiment.csv',  # saves results CSV
    n_trials=30                     # Optuna trials per model
)

print(f'Objective: {objective:.6f}')
# Negative = incremental learning wins
# Positive = combined (retrain from scratch) wins
```

### Option B: Run step-by-step for more control

```python
# 1. Load and preprocess
df = pd.read_csv('your_data.csv')
df = pipeline.preprocessing(df)

label = 'target_column'
ext_features = ['feature_A', 'feature_B']
base_features = [c for c in df.columns if c != label and c not in ext_features]

# 2. Setup pipeline
dm = RunPipeline()
dm.load_data(base_features, ext_features, df.copy(), label)
dm.set_has_extended()
dm.train_test_split()
dm.set_train_base_ext_datasets()

# 3. Train models individually
dm.train_all(n_trials=30, pruning_mode='optuna')

# 4. Evaluate
objective = dm.test_all(csv_name='my_experiment.csv')
print(f'Objective: {objective:.6f}')

# 5. Access individual models for inspection
print(dm.base_model.best_params)
print(dm.extended_model.best_params)
print(dm.combined_model.best_params)
```

### Option C: Run an existing dataset experiment

```python
# Example: Weather dataset
import kagglehub

path = kagglehub.dataset_download("rever3nd/weather-data")
csv_path = os.path.join(path, os.listdir(path)[0])
df = pd.read_csv(csv_path)

columns_to_drop = ['Unnamed: 0', 'Date', 'RISK_MM',
    'Humidity3pm', 'Pressure3pm', 'Cloud3pm', 'Temp3pm',
    'WindDir3pm', 'WindSpeed3pm']
df = df.drop(columns=columns_to_drop)
df = pipeline.preprocessing(df)

label = 'RainTomorrow'
ext_features = ['Evaporation', 'Sunshine', 'WindDir9am']
base_features = [c for c in df.columns if c != label and c not in ext_features]

dm = RunPipeline()
objective = dm.full_run(df.copy(), base_features, ext_features, label,
                        'weather_notebook.csv', n_trials=10)
print(f'Objective: {objective:.6f}')
```

## Notes

- Seed is fixed at 42 for reproducibility (`core/seed_utils.py`)
- XGBoost uses `tree_method='hist'` and auto-detects GPU via `device` setting
- Each experiment enumerates all valid feature combinations and deduplicates by null pattern
