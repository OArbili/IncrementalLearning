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

## Notes

- Seed is fixed at 42 for reproducibility (`core/seed_utils.py`)
- XGBoost uses `tree_method='hist'` and auto-detects GPU via `device` setting
- Each experiment enumerates all valid feature combinations and deduplicates by null pattern
