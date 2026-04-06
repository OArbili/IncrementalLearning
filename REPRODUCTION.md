# Reproduction Guide

## Prerequisites

```bash
git clone https://github.com/OArbili/IncrementalLearning.git
cd IncrementalLearning
pip install xgboost optuna pandas numpy scikit-learn scipy kagglehub
```

Kaggle API credentials are needed for datasets downloaded via `kagglehub`. Set up:
```bash
pip install kagglehub
# Follow https://www.kaggle.com/docs/api to configure ~/.kaggle/kaggle.json
```

## Dataset Types

### Natural Missingness (no augmentation needed)
These datasets have real-world structured missingness. The scripts download and preprocess them automatically:

| Dataset | Script | Data Source |
|---|---|---|
| Weather | `scripts/run_ablation_pruning.py` (filter: Weather) | `kagglehub: rever3nd/weather-data` |
| WeatherAUS | `scripts/run_ablation_pruning.py` (filter: WeatherAUS) | Local: `datasets/weatherAUS.csv` |
| DiabetesRecord | `scripts/run_ablation_pruning.py` (filter: DiabetesRecord) | `kagglehub: brandao/diabetes` |
| HRAnalytics | `scripts/run_ablation_pruning.py` (filter: HRAnalytics) | `kagglehub: arashnic/hr-analytics-job-change-of-data-scientists` |
| WIDS | `scripts/run_ablation_pruning.py` (filter: WIDS) | Local: `datasets/WIDS.csv` |
| CreditRisk | `scripts/run_ablation_pruning.py` (filter: CreditRisk) | Local: `datasets/CreditRisk/data_devsample.csv` + `data_to_score.csv` |

### Augmented Missingness (automatic, deterministic)
These datasets inject nulls into clean features to simulate evolving feature scenarios. **The augmentation is built into each script** using `np.random.RandomState(42)`, so it is fully reproducible — no manual adjustment needed.

| Dataset | Script | Augmentation | Null Rate |
|---|---|---|---|
| BankLoanSta | `scripts/run_ablation_pruning.py` (filter: BankLoanSta) | Inject nulls into Home Ownership + Purpose from rows already having nulls | ~13-20% |
| ClientRecord | `scripts/run_ablation_pruning.py` (filter: ClientRecordV2) | Inject 40% independent nulls into Contract, Tenure in Months, Monthly Charge | 40% each |
| MovieAugV2 | `scripts/run_ablation_pruning.py` (filter: MovieAugV2) | Inject 50% independent nulls into rating_count, rating_mean, rating_std, days_active, rating_frequency | 50% each |
| FlightDelay | `scripts/run_ablation_pruning.py` (filter: FlightDelay) | Inject 20% independent nulls into TAIL_NUM, DISTANCE, OP_CARRIER_FL_NUM, DAY_OF_MONTH | 20% each |

**How augmentation works:** Each script loads the raw dataset from Kaggle, then applies null injection using a fixed random seed (42). The injection selects random row indices per feature independently. Since the seed is fixed, the exact same rows get nullified on every run. No manual configuration is needed — just run the script.

## Running Experiments

### Full combo sweep (find best feature combination)
```bash
# Example: run BankLoanSta with 30 Optuna trials
python scripts/run_bankloansta.py 30

# Example: run CreditRisk with 30 trials
python scripts/run_credit_risk.py 30

# Example: run DiabetesRecord with 30 trials (skip first 3 combos)
python scripts/run_diabetes.py 30 3
```

### Ablation study (compare pruning modes on best combo)
```bash
# Run ablation for a specific dataset (30 trials)
python scripts/run_ablation_pruning.py 30 BankLoanSta
python scripts/run_ablation_pruning.py 30 Weather
python scripts/run_ablation_pruning.py 30 CreditRisk

# Run all datasets
python scripts/run_ablation_pruning.py 30
```

The ablation script trains:
- **Base model**: 30 trials (shared across modes)
- **Combined model**: 30 trials (shared across modes)
- **Extended model**: 30 trials (Optuna), 15 trials (no_pruning), 15 trials (fixed_50)

### What each script does
1. Downloads/loads raw data
2. Applies augmentation (if applicable) with seed=42
3. Identifies features with >N% nulls as extended candidates
4. Groups features by null pattern similarity (Jaccard > 0.95)
5. Enumerates valid feature combinations
6. For each combo: trains base → extended → combined models
7. Reports AUC comparison (our method vs baseline)

## Local Datasets

These datasets must be placed manually (not on Kaggle):
```
datasets/
├── weatherAUS.csv           # WeatherAUS dataset
├── WIDS.csv                 # WiDS Datathon 2020
└── CreditRisk/
    ├── data_devsample.csv   # Credit risk scoring
    └── data_to_score.csv
```

## Seed and Reproducibility

All experiments use `SEED = 42` from `core/seed_utils.py`. This seeds:
- Python's `random` module
- NumPy's random state
- XGBoost's `random_state` parameter
- Optuna's TPE sampler
- Train/test split stratification

The augmented null injection uses `np.random.RandomState(42)` specifically, ensuring deterministic row selection across runs and machines.

## Expected Runtime

| Dataset | Combos | ~Time (30 trials) |
|---|---|---|
| BankLoanSta | 7-12 | 2-4 hours |
| Weather | 23 | 3-5 hours |
| DiabetesRecord | 12 | 2-3 hours |
| HRAnalytics | 5 | 30 min |
| ClientRecord | 7 | 30 min |
| MovieAugV2 | 20 | 4-6 hours |
| WeatherAUS | 7-15 | 2-4 hours |
| WIDS | 21 | 4-6 hours |
| FlightDelay | 3 | 1-2 hours |
| CreditRisk | 10 | 2-3 hours |
