#!/usr/bin/env python3
"""Ablation study: Pruning mode comparison (optuna vs no_pruning vs fixed_50).

For each dataset's best combo, trains base+combined once, then extended model
3 times with different pruning modes. Saves models + results per dataset.
"""
import sys
import os
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
try:
    import kagglehub
except ImportError:
    kagglehub = None
import shutil
from core.GenericDataPipeline import GenericDataPipeline
from core.RunData import RunPipeline
from core.XGBoostModel import XGBoostModel
from sklearn.metrics import roc_auc_score
from core.seed_utils import SEED, set_all_seeds

pd.set_option('future.infer_string', False)
set_all_seeds()

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
# Per-mode trial overrides (default to N_TRIALS if not specified)
TRIALS_PER_MODE = {
    'optuna': N_TRIALS,
    'no_pruning': 15,
    'fixed_50': 15,
}
# Optional: filter to specific datasets (comma-separated), e.g. "WeatherAUS,WIDS"
DATASET_FILTER = sys.argv[2].split(',') if len(sys.argv) > 2 else None
ABLATION_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'ablation')
PRUNING_MODES = ['optuna', 'no_pruning', 'fixed_50']

pipeline = GenericDataPipeline()

# ============================================================================
# Dataset loading functions — each returns (df, label, ext_features)
# ============================================================================

def load_bankloansta():
    """BankLoanSta (Augmented — structured nulls). Best combo: TBD from sweep."""
    path = kagglehub.dataset_download("zaurbegiev/my-dataset")
    csv_files = sorted([f for f in os.listdir(path) if f.endswith('.csv')])
    csv_path = os.path.join(path, csv_files[1])
    df = pd.read_csv(csv_path)
    df.dropna(subset=['Loan Status'], inplace=True)
    df.drop(['Loan ID', 'Customer ID'], axis=1, inplace=True)
    df = pipeline.preprocessing(df)
    label = "Loan Status"
    df[label] = df[label].astype(int)

    # Augmentation: inject STRUCTURED nulls into 3 features (conditional on base features)
    rng = np.random.RandomState(SEED)
    # Current Loan Amount -> NaN for high debt (Monthly Debt > median) + noise
    debt_median = df['Monthly Debt'].median()
    high_debt = df[df['Monthly Debt'] > debt_median].index
    low_debt = df[df['Monthly Debt'] <= debt_median].index
    noise1 = rng.choice(low_debt, size=int(0.10 * len(low_debt)), replace=False)
    df.loc[np.concatenate([high_debt, noise1]), 'Current Loan Amount'] = np.nan
    # Annual Income -> NaN for short credit history + noise
    hist_median = df['Years of Credit History'].median()
    short_hist = df[df['Years of Credit History'] < hist_median].index
    long_hist = df[df['Years of Credit History'] >= hist_median].index
    noise2 = rng.choice(long_hist, size=int(0.10 * len(long_hist)), replace=False)
    df.loc[np.concatenate([short_hist, noise2]), 'Annual Income'] = np.nan
    # Credit Score -> NaN for many open accounts + noise
    acct_median = df['Number of Open Accounts'].median()
    many_accts = df[df['Number of Open Accounts'] > acct_median].index
    few_accts = df[df['Number of Open Accounts'] <= acct_median].index
    noise3 = rng.choice(few_accts, size=int(0.10 * len(few_accts)), replace=False)
    df.loc[np.concatenate([many_accts, noise3]), 'Credit Score'] = np.nan

    ext_features = ['Current Loan Amount', 'Annual Income', 'Credit Score']
    return df, label, ext_features


def load_weather():
    """Weather (Natural Nulls). Best combo: Evaporation ext."""
    path = kagglehub.dataset_download("rever3nd/weather-data")
    csv_files = sorted([f for f in os.listdir(path) if f.endswith('.csv')])
    csv_path = os.path.join(path, csv_files[0])
    df = pd.read_csv(csv_path)
    drop_cols = ['Unnamed: 0', 'Date', 'RISK_MM', 'Humidity3pm', 'Pressure3pm',
                 'Cloud3pm', 'Temp3pm', 'WindDir3pm', 'WindSpeed3pm']
    df.drop([c for c in drop_cols if c in df.columns], axis=1, inplace=True)
    df = pipeline.preprocessing(df)
    label = "RainTomorrow"
    df[label] = df[label].astype(int)

    ext_features = ['Evaporation']
    return df, label, ext_features


def load_diabetes():
    """DiabetesRecord (Natural Nulls). Best combo: medical_specialty + A1Cresult ext."""
    path = kagglehub.dataset_download("brandao/diabetes")
    df = pd.read_csv(os.path.join(path, "diabetic_data.csv"))
    df['readmitted'] = df['readmitted'].replace({'NO': 0, '<30': 1, '>30': 1}).astype(int)
    df.drop(['encounter_id', 'patient_nbr', 'number_inpatient', 'number_emergency', 'discharge_disposition_id'], axis=1, inplace=True)
    df = pipeline.preprocessing(df)
    label = "readmitted"
    df[label] = df[label].astype(int)

    ext_features = ['medical_specialty', 'A1Cresult']
    return df, label, ext_features


def load_hr_analytics():
    """HRAnalytics (Natural Nulls). Best combo: company_size ext."""
    path = kagglehub.dataset_download("arashnic/hr-analytics-job-change-of-data-scientists")
    df = pd.read_csv(os.path.join(path, "aug_train.csv"))
    df.drop(['enrollee_id'], axis=1, inplace=True)
    df = pipeline.preprocessing(df)
    label = "target"
    df[label] = df[label].astype(int)

    ext_features = ['company_size']
    return df, label, ext_features


def load_client_record_aug():
    """ClientRecord (Augmented). Best combo: Offer ext."""
    path = kagglehub.dataset_download("shilongzhuang/telecom-customer-churn-by-maven-analytics")
    csv_files = sorted([f for f in os.listdir(path) if f.endswith('.csv')])
    data1 = pd.read_csv(os.path.join(path, csv_files[2]))
    data2 = pd.read_csv(os.path.join(path, csv_files[0]))
    df = pd.merge(data2, data1, on='Zip Code')
    df['Customer Status'] = df['Customer Status'].apply(lambda x: 1 if x == 'Stayed' else 0)
    drop_cols = ['Customer ID', 'Churn Category', 'Churn Reason', 'Total Charges',
                 'Total Revenue', 'Total Refunds', 'Total Long Distance Charges',
                 'Zip Code', 'City', 'Latitude', 'Longitude']
    df.drop([c for c in drop_cols if c in df.columns], axis=1, inplace=True)
    df = pipeline.preprocessing(df)
    label = "Customer Status"
    df[label] = df[label].astype(int)

    # Augmentation: inject nulls into Contract, Monthly Charge, Payment Method
    rng = np.random.RandomState(SEED)
    null_features_original = [c for c in df.columns if c != label and df[c].isna().mean() > 0.05]
    null_rows = df[df[null_features_original].isna().any(axis=1)].index
    n_sample = int(0.20 * len(null_rows))
    sampled_idx = rng.choice(null_rows, size=n_sample, replace=False)
    choices = rng.choice(['contract', 'charge', 'payment', 'contract+charge',
                          'charge+payment', 'all'], size=len(sampled_idx))
    contract_idx = sampled_idx[np.isin(choices, ['contract', 'contract+charge', 'all'])]
    charge_idx = sampled_idx[np.isin(choices, ['charge', 'contract+charge', 'charge+payment', 'all'])]
    payment_idx = sampled_idx[np.isin(choices, ['payment', 'charge+payment', 'all'])]
    df.loc[contract_idx, 'Contract'] = np.nan
    df.loc[charge_idx, 'Monthly Charge'] = np.nan
    df.loc[payment_idx, 'Payment Method'] = np.nan

    ext_features = ['Offer']
    return df, label, ext_features


def load_client_record_v2():
    """ClientRecord v2 (Augmented — structured nulls). Best combo: Tenure in Months ext."""
    path = kagglehub.dataset_download("shilongzhuang/telecom-customer-churn-by-maven-analytics")
    csv_files = sorted([f for f in os.listdir(path) if f.endswith('.csv')])
    data1 = pd.read_csv(os.path.join(path, csv_files[2]))
    data2 = pd.read_csv(os.path.join(path, csv_files[0]))
    df = pd.merge(data2, data1, on='Zip Code')
    df['Customer Status'] = df['Customer Status'].apply(lambda x: 1 if x == 'Stayed' else 0)
    drop_cols = ['Customer ID', 'Churn Category', 'Churn Reason', 'Total Charges',
                 'Total Revenue', 'Total Refunds', 'Total Long Distance Charges',
                 'Zip Code', 'City', 'Latitude', 'Longitude']
    df.drop([c for c in drop_cols if c in df.columns], axis=1, inplace=True)
    df = pipeline.preprocessing(df)
    label = "Customer Status"
    df[label] = df[label].astype(int)

    # Augmentation: inject STRUCTURED nulls (conditional on base features)
    rng = np.random.RandomState(SEED)
    # Contract -> NaN for short-tenure customers + noise
    tenure_median = df['Tenure in Months'].median()
    contract_candidates = df[df['Tenure in Months'] < tenure_median].index
    above_median = df[df['Tenure in Months'] >= tenure_median].index
    noise_idx = rng.choice(above_median, size=int(0.10 * len(above_median)), replace=False)
    df.loc[np.concatenate([contract_candidates, noise_idx]), 'Contract'] = np.nan
    # Tenure -> NaN for no-referral customers + noise
    no_ref = df[df['Number of Referrals'] == 0].index
    has_ref = df[df['Number of Referrals'] > 0].index
    noise_idx2 = rng.choice(has_ref, size=int(0.15 * len(has_ref)), replace=False)
    df.loc[np.concatenate([no_ref, noise_idx2]), 'Tenure in Months'] = np.nan
    # Monthly Charge -> NaN for non-paperless customers + noise
    non_paper = df[df['Paperless Billing'] == 0].index
    paper = df[df['Paperless Billing'] == 1].index
    noise_idx3 = rng.choice(paper, size=int(0.10 * len(paper)), replace=False)
    df.loc[np.concatenate([non_paper, noise_idx3]), 'Monthly Charge'] = np.nan

    ext_features = ['Tenure in Months']
    return df, label, ext_features


def load_movie_aug_v2():
    """Movie (Augmented v2 — Heavy). Best combo: rating_mean + tags ext."""
    path = kagglehub.dataset_download("grouplens/movielens-20m-dataset")
    ratings = pd.read_csv(os.path.join(path, "rating.csv"))
    tags = pd.read_csv(os.path.join(path, "tag.csv"))
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp']).astype(np.int64) // 10**9
    tags['timestamp'] = pd.to_datetime(tags['timestamp']).astype(np.int64) // 10**9

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
        (ratings['timestamp'] >= cutoff_date) & (ratings['timestamp'] < target_date)
    ].groupby('userId')['rating'].count().reset_index()
    future_activity.columns = ['userId', 'future_ratings']
    future_activity['TARGET'] = (future_activity['future_ratings'] >
                                  future_activity['future_ratings'].median()).astype(int)
    df = df.merge(future_activity[['userId', 'TARGET']], on='userId', how='left')
    df['TARGET'] = df['TARGET'].fillna(0)

    tag_activity = tags[tags['timestamp'] < cutoff_date].groupby('userId').agg({
        'tag': ['count', 'nunique'], 'timestamp': ['min', 'max']
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

    columns_to_keep = ['rating_count', 'rating_mean', 'rating_std', 'days_active',
                       'rating_frequency', 'tag_count', 'unique_tags', 'avg_tag_length',
                       'tag_frequency', 'last_tag', 'TARGET']
    df = df[columns_to_keep]
    df = pipeline.preprocessing(df)
    label = "TARGET"
    df[label] = df[label].astype(int)

    # Heavy augmentation: 50% null per feature, independent
    rng = np.random.RandomState(SEED)
    augment_features = ['rating_count', 'rating_mean', 'rating_std', 'days_active', 'rating_frequency']
    n_null = int(0.50 * len(df))
    for feat in augment_features:
        null_idx = rng.choice(df.index, size=n_null, replace=False)
        df.loc[null_idx, feat] = np.nan

    ext_features = ['rating_mean', 'tag_count', 'unique_tags', 'avg_tag_length', 'tag_frequency', 'last_tag']
    return df, label, ext_features


def load_weatheraus():
    """WeatherAUS (Natural Nulls). Best combo: Evaporation + Cloud9am ext."""
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'weatherAUS.csv')
    df = pd.read_csv(csv_path, na_values=['NA'])
    columns_to_drop = ['Date', 'Location', 'RISK_MM']
    df = df.drop(columns=columns_to_drop)
    df = pipeline.preprocessing(df)
    label = "RainTomorrow"
    df[label] = df[label].astype(int)

    ext_features = ['Evaporation', 'Cloud9am']
    return df, label, ext_features


def load_wids():
    """WIDS (Natural Nulls). Best combo: h1_lactate + d1_lactate + d1_pao2fio2ratio (5 features)."""
    import xgboost as xgb
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'WIDS.csv')
    df = pd.read_csv(csv_path, na_values=['NA'])
    columns_to_drop = ['encounter_id', 'patient_id', 'hospital_id']
    df = df.drop(columns=columns_to_drop)
    df = pipeline.preprocessing(df)
    label = "hospital_death"
    df[label] = df[label].astype(int)

    # Feature selection: top 50 by importance (same as run_wids.py)
    X_all = df.drop(label, axis=1)
    selector = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                  tree_method='hist', random_state=SEED, eval_metric='auc')
    selector.fit(X_all, df[label], verbose=False)
    imp = pd.Series(selector.feature_importances_, index=X_all.columns).sort_values(ascending=False)
    keep = set(imp.head(50).index.tolist()) | {label}
    df = df[[c for c in df.columns if c in keep]]

    # Best combo from weighted sweep: h1_lactate + d1_lactate + d1_pao2fio2ratio
    ext_features = [
        'h1_lactate_min',
        'd1_lactate_max', 'd1_lactate_min',
        'd1_pao2fio2ratio_max', 'd1_pao2fio2ratio_min',
    ]
    return df, label, ext_features


def load_flight_delay():
    """FlightDelay (Augmented). Best combo: OP_CARRIER_FL_NUM ext."""
    path = kagglehub.dataset_download("divyansh22/flight-delay-prediction")
    df = pd.read_csv(os.path.join(path, "Jan_2019_ontime.csv"))

    # Drop leaky/useless columns
    drop_cols = ['Unnamed: 21', 'CANCELLED', 'DIVERTED', 'DEP_TIME', 'ARR_TIME',
                 'ARR_DEL15', 'ORIGIN_AIRPORT_SEQ_ID', 'DEST_AIRPORT_SEQ_ID',
                 'OP_CARRIER_AIRLINE_ID']
    df.drop([c for c in drop_cols if c in df.columns], axis=1, inplace=True)

    df.dropna(subset=['DEP_DEL15'], inplace=True)
    label = "DEP_DEL15"
    df[label] = df[label].astype(int)
    # Sample to 100K rows to avoid OOM
    if len(df) > 100000:
        df = df.sample(n=100000, random_state=SEED).reset_index(drop=True)
    df = pipeline.preprocessing(df)

    # Augmentation: inject 20% nulls into selected features
    rng = np.random.RandomState(SEED)
    augment_features = ['TAIL_NUM', 'DISTANCE', 'OP_CARRIER_FL_NUM', 'DAY_OF_MONTH']
    null_pct = 0.20
    n_null = int(null_pct * len(df))
    for feat in augment_features:
        null_idx = rng.choice(df.index, size=n_null, replace=False)
        df.loc[null_idx, feat] = np.nan

    # Best combo: OP_CARRIER_FL_NUM as extended feature
    ext_features = ['OP_CARRIER_FL_NUM']
    return df, label, ext_features


def load_credit_risk():
    """CreditRisk (Natural Nulls). Best combo: Groups 1+2 (credit bureau history, 5 features)."""
    import xgboost as xgb
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'CreditRisk')
    data1 = pd.read_csv(os.path.join(data_dir, 'data_devsample.csv'))
    data2 = pd.read_csv(os.path.join(data_dir, 'data_to_score.csv'))
    df = pd.merge(data1, data2, on='SK_ID_CURR', how='inner')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    cols_to_remove = [c for c in df.columns if c.endswith('_y') or c in ['TIME_x','BASE_x','DAY_x','MONTH_x']]
    df.drop(columns=cols_to_remove, errors='ignore', inplace=True)
    df.drop(columns=['SK_ID_CURR'], errors='ignore', inplace=True)

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].replace(['?', ''], np.nan)
            if df[col].isna().sum() < len(df):
                df[col] = pd.Categorical(df[col]).codes
                df[col] = df[col].replace(-1, np.nan)
    for col in df.select_dtypes(include=['bool']).columns:
        df[col] = df[col].astype(int)

    df = pipeline.preprocessing(df)
    label = "TARGET"
    df[label] = df[label].astype(int)

    # Feature selection: top 40 by importance (same as run_credit_risk.py)
    X_all = df.drop(label, axis=1)
    selector = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                  tree_method='hist', random_state=SEED, eval_metric='auc')
    selector.fit(X_all, df[label], verbose=False)
    imp = pd.Series(selector.feature_importances_, index=X_all.columns).sort_values(ascending=False)
    keep = set(imp.head(40).index.tolist()) | {label}
    df = df[[c for c in df.columns if c in keep]]

    # Best combo from sweep: Groups 1+2 (exact features, same as combo 4 in run_credit_risk.py)
    ext_features = [
        'MEAN_AMTCR_1M_3M_TYPE_EQ_ACTIVE_DIV_MEAN_AMTCR_3M_12M_TYPE_EQ_ACTIVE_x',
        'STD_AMTCR_0M_6M_x',
        'MEAN_AMTCR_0M_6M_TYPE_EQ_CLOSED_x',
        'MEAN_AMTCR_0M_6M_TYPE_EQ_ACTIVE_x',
        'MEDIAN_AMTCR_0M_6M_x',
    ]
    return df, label, ext_features


# ============================================================================
# Dataset registry
# ============================================================================

DATASETS = [
    ('BankLoanSta', load_bankloansta),
    ('Weather', load_weather),
    ('DiabetesRecord', load_diabetes),
    ('HRAnalytics', load_hr_analytics),
    ('ClientRecordAug', load_client_record_aug),
    ('MovieAugV2', load_movie_aug_v2),
    ('WeatherAUS', load_weatheraus),
    ('WIDS', load_wids),
    ('FlightDelay', load_flight_delay),
    ('ClientRecordV2', load_client_record_v2),
    ('CreditRisk', load_credit_risk),
]

# ============================================================================
# Main ablation loop
# ============================================================================

all_results = []

for ds_name, load_fn in DATASETS:
    if DATASET_FILTER and ds_name not in DATASET_FILTER:
        print(f"\nSkipping {ds_name} (not in filter: {DATASET_FILTER})")
        continue
    print(f"\n{'#'*100}")
    print(f"# DATASET: {ds_name}")
    print(f"{'#'*100}")
    sys.stdout.flush()

    # Create output directory
    ds_dir = os.path.join(ABLATION_DIR, ds_name)
    os.makedirs(ds_dir, exist_ok=True)

    # Load data
    df, label, ext_features = load_fn()
    all_features_list = [c for c in df.columns if c != label]
    base_features = [f for f in all_features_list if f not in ext_features]

    print(f"Shape: {df.shape}, Label: {label}, Extended: {ext_features}")
    print(f"Target distribution:\n{df[label].value_counts()}")

    # Setup pipeline (data prep + split)
    dm = RunPipeline()
    dm.load_data(base_features, ext_features, df.copy(), label)
    dm.set_has_extended()
    ret = dm.train_test_split()
    if ret == 999:
        print(f"SKIPPING {ds_name}: train_test_split failed")
        continue
    dm.set_train_base_ext_datasets()

    # Save ext_df rows with extended data (before any filtering)
    ext_train = dm.ext_df[dm.ext_df['has_extended'] == 1].copy()

    # Test sets
    test_with = dm.test_df[dm.test_df['has_extended'] == 1]
    test_without = dm.test_df[dm.test_df['has_extended'] == 0]
    print(f"\nTest: {len(test_without)} no-ext, {len(test_with)} with-ext")

    # --- Train Base Model (once) ---
    print("\n=== Training Base Model (shared) ===")
    dm.base_model = XGBoostModel(name="base_model")
    dm.base_model.train(
        X=dm.base_df[dm.base_features + dm.ext_features],
        y=dm.base_df[dm.label],
        n_trials=N_TRIALS
    )
    dm.base_model.save_model()  # saves base_model.json in cwd
    # Copy to ablation folder
    shutil.copy2("base_model.json", os.path.join(ds_dir, "base_model.json"))

    base_auc = roc_auc_score(test_without[label],
                              dm.base_model.predict(test_without[dm.all_features]))
    print(f"\nBase AUC (no-ext test): {base_auc:.6f}")

    # --- Train Combined Model (once) ---
    print("\n=== Training Combined Model (shared) ===")
    dm.combined_model = XGBoostModel(name="combined_model")
    dm.combined_model.train(
        dm.train_df[dm.base_features + dm.ext_features],
        dm.train_df[dm.label],
        n_trials=N_TRIALS
    )
    dm.combined_model.save_model()
    shutil.copy2("combined_model.json", os.path.join(ds_dir, "combined_model.json"))

    comb_no_auc = roc_auc_score(test_without[label],
                                 dm.combined_model.predict(test_without[dm.all_features]))
    comb_ext_auc = roc_auc_score(test_with[label],
                                  dm.combined_model.predict(test_with[dm.all_features]))
    print(f"Combined AUC (no-ext): {comb_no_auc:.6f}, Combined AUC (with-ext): {comb_ext_auc:.6f}")

    # --- Train Extended Model (3 modes) ---
    for mode in PRUNING_MODES:
        mode_trials = TRIALS_PER_MODE.get(mode, N_TRIALS)
        print(f"\n{'='*80}")
        print(f"Training Extended Model — pruning_mode={mode}, n_trials={mode_trials}")
        print(f"{'='*80}")
        sys.stdout.flush()

        dm.extended_model = XGBoostModel(name=f"extended_{mode}")
        dm.extended_model.train(
            ext_train[dm.base_features + dm.ext_features],
            ext_train[dm.label],
            base_model_path="base_model.json",
            n_trials=mode_trials,
            pruning_mode=mode
        )
        # Save model
        dm.extended_model.model.save_model(os.path.join(ds_dir, f"extended_{mode}.json"))

        ext_auc = roc_auc_score(test_with[label],
                                 dm.extended_model.predict(test_with[dm.all_features]))

        n_total = len(test_with) + len(test_without)
        objective = (len(test_without) * (comb_no_auc - base_auc) + len(test_with) * (comb_ext_auc - ext_auc)) / n_total

        print(f"\n>>> {ds_name} | {mode}: objective={objective:.6f}")
        print(f"    Base AUC: {base_auc:.6f}, Ext AUC: {ext_auc:.6f}")
        print(f"    Comb-no: {comb_no_auc:.6f}, Comb+ext: {comb_ext_auc:.6f}")
        sys.stdout.flush()

        all_results.append({
            'dataset': ds_name,
            'mode': mode,
            'objective': objective,
            'base_auc': base_auc,
            'ext_auc': ext_auc,
            'comb_no_auc': comb_no_auc,
            'comb_ext_auc': comb_ext_auc,
            'n_test_no': len(test_without),
            'n_test_ext': len(test_with),
        })

# ============================================================================
# Final summary
# ============================================================================

print(f"\n{'='*120}")
print("ABLATION STUDY: PRUNING MODE COMPARISON")
print(f"{'='*120}")
print(f"{'Dataset':<20} {'Mode':<14} {'Objective':<12} {'Base AUC':<10} {'Ext AUC':<10} {'Comb-no':<10} {'Comb+ext':<10} {'N_no':<8} {'N_ext':<8}")
print("-" * 102)

for r in all_results:
    print(f"{r['dataset']:<20} {r['mode']:<14} {r['objective']:<12.6f} {r['base_auc']:<10.4f} {r['ext_auc']:<10.4f} {r['comb_no_auc']:<10.4f} {r['comb_ext_auc']:<10.4f} {r['n_test_no']:<8} {r['n_test_ext']:<8}")

# Write summary file
summary_path = os.path.join(ABLATION_DIR, "ablation_summary.txt")
with open(summary_path, 'w') as f:
    f.write("=" * 120 + "\n")
    f.write("ABLATION STUDY: PRUNING MODE COMPARISON\n")
    f.write("=" * 120 + "\n\n")
    f.write("Modes:\n")
    f.write("  optuna     — Optuna picks use_base_model (True/False) and n_trees_keep (1..N)\n")
    f.write("  no_pruning — Always warm-start from base, keep ALL trees\n")
    f.write("  fixed_50   — Always warm-start, keep 50% of base trees\n\n")
    f.write(f"{'Dataset':<20} {'Mode':<14} {'Objective':<12} {'Base AUC':<10} {'Ext AUC':<10} {'Comb-no':<10} {'Comb+ext':<10} {'N_no':<8} {'N_ext':<8}\n")
    f.write("-" * 102 + "\n")
    for r in all_results:
        f.write(f"{r['dataset']:<20} {r['mode']:<14} {r['objective']:<12.6f} {r['base_auc']:<10.4f} {r['ext_auc']:<10.4f} {r['comb_no_auc']:<10.4f} {r['comb_ext_auc']:<10.4f} {r['n_test_no']:<8} {r['n_test_ext']:<8}\n")

    # Per-dataset comparison
    f.write(f"\n\n{'='*80}\n")
    f.write("PER-DATASET COMPARISON\n")
    f.write(f"{'='*80}\n\n")
    datasets_seen = []
    for r in all_results:
        if r['dataset'] not in datasets_seen:
            datasets_seen.append(r['dataset'])
    for ds in datasets_seen:
        ds_results = [r for r in all_results if r['dataset'] == ds]
        f.write(f"\n{ds}:\n")
        best = min(ds_results, key=lambda x: x['objective'])
        for r in ds_results:
            marker = " <-- BEST" if r['mode'] == best['mode'] else ""
            f.write(f"  {r['mode']:<14} objective={r['objective']:.6f}  ext_auc={r['ext_auc']:.4f}{marker}\n")

print(f"\nSummary saved to: {summary_path}")
print("\nDone!")
