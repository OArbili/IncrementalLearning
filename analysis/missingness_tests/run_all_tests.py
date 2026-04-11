#!/usr/bin/env python3
"""Missingness Structure Tests: Chi-squared + Logistic Regression for all 10 datasets.

Tests whether missingness is structured (NMAR/MAR) or random (MCAR).
- Chi-squared: tests if label distribution differs between has_extended=0 and has_extended=1
- Logistic regression: predicts has_extended from base features (AUC >> 0.5 = structured)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from core.GenericDataPipeline import GenericDataPipeline
from core.seed_utils import SEED, set_all_seeds

pd.set_option('future.infer_string', False)
set_all_seeds()

pipeline = GenericDataPipeline()
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

# ============================================================================
# Dataset loaders (same preprocessing as run_ablation_pruning.py)
# ============================================================================

def load_bankloansta():
    import kagglehub
    path = kagglehub.dataset_download("zaurbegiev/my-dataset")
    csv_files = sorted([f for f in os.listdir(path) if f.endswith('.csv')])
    df = pd.read_csv(os.path.join(path, csv_files[1]))
    df.dropna(subset=['Loan Status'], inplace=True)
    df.drop(['Loan ID', 'Customer ID'], axis=1, inplace=True)
    df = pipeline.preprocessing(df)
    label = "Loan Status"
    df[label] = df[label].astype(int)
    # Augmentation: STRUCTURED nulls (conditional on base features)
    rng = np.random.RandomState(SEED)
    debt_median = df['Monthly Debt'].median()
    high_debt = df[df['Monthly Debt'] > debt_median].index
    low_debt = df[df['Monthly Debt'] <= debt_median].index
    noise_idx = rng.choice(low_debt, size=int(0.10 * len(low_debt)), replace=False)
    df.loc[np.concatenate([high_debt, noise_idx]), 'Current Loan Amount'] = np.nan
    ext_features = ['Current Loan Amount']
    return df, label, ext_features, 'augmented'

def load_weather():
    import kagglehub
    path = kagglehub.dataset_download("rever3nd/weather-data")
    csv_files = sorted([f for f in os.listdir(path) if f.endswith('.csv')])
    df = pd.read_csv(os.path.join(path, csv_files[0]))
    drop_cols = ['Unnamed: 0', 'Date', 'RISK_MM', 'Humidity3pm', 'Pressure3pm', 'Cloud3pm', 'Temp3pm', 'WindDir3pm', 'WindSpeed3pm']
    df.drop([c for c in drop_cols if c in df.columns], axis=1, inplace=True)
    df = pipeline.preprocessing(df)
    label = "RainTomorrow"
    df[label] = df[label].astype(int)
    ext_features = ['Evaporation']
    return df, label, ext_features, 'natural'

def load_diabetes():
    import kagglehub
    path = kagglehub.dataset_download("brandao/diabetes")
    df = pd.read_csv(os.path.join(path, "diabetic_data.csv"))
    df['readmitted'] = df['readmitted'].replace({'NO': 0, '<30': 1, '>30': 1}).astype(int)
    df.drop(['encounter_id', 'patient_nbr', 'number_inpatient', 'number_emergency', 'discharge_disposition_id'], axis=1, inplace=True)
    df = pipeline.preprocessing(df)
    label = "readmitted"
    df[label] = df[label].astype(int)
    ext_features = ['medical_specialty', 'A1Cresult']
    return df, label, ext_features, 'natural'

def load_creditrisk():
    import xgboost as xgb
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'CreditRisk')
    data1 = pd.read_csv(os.path.join(data_dir, 'data_devsample.csv'))
    data2 = pd.read_csv(os.path.join(data_dir, 'data_to_score.csv'))
    df = pd.merge(data1, data2, on='SK_ID_CURR', how='inner')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    cols_to_remove = [c for c in df.columns if c.endswith('_y') or c in ['TIME_x', 'BASE_x', 'DAY_x', 'MONTH_x']]
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
    # Feature selection: top 40
    X_all = df.drop(label, axis=1)
    selector = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, tree_method='hist', random_state=SEED, eval_metric='auc')
    selector.fit(X_all, df[label], verbose=False)
    imp = pd.Series(selector.feature_importances_, index=X_all.columns).sort_values(ascending=False)
    keep = set(imp.head(40).index.tolist()) | {label}
    df = df[[c for c in df.columns if c in keep]]
    ext_features = [
        'MEAN_AMTCR_1M_3M_TYPE_EQ_ACTIVE_DIV_MEAN_AMTCR_3M_12M_TYPE_EQ_ACTIVE_x',
        'STD_AMTCR_0M_6M_x', 'MEAN_AMTCR_0M_6M_TYPE_EQ_CLOSED_x',
        'MEAN_AMTCR_0M_6M_TYPE_EQ_ACTIVE_x', 'MEDIAN_AMTCR_0M_6M_x',
    ]
    return df, label, ext_features, 'natural'

def load_hranalytics():
    import kagglehub
    path = kagglehub.dataset_download("arashnic/hr-analytics-job-change-of-data-scientists")
    df = pd.read_csv(os.path.join(path, "aug_train.csv"))
    df.drop(['enrollee_id'], axis=1, inplace=True)
    df = pipeline.preprocessing(df)
    label = "target"
    df[label] = df[label].astype(int)
    ext_features = ['company_size']
    return df, label, ext_features, 'natural'

def load_clientrecord():
    import kagglehub
    path = kagglehub.dataset_download("shilongzhuang/telecom-customer-churn-by-maven-analytics")
    csv_files = sorted([f for f in os.listdir(path) if f.endswith('.csv')])
    data1 = pd.read_csv(os.path.join(path, csv_files[2]))
    data2 = pd.read_csv(os.path.join(path, csv_files[0]))
    df = pd.merge(data2, data1, on='Zip Code')
    df['Customer Status'] = df['Customer Status'].apply(lambda x: 1 if x == 'Stayed' else 0)
    drop_cols = ['Customer ID', 'Churn Category', 'Churn Reason', 'Total Charges', 'Total Revenue',
                 'Total Refunds', 'Total Long Distance Charges', 'Zip Code', 'City', 'Latitude', 'Longitude']
    df.drop([c for c in drop_cols if c in df.columns], axis=1, inplace=True)
    df = pipeline.preprocessing(df)
    label = "Customer Status"
    df[label] = df[label].astype(int)
    # Augmentation: STRUCTURED nulls (conditional on base features)
    rng = np.random.RandomState(SEED)
    tenure_median = df['Tenure in Months'].median()
    contract_candidates = df[df['Tenure in Months'] < tenure_median].index
    above_median = df[df['Tenure in Months'] >= tenure_median].index
    noise_idx = rng.choice(above_median, size=int(0.10 * len(above_median)), replace=False)
    df.loc[np.concatenate([contract_candidates, noise_idx]), 'Contract'] = np.nan
    no_ref = df[df['Number of Referrals'] == 0].index
    has_ref = df[df['Number of Referrals'] > 0].index
    noise_idx2 = rng.choice(has_ref, size=int(0.15 * len(has_ref)), replace=False)
    df.loc[np.concatenate([no_ref, noise_idx2]), 'Tenure in Months'] = np.nan
    non_paper = df[df['Paperless Billing'] == 0].index
    paper = df[df['Paperless Billing'] == 1].index
    noise_idx3 = rng.choice(paper, size=int(0.10 * len(paper)), replace=False)
    df.loc[np.concatenate([non_paper, noise_idx3]), 'Monthly Charge'] = np.nan
    ext_features = ['Offer', 'Monthly Charge']
    return df, label, ext_features, 'augmented'

def load_movieaugv2():
    import kagglehub
    path = kagglehub.dataset_download("grouplens/movielens-20m-dataset")
    ratings = pd.read_csv(os.path.join(path, "rating.csv"))
    tags = pd.read_csv(os.path.join(path, "tag.csv"))
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp']).astype(np.int64) // 10**9
    tags['timestamp'] = pd.to_datetime(tags['timestamp']).astype(np.int64) // 10**9
    cutoff_date = ratings['timestamp'].quantile(0.8)
    target_date = ratings['timestamp'].quantile(0.9)
    user_stats = ratings[ratings['timestamp'] < cutoff_date].groupby('userId').agg({
        'rating': ['count', 'mean', 'std'], 'timestamp': ['min', 'max']}).round(3)
    user_stats.columns = ['rating_count', 'rating_mean', 'rating_std', 'first_rating', 'last_rating']
    df = user_stats.reset_index()
    df['days_active'] = (df['last_rating'] - df['first_rating']) / (24 * 60 * 60)
    df['rating_frequency'] = df['rating_count'] / df['days_active'].clip(lower=1)
    future_activity = ratings[(ratings['timestamp'] >= cutoff_date) & (ratings['timestamp'] < target_date)].groupby('userId')['rating'].count().reset_index()
    future_activity.columns = ['userId', 'future_ratings']
    future_activity['TARGET'] = (future_activity['future_ratings'] > future_activity['future_ratings'].median()).astype(int)
    df = df.merge(future_activity[['userId', 'TARGET']], on='userId', how='left')
    df['TARGET'] = df['TARGET'].fillna(0)
    tag_activity = tags[tags['timestamp'] < cutoff_date].groupby('userId').agg({'tag': ['count', 'nunique'], 'timestamp': ['min', 'max']})
    tag_activity.columns = ['tag_count', 'unique_tags', 'first_tag', 'last_tag']
    tag_activity = tag_activity.reset_index()
    tag_activity['days_tagging'] = (tag_activity['last_tag'] - tag_activity['first_tag']) / (24 * 60 * 60)
    tag_activity['tag_frequency'] = tag_activity['tag_count'] / tag_activity['days_tagging'].clip(lower=1)
    tag_lengths = tags[tags['timestamp'] < cutoff_date].groupby('userId')['tag'].apply(lambda x: np.mean([len(str(t)) for t in x])).reset_index()
    tag_lengths.columns = ['userId', 'avg_tag_length']
    tag_activity = tag_activity.merge(tag_lengths, on='userId', how='left')
    df = df.merge(tag_activity, on='userId', how='left')
    columns_to_keep = ['rating_count', 'rating_mean', 'rating_std', 'days_active', 'rating_frequency',
                       'tag_count', 'unique_tags', 'avg_tag_length', 'tag_frequency', 'last_tag', 'TARGET']
    df = df[columns_to_keep]
    df = pipeline.preprocessing(df)
    label = "TARGET"
    df[label] = df[label].astype(int)
    rng = np.random.RandomState(SEED)
    for feat in ['rating_count', 'rating_mean', 'rating_std', 'days_active', 'rating_frequency']:
        null_idx = rng.choice(df.index, size=int(0.50 * len(df)), replace=False)
        df.loc[null_idx, feat] = np.nan
    ext_features = ['rating_mean', 'tag_count', 'unique_tags', 'avg_tag_length', 'tag_frequency', 'last_tag']
    return df, label, ext_features, 'augmented'

def load_weatheraus():
    csv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'weatherAUS.csv')
    df = pd.read_csv(csv_path, na_values=['NA'])
    df.drop(columns=['Date', 'Location', 'RISK_MM', 'RainToday'], inplace=True)
    df = pipeline.preprocessing(df)
    label = "RainTomorrow"
    df[label] = df[label].astype(int)
    ext_features = ['Evaporation', 'Cloud9am']
    return df, label, ext_features, 'natural'

def load_wids():
    import xgboost as xgb
    csv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'WIDS.csv')
    df = pd.read_csv(csv_path, na_values=['NA'])
    df.drop(columns=['encounter_id', 'patient_id', 'hospital_id'], inplace=True)
    df = pipeline.preprocessing(df)
    label = "hospital_death"
    df[label] = df[label].astype(int)
    # Feature selection: top 50
    X_all = df.drop(label, axis=1)
    selector = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, tree_method='hist', random_state=SEED, eval_metric='auc')
    selector.fit(X_all, df[label], verbose=False)
    imp = pd.Series(selector.feature_importances_, index=X_all.columns).sort_values(ascending=False)
    keep = set(imp.head(50).index.tolist()) | {label}
    df = df[[c for c in df.columns if c in keep]]
    ext_features = ['h1_lactate_min', 'd1_lactate_max', 'd1_lactate_min', 'd1_pao2fio2ratio_max', 'd1_pao2fio2ratio_min']
    return df, label, ext_features, 'natural'

def load_flightdelay():
    import kagglehub
    path = kagglehub.dataset_download("divyansh22/flight-delay-prediction")
    df = pd.read_csv(os.path.join(path, "Jan_2019_ontime.csv"))
    drop_cols = ['Unnamed: 21', 'CANCELLED', 'DIVERTED', 'DEP_TIME', 'ARR_TIME', 'ARR_DEL15',
                 'ORIGIN_AIRPORT_SEQ_ID', 'DEST_AIRPORT_SEQ_ID', 'OP_CARRIER_AIRLINE_ID']
    df.drop([c for c in drop_cols if c in df.columns], axis=1, inplace=True)
    df.dropna(subset=['DEP_DEL15'], inplace=True)
    label = "DEP_DEL15"
    df[label] = df[label].astype(int)
    if len(df) > 100000:
        df = df.sample(n=100000, random_state=SEED).reset_index(drop=True)
    df = pipeline.preprocessing(df)
    rng = np.random.RandomState(SEED)
    for feat in ['TAIL_NUM', 'DISTANCE', 'OP_CARRIER_FL_NUM', 'DAY_OF_MONTH']:
        null_idx = rng.choice(df.index, size=int(0.20 * len(df)), replace=False)
        df.loc[null_idx, feat] = np.nan
    ext_features = ['OP_CARRIER_FL_NUM']
    return df, label, ext_features, 'augmented'

# ============================================================================
# Test functions
# ============================================================================

def run_tests(df, label, ext_features, dataset_name, dataset_type):
    """Run chi-squared and logistic regression tests."""
    print(f"\n{'='*70}")
    print(f"DATASET: {dataset_name} ({dataset_type})")
    print(f"{'='*70}")

    # Check ext features exist
    missing_feats = [f for f in ext_features if f not in df.columns]
    if missing_feats:
        print(f"WARNING: Missing ext features: {missing_feats}")
        ext_features = [f for f in ext_features if f in df.columns]
        if not ext_features:
            return None

    # Compute has_extended
    has_ext = df[ext_features].notnull().any(axis=1).astype(int)
    n_ext = has_ext.sum()
    n_no_ext = len(df) - n_ext

    print(f"Rows: {len(df)}, has_extended: {n_ext} ({n_ext/len(df)*100:.1f}%), no_ext: {n_no_ext}")

    # --- Test A: Chi-squared ---
    ct = pd.crosstab(has_ext, df[label])
    chi2, chi2_p, dof, expected = stats.chi2_contingency(ct)

    label_rate_ext = df[has_ext == 1][label].mean()
    label_rate_no_ext = df[has_ext == 0][label].mean()

    print(f"\nTest A — Chi-squared (label vs has_extended):")
    print(f"  Label rate (no ext):  {label_rate_no_ext:.4f}")
    print(f"  Label rate (has ext): {label_rate_ext:.4f}")
    print(f"  Chi2 = {chi2:.2f}, p = {chi2_p:.2e}")
    print(f"  {'STRUCTURED (p < 0.05)' if chi2_p < 0.05 else 'NOT SIGNIFICANT (p >= 0.05)'}")

    # --- Test B: Logistic Regression ---
    base_features = [f for f in df.columns if f != label and f not in ext_features]
    X = df[base_features].copy()

    # Fill NaN for logistic regression (mean imputation)
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    y = has_ext.values

    # Check for enough samples in both classes
    if y.sum() < 10 or (len(y) - y.sum()) < 10:
        print(f"\nTest B — Logistic Regression: SKIPPED (insufficient class balance)")
        logreg_auc_mean = np.nan
        logreg_auc_std = np.nan
    else:
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(max_iter=1000, random_state=SEED, solver='lbfgs'))
        ])
        scores = cross_val_score(pipe, X, y, cv=5, scoring='roc_auc')
        logreg_auc_mean = scores.mean()
        logreg_auc_std = scores.std()

        print(f"\nTest B — Logistic Regression (predict has_extended from base features):")
        print(f"  5-fold CV AUC: {logreg_auc_mean:.4f} +/- {logreg_auc_std:.4f}")
        if logreg_auc_mean > 0.6:
            print(f"  STRUCTURED (AUC >> 0.5, missingness predictable from base features)")
        elif logreg_auc_mean > 0.55:
            print(f"  WEAKLY STRUCTURED (AUC slightly > 0.5)")
        else:
            print(f"  RANDOM (AUC ~ 0.5, missingness not predictable)")

    # Classification
    if chi2_p < 0.05 and logreg_auc_mean > 0.6:
        missingness_type = "Structured"
    elif chi2_p < 0.05 or logreg_auc_mean > 0.55:
        missingness_type = "Weakly Structured"
    else:
        missingness_type = "Random"

    print(f"\n  CLASSIFICATION: {missingness_type}")

    return {
        'dataset': dataset_name,
        'type': dataset_type,
        'n_rows': len(df),
        'n_no_ext': n_no_ext,
        'n_ext': n_ext,
        'label_rate_no_ext': round(label_rate_no_ext, 4),
        'label_rate_ext': round(label_rate_ext, 4),
        'chi2_stat': round(chi2, 2),
        'chi2_p': chi2_p,
        'logreg_auc_mean': round(logreg_auc_mean, 4) if not np.isnan(logreg_auc_mean) else np.nan,
        'logreg_auc_std': round(logreg_auc_std, 4) if not np.isnan(logreg_auc_std) else np.nan,
        'missingness_type': missingness_type,
    }

# ============================================================================
# Main
# ============================================================================

DATASETS = [
    ('BankLoanSta', load_bankloansta),
    ('Weather', load_weather),
    ('DiabetesRecord', load_diabetes),
    ('CreditRisk', load_creditrisk),
    ('HRAnalytics', load_hranalytics),
    ('ClientRecord', load_clientrecord),
    ('MovieAugV2', load_movieaugv2),
    ('WeatherAUS', load_weatheraus),
    ('WIDS', load_wids),
    ('FlightDelay', load_flightdelay),
]

if __name__ == '__main__':
    # Filter if argument provided
    ds_filter = sys.argv[1].split(',') if len(sys.argv) > 1 else None

    results = []
    for ds_name, load_fn in DATASETS:
        if ds_filter and ds_name not in ds_filter:
            print(f"\nSkipping {ds_name}")
            continue
        try:
            df, label, ext_features, ds_type = load_fn()
            result = run_tests(df, label, ext_features, ds_name, ds_type)
            if result:
                results.append(result)
        except Exception as e:
            print(f"\nERROR loading {ds_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save CSV
    if results:
        results_df = pd.DataFrame(results)
        csv_path = os.path.join(RESULTS_DIR, 'missingness_results.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"\n\nResults saved to: {csv_path}")

        # Summary
        print(f"\n{'='*80}")
        print("SUMMARY: MISSINGNESS STRUCTURE TESTS")
        print(f"{'='*80}\n")
        print(f"{'Dataset':<20} {'Type':<12} {'Chi2 p':>12} {'LogReg AUC':>12} {'Classification':<20}")
        print("-" * 80)
        for r in results:
            chi2_s = f"{r['chi2_p']:.2e}" if r['chi2_p'] is not None else "N/A"
            auc_s = f"{r['logreg_auc_mean']:.4f}" if not np.isnan(r['logreg_auc_mean']) else "N/A"
            print(f"{r['dataset']:<20} {r['type']:<12} {chi2_s:>12} {auc_s:>12} {r['missingness_type']:<20}")

        # Save summary
        summary_path = os.path.join(RESULTS_DIR, 'missingness_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("MISSINGNESS STRUCTURE TESTS\n")
            f.write("=" * 80 + "\n\n")
            f.write("Tests:\n")
            f.write("  A. Chi-squared: tests if label distribution differs between has_extended=0 and =1\n")
            f.write("  B. Logistic Regression: predicts has_extended from base features (5-fold CV AUC)\n\n")
            f.write(f"{'Dataset':<20} {'Type':<12} {'Chi2 p':>12} {'LogReg AUC':>12} {'Classification':<20}\n")
            f.write("-" * 80 + "\n")
            for r in results:
                chi2_s = f"{r['chi2_p']:.2e}" if r['chi2_p'] is not None else "N/A"
                auc_s = f"{r['logreg_auc_mean']:.4f}" if not np.isnan(r['logreg_auc_mean']) else "N/A"
                f.write(f"{r['dataset']:<20} {r['type']:<12} {chi2_s:>12} {auc_s:>12} {r['missingness_type']:<20}\n")
        print(f"\nSummary saved to: {summary_path}")

    print("\nDone!")
