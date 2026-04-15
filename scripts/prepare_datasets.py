"""
Prepare WIDS and ClientRecord datasets: preprocessing, feature split, no model training.
Returns label column, base features, and extended features for each dataset.
"""
import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import kagglehub

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.GenericDataPipeline import GenericDataPipeline
from core.seed_utils import set_all_seeds

SEED = 42
set_all_seeds()
pipeline = GenericDataPipeline()


def prepare_wids():
    """WIDS: natural nulls, top 50 features by importance, 5 ext features."""
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'WIDS.csv')
    df = pd.read_csv(csv_path, na_values=['NA'])
    df.drop(columns=['encounter_id', 'patient_id', 'hospital_id'], inplace=True)
    df = pipeline.preprocessing(df)
    label = "hospital_death"
    df[label] = df[label].astype(int)

    # Feature selection: top 50 by importance
    X_all = df.drop(label, axis=1)
    selector = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                  tree_method='hist', random_state=SEED, eval_metric='auc')
    selector.fit(X_all, df[label], verbose=False)
    imp = pd.Series(selector.feature_importances_, index=X_all.columns).sort_values(ascending=False)
    keep = set(imp.head(50).index.tolist()) | {label}
    df = df[[c for c in df.columns if c in keep]]

    ext_features = [
        'h1_lactate_min',
        'd1_lactate_max', 'd1_lactate_min',
        'd1_pao2fio2ratio_max', 'd1_pao2fio2ratio_min',
    ]
    base_features = [c for c in df.columns if c != label and c not in ext_features]

    # has_extended: 1 if ANY ext feature is non-null
    df['has_extended'] = df[ext_features].notnull().any(axis=1).astype(int)

    return df, label, base_features, ext_features


def prepare_client_record():
    """ClientRecord v2: structured null injection, Tenure in Months ext."""
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

    # Structured null injection
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
    base_features = [c for c in df.columns if c != label and c not in ext_features]

    # has_extended: 1 if ANY ext feature is non-null
    df['has_extended'] = df[ext_features].notnull().any(axis=1).astype(int)

    return df, label, base_features, ext_features


if __name__ == '__main__':
    for name, prepare_fn in [('WIDS', prepare_wids), ('ClientRecord', prepare_client_record)]:
        print(f"\n{'='*80}")
        print(f"Dataset: {name}")
        print(f"{'='*80}")
        df, label, base_features, ext_features = prepare_fn()

        n_ext = df['has_extended'].sum()
        n_no = len(df) - n_ext

        print(f"Shape: {df.shape}")
        print(f"Label: {label}")
        print(f"  Distribution: {df[label].value_counts().to_dict()}")
        print(f"\nBase features ({len(base_features)}):")
        for f in base_features:
            null_pct = df[f].isna().mean() * 100
            print(f"  {f:<40} null: {null_pct:.1f}%")
        print(f"\nExtended features ({len(ext_features)}):")
        for f in ext_features:
            null_pct = df[f].isna().mean() * 100
            print(f"  {f:<40} null: {null_pct:.1f}%")
        print(f"\nhas_extended: {n_ext} ({n_ext/len(df)*100:.1f}%), no_extended: {n_no} ({n_no/len(df)*100:.1f}%)")
