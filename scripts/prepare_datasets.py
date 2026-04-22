"""
Prepare datasets for baselines with EXACTLY the same preprocessing,
feature split, and train/test split as our incremental framework.

Self-contained — safe to import without side effects. The loader bodies
are duplicated verbatim from scripts/run_ablation_pruning.py so any change
there must be mirrored here (and vice-versa).

Public API per dataset:
    prepare_<name>() -> (df, label, base_features, ext_features)

Helper:
    split_train_test(df, label, ext_features, test_size=0.2, seed=42)

Registry:
    DATASET_LOADERS — dict mapping pretty name -> prepare function.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import kagglehub
except ImportError:
    kagglehub = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.GenericDataPipeline import GenericDataPipeline
from core.seed_utils import SEED, set_all_seeds

set_all_seeds()
pipeline = GenericDataPipeline()


# ============================================================================
# Raw loaders — mirror scripts/run_ablation_pruning.py exactly
# ============================================================================

def _load_bankloansta():
    path = kagglehub.dataset_download("zaurbegiev/my-dataset")
    csv_files = sorted([f for f in os.listdir(path) if f.endswith('.csv')])
    csv_path = os.path.join(path, csv_files[1])
    df = pd.read_csv(csv_path)
    df.dropna(subset=['Loan Status'], inplace=True)
    df.drop(['Loan ID', 'Customer ID'], axis=1, inplace=True)
    df = pipeline.preprocessing(df)
    label = "Loan Status"
    df[label] = df[label].astype(int)
    rng = np.random.RandomState(SEED)
    debt_median = df['Monthly Debt'].median()
    high_debt = df[df['Monthly Debt'] > debt_median].index
    low_debt = df[df['Monthly Debt'] <= debt_median].index
    noise1 = rng.choice(low_debt, size=int(0.10 * len(low_debt)), replace=False)
    df.loc[np.concatenate([high_debt, noise1]), 'Current Loan Amount'] = np.nan
    hist_median = df['Years of Credit History'].median()
    short_hist = df[df['Years of Credit History'] < hist_median].index
    long_hist = df[df['Years of Credit History'] >= hist_median].index
    noise2 = rng.choice(long_hist, size=int(0.10 * len(long_hist)), replace=False)
    df.loc[np.concatenate([short_hist, noise2]), 'Annual Income'] = np.nan
    acct_median = df['Number of Open Accounts'].median()
    many_accts = df[df['Number of Open Accounts'] > acct_median].index
    few_accts = df[df['Number of Open Accounts'] <= acct_median].index
    noise3 = rng.choice(few_accts, size=int(0.10 * len(few_accts)), replace=False)
    df.loc[np.concatenate([many_accts, noise3]), 'Credit Score'] = np.nan
    return df, label, ['Credit Score']


def _load_weather():
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
    return df, label, ['Evaporation']


def _load_diabetes():
    path = kagglehub.dataset_download("brandao/diabetes")
    df = pd.read_csv(os.path.join(path, "diabetic_data.csv"))
    df['readmitted'] = df['readmitted'].replace({'NO': 0, '<30': 1, '>30': 1}).astype(int)
    df.drop(['encounter_id', 'patient_nbr', 'number_inpatient',
             'number_emergency', 'discharge_disposition_id'], axis=1, inplace=True)
    df = pipeline.preprocessing(df)
    label = "readmitted"
    df[label] = df[label].astype(int)
    return df, label, ['medical_specialty', 'A1Cresult']


def _load_hr_analytics():
    path = kagglehub.dataset_download("arashnic/hr-analytics-job-change-of-data-scientists")
    df = pd.read_csv(os.path.join(path, "aug_train.csv"))
    df.drop(['enrollee_id'], axis=1, inplace=True)
    df = pipeline.preprocessing(df)
    label = "target"
    df[label] = df[label].astype(int)
    return df, label, ['company_size']


def _load_client_record_v2():
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
    return df, label, ['Tenure in Months']


def _load_movie_aug_v2():
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
    rng = np.random.RandomState(SEED)
    days_median = df['days_active'].median()
    inactive = df[df['days_active'] < days_median].index
    active = df[df['days_active'] >= days_median].index
    noise1 = rng.choice(active, size=int(0.10 * len(active)), replace=False)
    df.loc[np.concatenate([inactive, noise1]), 'rating_count'] = np.nan
    count_median = df['rating_count'].median()
    low_count = df[df['rating_count'] < count_median].index if not df['rating_count'].isna().all() else pd.Index([])
    high_count = df[df['rating_count'] >= count_median].index if not df['rating_count'].isna().all() else df.index
    if len(low_count) > 0 and len(high_count) > 0:
        noise2 = rng.choice(high_count, size=int(0.10 * len(high_count)), replace=False)
        df.loc[np.concatenate([low_count, noise2]), 'rating_mean'] = np.nan
    df.loc[np.concatenate([low_count, noise2]) if len(low_count) > 0 else pd.Index([]), 'rating_std'] = np.nan
    q1 = df['rating_count'].quantile(0.25)
    very_low = df[df['rating_count'] < q1].index if not df['rating_count'].isna().all() else pd.Index([])
    not_very_low = df[df['rating_count'] >= q1].index if not df['rating_count'].isna().all() else df.index
    if len(very_low) > 0 and len(not_very_low) > 0:
        noise3 = rng.choice(not_very_low, size=int(0.15 * len(not_very_low)), replace=False)
        df.loc[np.concatenate([very_low, noise3]), 'days_active'] = np.nan
        df.loc[np.concatenate([very_low, noise3]), 'rating_frequency'] = np.nan
    return df, label, ['rating_mean', 'tag_count', 'unique_tags',
                        'avg_tag_length', 'tag_frequency', 'last_tag']


def _load_weatheraus():
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'weatherAUS.csv')
    df = pd.read_csv(csv_path, na_values=['NA'])
    df = df.drop(columns=['Date', 'Location', 'RISK_MM', 'RainToday'])
    df = pipeline.preprocessing(df)
    label = "RainTomorrow"
    df[label] = df[label].astype(int)
    return df, label, ['Evaporation', 'Cloud9am']


def _load_wids():
    import xgboost as xgb
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'WIDS.csv')
    df = pd.read_csv(csv_path, na_values=['NA'])
    df = df.drop(columns=['encounter_id', 'patient_id', 'hospital_id'])
    df = pipeline.preprocessing(df)
    label = "hospital_death"
    df[label] = df[label].astype(int)
    X_all = df.drop(label, axis=1)
    selector = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                  tree_method='hist', random_state=SEED, eval_metric='auc')
    selector.fit(X_all, df[label], verbose=False)
    imp = pd.Series(selector.feature_importances_, index=X_all.columns).sort_values(ascending=False)
    keep = set(imp.head(50).index.tolist()) | {label}
    df = df[[c for c in df.columns if c in keep]]
    return df, label, ['h1_lactate_min',
                        'd1_lactate_max', 'd1_lactate_min',
                        'd1_pao2fio2ratio_max', 'd1_pao2fio2ratio_min']


def _load_flight_delay():
    path = kagglehub.dataset_download("divyansh22/flight-delay-prediction")
    df = pd.read_csv(os.path.join(path, "Jan_2019_ontime.csv"))
    drop_cols = ['Unnamed: 21', 'CANCELLED', 'DIVERTED', 'DEP_TIME', 'ARR_TIME',
                 'ARR_DEL15', 'ORIGIN_AIRPORT_SEQ_ID', 'DEST_AIRPORT_SEQ_ID',
                 'OP_CARRIER_AIRLINE_ID']
    df.drop([c for c in drop_cols if c in df.columns], axis=1, inplace=True)
    df.dropna(subset=['DEP_DEL15'], inplace=True)
    label = "DEP_DEL15"
    df[label] = df[label].astype(int)
    if len(df) > 100000:
        df = df.sample(n=100000, random_state=SEED).reset_index(drop=True)
    df = pipeline.preprocessing(df)
    rng = np.random.RandomState(SEED)
    regional = ['9E', 'EV', 'MQ', 'OH', 'OO', 'YV', 'YX']
    regional_idx = df[df['OP_CARRIER'].isin(regional)].index
    major_idx = df[~df['OP_CARRIER'].isin(regional)].index
    noise1 = rng.choice(major_idx, size=int(0.05 * len(major_idx)), replace=False)
    df.loc[np.concatenate([regional_idx, noise1]), 'OP_CARRIER_FL_NUM'] = np.nan
    weekend_idx = df[df['DAY_OF_WEEK'].isin([6, 7])].index
    weekday_idx = df[df['DAY_OF_WEEK'].isin([1, 2, 3, 4, 5])].index
    noise2 = rng.choice(weekday_idx, size=int(0.05 * len(weekday_idx)), replace=False)
    df.loc[np.concatenate([weekend_idx, noise2]), 'TAIL_NUM'] = np.nan
    early_idx = df[df['DEP_TIME_BLK'].isin(['0001-0559', '0600-0659'])].index
    other_idx = df[~df['DEP_TIME_BLK'].isin(['0001-0559', '0600-0659'])].index
    noise3 = rng.choice(other_idx, size=int(0.10 * len(other_idx)), replace=False)
    df.loc[np.concatenate([early_idx, noise3]), 'DISTANCE'] = np.nan
    dist_median = df['DISTANCE'].median()
    short_idx = df[df['DISTANCE'] < dist_median].index if not df['DISTANCE'].isna().all() else pd.Index([])
    long_idx = df[df['DISTANCE'] >= dist_median].index if not df['DISTANCE'].isna().all() else df.index
    if len(short_idx) > 0 and len(long_idx) > 0:
        noise4 = rng.choice(long_idx, size=int(0.10 * len(long_idx)), replace=False)
        df.loc[np.concatenate([short_idx, noise4]), 'DAY_OF_MONTH'] = np.nan
    return df, label, ['OP_CARRIER_FL_NUM']


def _load_credit_risk():
    import xgboost as xgb
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'CreditRisk')
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
    X_all = df.drop(label, axis=1)
    selector = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                  tree_method='hist', random_state=SEED, eval_metric='auc')
    selector.fit(X_all, df[label], verbose=False)
    imp = pd.Series(selector.feature_importances_, index=X_all.columns).sort_values(ascending=False)
    keep = set(imp.head(40).index.tolist()) | {label}
    df = df[[c for c in df.columns if c in keep]]
    return df, label, [
        'MEAN_AMTCR_1M_3M_TYPE_EQ_ACTIVE_DIV_MEAN_AMTCR_3M_12M_TYPE_EQ_ACTIVE_x',
        'STD_AMTCR_0M_6M_x',
        'MEAN_AMTCR_0M_6M_TYPE_EQ_CLOSED_x',
        'MEAN_AMTCR_0M_6M_TYPE_EQ_ACTIVE_x',
        'MEDIAN_AMTCR_0M_6M_x',
    ]


# ============================================================================
# Public wrappers — add base_features and has_extended column
# ============================================================================

def _wrap(loader):
    df, label, ext_features = loader()
    base_features = [c for c in df.columns if c != label and c not in ext_features]
    if 'has_extended' in df.columns:
        df = df.drop(columns=['has_extended'])
    df['has_extended'] = df[ext_features].notnull().any(axis=1).astype(int)
    return df, label, base_features, ext_features


def prepare_bankloansta():     return _wrap(_load_bankloansta)
def prepare_weather():         return _wrap(_load_weather)
def prepare_diabetes():        return _wrap(_load_diabetes)
def prepare_hr_analytics():    return _wrap(_load_hr_analytics)
def prepare_client_record():   return _wrap(_load_client_record_v2)
def prepare_movie_aug_v2():    return _wrap(_load_movie_aug_v2)
def prepare_weatheraus():      return _wrap(_load_weatheraus)
def prepare_wids():            return _wrap(_load_wids)
def prepare_flight_delay():    return _wrap(_load_flight_delay)
def prepare_credit_risk():     return _wrap(_load_credit_risk)


DATASET_LOADERS = {
    'BankLoanSta':     prepare_bankloansta,
    'Weather':         prepare_weather,
    'DiabetesRecord':  prepare_diabetes,
    'CreditRisk':      prepare_credit_risk,
    'HRAnalytics':     prepare_hr_analytics,
    'ClientRecord':    prepare_client_record,
    'MovieAugV2':      prepare_movie_aug_v2,
    'WeatherAUS':      prepare_weatheraus,
    'WIDS':            prepare_wids,
    'FlightDelay':     prepare_flight_delay,
}


def split_train_test(df, label, ext_features, test_size=0.2, seed=SEED):
    """Stratified train/test split by (label, has_extended) — matches core/RunData.py."""
    df = df.copy()
    if 'has_extended' not in df.columns:
        df['has_extended'] = df[ext_features].notnull().any(axis=1).astype(int)
    strat_col = df[label].astype(str) + '_' + df['has_extended'].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(label, axis=1),
        df[label],
        test_size=test_size,
        random_state=seed,
        stratify=strat_col,
    )
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    return train_df, test_df


if __name__ == '__main__':
    sel = sys.argv[1].split(',') if len(sys.argv) > 1 else list(DATASET_LOADERS)
    for name in sel:
        if name not in DATASET_LOADERS:
            print(f"Unknown dataset: {name}")
            continue
        print(f"\n{'='*80}\nDataset: {name}\n{'='*80}")
        df, label, base_features, ext_features = DATASET_LOADERS[name]()
        n_ext = int(df['has_extended'].sum())
        n_no = len(df) - n_ext
        print(f"Shape: {df.shape}")
        print(f"Label: {label}  distribution: {df[label].value_counts().to_dict()}")
        print(f"Base features ({len(base_features)}), Ext features ({len(ext_features)}): {ext_features}")
        print(f"has_extended: {n_ext} ({n_ext/len(df)*100:.1f}%)  no_extended: {n_no} ({n_no/len(df)*100:.1f}%)")
        train_df, test_df = split_train_test(df, label, ext_features)
        t_no = int((test_df['has_extended'] == 0).sum())
        t_ext = int((test_df['has_extended'] == 1).sum())
        print(f"Test: n_no={t_no}  n_ext={t_ext}")
