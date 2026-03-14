import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from RunData import RunPipeline
from seed_utils import SEED, DEVICE


class GenericDataPipeline:

    def rank_features(self, X, y, n_folds=5):
        print(f"Starting feature importance calculation with XGBoost and {n_folds}-fold CV...")

        X_copy = X.copy()

        null_ratios = {}
        for col in X_copy.columns:
            null_ratios[col] = X_copy[col].isna().mean()

        for col in X_copy.select_dtypes(include=['int64', 'float64']).columns:
            X_copy[col] = X_copy[col].replace([np.inf, -np.inf], np.nan)

            non_null = X_copy[col].dropna()
            if len(non_null) > 0:
                mean_val = non_null.mean()
                std_val = non_null.std()
                if std_val > 0 and not np.isnan(mean_val):
                    upper_bound = mean_val + 5 * std_val
                    lower_bound = mean_val - 5 * std_val
                    X_copy[col] = X_copy[col].clip(lower=lower_bound, upper=upper_bound)

        num_cols = X_copy.select_dtypes(include=['int64', 'float64']).columns
        cat_cols = X_copy.select_dtypes(include=['object']).columns

        X_processed = X_copy.copy()

        for col in cat_cols:
            if X_processed[col].isna().sum() > 0:
                most_freq = X_processed[col].mode()[0] if not X_processed[col].isna().all() else 'Unknown'
                X_processed[col] = X_processed[col].fillna(most_freq)
            X_processed[col] = X_processed[col].astype('category')

        for col in num_cols:
            if X_processed[col].isna().sum() > 0:
                median_val = X_processed[col].median() if not X_processed[col].isna().all() else 0
                X_processed[col] = X_processed[col].fillna(median_val)

        X_filled = X_processed

        mean_target = float(np.mean(y))
        print(f"Mean target value (for base_score): {mean_target:.6f}")

        base_score = 0.5
        if 0 < mean_target < 1:
            base_score = mean_target

        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'alpha': 0.01,
            'lambda': 1.0,
            'gamma': 0.1,
            'base_score': base_score,
            'seed': SEED,
            'tree_method': 'hist',
            'device': DEVICE
        }

        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

        fold_importances = []

        fold_scores = []

        print(f"Training XGBoost models across {n_folds} folds...")

        y_values = set(y.unique())
        print(f"Unique target values: {y_values}")

        if not all(isinstance(val, (int, float)) for val in y_values):
            print("Warning: Target variable contains non-numeric values. Converting to numeric.")
            y = y.astype(float)

        if not all(val in [0, 1] for val in y_values):
            print("Warning: Target variable contains values other than 0 and 1.")
            print("Converting target to binary: 0 for negative class, 1 for positive class.")
            y = (y > 0).astype(int)

        # Perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_filled, y)):
            print(f"Fold {fold+1}/{n_folds}")

            X_train, X_val = X_filled.iloc[train_idx], X_filled.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            train_pos = np.mean(y_train)
            val_pos = np.mean(y_val)
            print(f"  Train positive rate: {train_pos:.4f}, Validation positive rate: {val_pos:.4f}")

            try:
                dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
                dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

                # Train model
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=100,
                    early_stopping_rounds=20,
                    evals=[(dval, 'validation')],
                    verbose_eval=False
                )

                y_pred = model.predict(dval)
                score = roc_auc_score(y_val, y_pred)
                fold_scores.append(score)

                try:
                    importance = model.get_score(importance_type='gain')

                    fold_importance = {col: importance.get(col, 0) for col in X_filled.columns}

                    max_value = max(fold_importance.values()) if max(fold_importance.values()) > 0 else 1
                    normalized_importance = {col: val/max_value for col, val in fold_importance.items()}

                    fold_importances.append(normalized_importance)
                except Exception as e:
                    print(f"Warning: Could not get feature importance for fold {fold+1}: {str(e)}")

            except Exception as e:
                print(f"Error in fold {fold+1}: {str(e)}")
                print("Skipping this fold and continuing with others.")

        if fold_scores:
            mean_auc = np.mean(fold_scores)
            print(f"Mean AUC across {len(fold_scores)} successful folds: {mean_auc:.4f}")
        else:
            print("No successful folds. Cannot calculate mean AUC.")

        if not fold_importances:
            print("No feature importance information could be gathered from any fold.")
            print("Using a simple fallback method for feature scoring.")

            rng = np.random.RandomState(SEED)
            avg_importance = {}
            for col in X_filled.columns:
                try:
                    valid_mask = ~pd.isna(X_copy[col]) & ~pd.isna(y)
                    if valid_mask.sum() > 10:
                        if X_filled[col].dtype.name == 'category':
                            col_values = X_filled[col][valid_mask].cat.codes
                        else:
                            col_values = X_filled[col][valid_mask]

                        y_values = y[valid_mask]
                        corr = abs(np.corrcoef(col_values, y_values)[0, 1])
                        avg_importance[col] = corr if not np.isnan(corr) else 0
                    else:
                        avg_importance[col] = rng.uniform(0, 0.001)
                except Exception as e:
                    print(f"Warning: Could not calculate correlation for {col}: {str(e)}")
                    avg_importance[col] = rng.uniform(0, 0.001)
        else:
            avg_importance = {}
            for col in X_filled.columns:
                importances = [fold_imp.get(col, 0) for fold_imp in fold_importances]
                if importances:
                    avg_importance[col] = np.mean(importances)
                else:
                    null_ratio = null_ratios.get(col, 0)
                    avg_importance[col] = max(0.001 * (1 - null_ratio), 0.0001)

        feature_data = []
        for col in X_copy.columns:
            gain_score = avg_importance.get(col, 0)
            null_ratio = null_ratios.get(col, 0)
            feature_data.append({
                'feature_name': col,
                'gain_score': gain_score,
                'null_ratio': null_ratio
            })

        feature_df = pd.DataFrame(feature_data)
        feature_df = feature_df.sort_values('gain_score', ascending=False)

        print("\nFeature scores:")
        print("-" * 100)
        print(f"{'Rank':<5} | {'Feature':<30} | {'Gain Score':<15} | {'Null Ratio':<10}")
        print("-" * 100)

        for rank, (_, row) in enumerate(feature_df.iterrows(), 1):
            feat = row['feature_name']
            gain = row['gain_score']
            null_ratio = row['null_ratio']
            print(f"{rank:<5} | {feat:<30} | {gain:.6f} | {null_ratio:.4f}")

        return feature_df

    def objective(self, trial, feature_scores_with_nulls, all_features_score, df, label, n_trials=20):
        K = 10
        selected_features = feature_scores_with_nulls['feature_name'].to_list()[:K]
        print("selected_features")
        print(selected_features)
        all_features = all_features_score['feature_name'].to_list()
        # Create binary assignment for each of the selected features:
        assignment = {}
        for feat in selected_features:
            # 1 means feature goes to stage1, 0 means feature is used in stage2
            assignment[feat] = trial.suggest_categorical(f"assign_{feat}", [1, 0])

        for feat in all_features:
            if feat not in assignment.keys():
                assignment[feat] = 1
        vals = 0
        for key,value in assignment.items():
            vals +=value
        if vals==0:
            return 99999
        stage1_features = [feat for feat, assign in assignment.items() if assign == 1]
        stage2_features = [feat for feat, assign in assignment.items() if assign == 0]
        csv_name = f"trial_{trial.number}_results.csv"
        dm = RunPipeline()
        return dm.full_run(df, stage1_features, stage2_features, label, csv_name, n_trials=n_trials)

    def preprocessing(self,df):
        for col in df.columns:
            if df[col].dtype == 'object':
                # Replace ? with NaN
                df[col] = df[col].replace(['?', ''], np.nan)
                # Convert to categorical codes
                if df[col].isna().sum() < len(df):  # If not all values are NaN
                    df[col] = pd.Categorical(df[col]).codes

                    # Ensure -1 (missing) values are converted to NaN
                    df[col] = df[col].replace(-1, np.nan)
        # Convert boolean columns to int
        for col in df.select_dtypes(include=['bool']).columns:
            df[col] = df[col].astype(int)
        return df
