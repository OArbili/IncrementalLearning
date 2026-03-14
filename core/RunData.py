import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from .XGBoostModel import XGBoostModel
from .seed_utils import SEED

class RunPipeline():

    def __init__(self):
        self.data = None
        self.base_features = None
        self.ext_features = None
        self.label = None
        self.all_features = None
        self.seed = SEED

    def load_data(self, in_base_features, in_ext_features, data, label):
        self.data = data
        self.base_features = in_base_features
        self.ext_features = in_ext_features
        self.all_features = self.base_features + self.ext_features
        self.label = label

    def set_has_extended(self):
        self.data['has_extended'] = np.where(
            self.data[self.ext_features].notnull().any(axis=1).astype(int) == 0, 0, 1
        )
        if 'has_extended' not in self.ext_features:
            self.ext_features = self.ext_features + ['has_extended']
        if 'has_extended' not in self.all_features:
            self.all_features = self.all_features + ['has_extended']

    def train_test_split(self):
        self.data['strat_col'] = self.data[self.label].astype(str) + '_' + self.data['has_extended'].astype(str)

        print("\n=== Breakdown BEFORE splitting ===")
        print(self.data['has_extended'].value_counts(dropna=False))
        print("Extended percentage:", round(self.data['has_extended'].mean()*100, 2), "%")

        if self.data['has_extended'].nunique() < 2:
            print("Not enough variation in has_extended — all rows have or all rows lack extended features!")
            return 999

        if self.data['strat_col'].value_counts().min() < 2:
            print("Not enough samples in at least one group of strat_col")
            return 999

        X_train, X_test, y_train, y_test = train_test_split(
            self.data.drop(self.label, axis=1),
            self.data[self.label],
            test_size=0.2,
            random_state=self.seed,
            stratify=self.data['strat_col']
        )

        self.train_df = pd.concat([X_train, y_train], axis=1)
        self.test_df = pd.concat([X_test, y_test], axis=1)

        self.train_df = self.train_df.drop('strat_col', axis=1)
        self.test_df = self.test_df.drop('strat_col', axis=1)

        print("Train set distribution:")
        print(self.train_df.groupby([self.label, 'has_extended']).size())
        print("\nTest set distribution:")
        print(self.test_df.groupby([self.label, 'has_extended']).size())
        train_counts = self.train_df.groupby([self.label, 'has_extended']).size()
        test_counts = self.test_df.groupby([self.label, 'has_extended']).size()
        if (train_counts < 100).any() or (test_counts < 100).any():
            print("\n One of the train/test groups has fewer than 100 samples!")
            return 999

    def set_train_base_ext_datasets(self):
        self.base_df = self.train_df[self.base_features + [self.label]].copy()
        for feature in self.ext_features:
            if feature not in self.base_df.columns:
                self.base_df[feature] = np.nan
        self.ext_df = self.train_df[self.base_features + self.ext_features + [self.label]].copy()

    def train_all(self, n_trials=20):
        ext_train_df = self.ext_df[self.ext_df['has_extended'] == 1]

        if len(ext_train_df) == 0:
            print("\n Skipping Extended model training — no rows with extended data")
            return False

        print("\n=== Training Base Model ===")
        self.base_model = XGBoostModel(name="base_model")
        self.base_model.train(
            X=self.base_df[self.base_features + self.ext_features],
            y=self.base_df[self.label],
            n_trials=n_trials
        )
        # Save base model — needed by extended model for warm-starting
        self.base_model.save_model()
        self.ext_df = self.ext_df[self.ext_df['has_extended'] == 1]

        print("\n=== Training Extended Model (Incremental) ===")
        self.extended_model = XGBoostModel(name="extended_model")
        self.extended_model.train(
            self.ext_df[self.base_features + self.ext_features],
            self.ext_df[self.label],
            base_model_path="base_model.json",
            n_trials=n_trials
        )

        print("\n=== Training Combined Model ===")
        self.combined_model = XGBoostModel(name="combined_model")
        self.combined_model.train(
            self.train_df[self.base_features + self.ext_features],
            self.train_df[self.label],
            n_trials=n_trials
        )
        return True

    def test_all(self, csv_name=None):
        test_with_extended = self.test_df[self.test_df['has_extended'] == 1]
        test_without_extended = self.test_df[self.test_df['has_extended'] == 0]

        print('len with ext', len(test_with_extended.index))
        print('len without ext', len(test_without_extended.index))

        results = []

        def evaluate(model, data, label_name, model_name):
            y_true = data[label_name]
            y_pred_proba = model.predict(data[self.all_features])
            auc = roc_auc_score(y_true, y_pred_proba)

            y_pred_binary = (y_pred_proba >= 0.5).astype(int)
            acc = accuracy_score(y_true, y_pred_binary)
            f1 = f1_score(y_true, y_pred_binary)

            print(f"\n{model_name}")
            print(f"AUC: {auc:.4f}, Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

            results.append({
                'Model': model_name,
                'AUC': auc,
                'Accuracy': acc,
                'F1': f1
            })

            return auc

        base_auc = evaluate(self.base_model, test_without_extended, self.label, "Base model (no extended)")
        extended_auc = evaluate(self.extended_model, test_with_extended, self.label, "Extended model (with extended)")
        combined_no_ext_auc = evaluate(self.combined_model, test_without_extended, self.label, "Combined model (no extended)")
        combined_with_ext_auc = evaluate(self.combined_model, test_with_extended, self.label, "Combined model (with extended)")

        results_df = pd.DataFrame(results)
        results_df["Base_Features"] = ', '.join(self.base_features)
        results_df["Extended_Features"] = ', '.join(self.ext_features)
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.width', None)

        print("\nResults Summary:")
        print(results_df)
        return (combined_with_ext_auc - extended_auc) + \
               (combined_no_ext_auc - base_auc)

    def full_run(self, data, in_base_features, in_ext_features, label, csv_name, n_trials=20):
        self.load_data(in_base_features, in_ext_features, data, label)
        self.set_has_extended()
        a = self.train_test_split()
        if a == 999:
            return 999
        self.set_train_base_ext_datasets()
        result = self.train_all(n_trials=n_trials)
        if result == True:
            ret_val = self.test_all(csv_name)
            return ret_val
        else:
            return 999
