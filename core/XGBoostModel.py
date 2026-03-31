import numpy as np
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import optuna
from sklearn.model_selection import StratifiedKFold
from .seed_utils import SEED, DEVICE

class XGBoostModel:
    def __init__(self, name="xgb_model"):
        self.name = name
        self.model = None
        self.best_params = None
        self.used_base_model = True
        self.seed = SEED

    def objective(self, trial, X, y, base_model_path=None, n_splits=3, pruning_mode='optuna'):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
            'gamma': trial.suggest_float('gamma', 1e-3, 5.0, log=True),
            'lambda': trial.suggest_float('lambda', 1e-2, 10.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 7.0),
            "random_state": self.seed,
            "eval_metric": "auc",
            'tree_method': 'hist',
            'device': DEVICE
        }

        n_trees_keep = None
        use_base_model = True

        if base_model_path is not None:
            base_booster = xgb.Booster()
            base_booster.load_model(base_model_path)
            total_trees_in_base = len(base_booster.get_dump())

            if pruning_mode == 'optuna':
                # Optuna chooses a pruning strategy: flexible tree count, no pruning, fixed 50%, or no base model
                strategy = trial.suggest_categorical('pruning_strategy', ['flexible', 'no_pruning', 'fixed_50', 'no_base'])
                if strategy == 'no_base':
                    use_base_model = False
                elif strategy == 'no_pruning':
                    use_base_model = True
                    n_trees_keep = total_trees_in_base
                elif strategy == 'fixed_50':
                    use_base_model = True
                    n_trees_keep = max(1, total_trees_in_base // 2)
                else:  # flexible
                    use_base_model = True
                    n_trees_keep = trial.suggest_int("n_trees_keep", 1, total_trees_in_base)
            elif pruning_mode == 'no_pruning':
                # Always use full base model, no pruning
                use_base_model = True
                n_trees_keep = total_trees_in_base
            elif pruning_mode == 'fixed_50':
                # Always use base model, keep 50% of trees
                use_base_model = True
                n_trees_keep = max(1, total_trees_in_base // 2)

        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        auc_scores = []

        for train_index, valid_index in kf.split(X, y):
            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

            if base_model_path is not None and use_base_model and n_trees_keep > 0:
                fold_booster = xgb.Booster()
                fold_booster.load_model(base_model_path)
                fold_booster = fold_booster[:n_trees_keep]

                model = xgb.XGBClassifier(**params)
                model.fit(X_train, y_train,
                          xgb_model=fold_booster,
                          eval_set=[(X_valid, y_valid)],
                          verbose=False)
            else:
                model = xgb.XGBClassifier(**params)
                model.fit(X_train, y_train,
                          eval_set=[(X_valid, y_valid)],
                          verbose=False)

            y_pred_proba = model.predict_proba(X_valid)[:, 1]
            auc_val = roc_auc_score(y_valid, y_pred_proba)
            auc_scores.append(auc_val)

        mean_auc = np.mean(auc_scores)
        return mean_auc

    def train(self, X, y, n_trials=20, base_model_path=None, pruning_mode='optuna'):
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.seed))

        # When using optuna pruning mode with a base model, enqueue trials
        # for no_pruning and fixed_50 strategies to guarantee they are tested.
        if pruning_mode == 'optuna' and base_model_path is not None:
            study.enqueue_trial({"pruning_strategy": "no_pruning"})
            study.enqueue_trial({"pruning_strategy": "fixed_50"})

        def _callback(study, trial):
            print(f"  Trial {trial.number+1}/{n_trials}: AUC={trial.value:.4f}", flush=True)

        study.optimize(
            lambda trial: self.objective(trial, X, y, base_model_path, pruning_mode=pruning_mode),
            n_trials=n_trials,
            callbacks=[_callback]
        )

        self.study = study
        self.best_params = study.best_params
        print(f"Best parameters found: {self.best_params}")
        print(f"Best CV AUC score: {study.best_value:.4f}")

        best_params_clean = {k: v for k, v in self.best_params.items()
                           if k not in ['n_trees_keep', 'use_base_model', 'pruning_strategy']}
        # Add fixed params not tuned by Optuna
        best_params_clean['tree_method'] = 'hist'
        best_params_clean['device'] = DEVICE
        best_params_clean['eval_metric'] = 'auc'
        best_params_clean['random_state'] = self.seed

        if pruning_mode == 'optuna':
            strategy = self.best_params.get('pruning_strategy', 'flexible')
            if strategy == 'no_base':
                use_base_model = False
                self.used_base_model = False
                n_trees_keep = 0
            elif strategy == 'no_pruning':
                use_base_model = True
                self.used_base_model = True
                if base_model_path is not None:
                    base_booster = xgb.Booster()
                    base_booster.load_model(base_model_path)
                    n_trees_keep = len(base_booster.get_dump())
                else:
                    n_trees_keep = 0
            elif strategy == 'fixed_50':
                use_base_model = True
                self.used_base_model = True
                if base_model_path is not None:
                    base_booster = xgb.Booster()
                    base_booster.load_model(base_model_path)
                    n_trees_keep = max(1, len(base_booster.get_dump()) // 2)
                else:
                    n_trees_keep = 0
            else:  # flexible
                use_base_model = True
                self.used_base_model = True
                n_trees_keep = self.best_params.get('n_trees_keep', 0)
        elif pruning_mode == 'no_pruning':
            # Always use full base model
            use_base_model = True
            self.used_base_model = True
            if base_model_path is not None:
                base_booster = xgb.Booster()
                base_booster.load_model(base_model_path)
                n_trees_keep = len(base_booster.get_dump())
            else:
                n_trees_keep = 0
        elif pruning_mode == 'fixed_50':
            # Always use base model, keep 50% of trees
            use_base_model = True
            self.used_base_model = True
            if base_model_path is not None:
                base_booster = xgb.Booster()
                base_booster.load_model(base_model_path)
                n_trees_keep = max(1, len(base_booster.get_dump()) // 2)
            else:
                n_trees_keep = 0

        if base_model_path is not None and use_base_model and n_trees_keep > 0:
            print(f"Using base model with {n_trees_keep} trees kept (mode: {pruning_mode})")
            pruned_model = xgb.Booster()
            pruned_model.load_model(base_model_path)
            pruned_model = pruned_model[:n_trees_keep]

            self.model = xgb.XGBClassifier(**best_params_clean)
            self.model.fit(X, y, xgb_model=pruned_model)
        else:
            if base_model_path is not None:
                print("FALLBACK ACTIVATED: Training from scratch due to potential concept drift")

            self.model = xgb.XGBClassifier(**best_params_clean)
            self.model.fit(X, y)

        # Print feature importance
        importance = self.model.feature_importances_
        feature_importance = []
        for idx, imp in enumerate(importance):
            feature_name = X.columns[idx] if hasattr(X, 'columns') else f"Feature {idx}"
            feature_importance.append((feature_name, imp))

        sorted_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
        print("\nTop 10 features by importance:")
        for name, imp in sorted_importance[:10]:
            print(f"{name}: {imp:.4f}")

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)[:, 1]

    def save_model(self):
        if self.model is None:
            raise ValueError("Model not trained yet")
        self.model.save_model(f"{self.name}.json")
        metadata = {
            'used_base_model': self.used_base_model,
            'best_params': self.best_params
        }
        import json
        with open(f"{self.name}_metadata.json", 'w') as f:
            json.dump(metadata, f)

    @staticmethod
    def load_model(model_path, metadata_path=None):
        model = xgb.Booster()
        model.load_model(model_path)

        if metadata_path:
            import json
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                print(f"Model metadata: {metadata}")
            except FileNotFoundError:
                print("Metadata file not found")

        return model
