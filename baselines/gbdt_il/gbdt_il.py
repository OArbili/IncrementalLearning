"""GBDT-IL — Incremental Learning of Gradient Boosting Decision Trees.

Paper: Chen, Dai, Zhang, Zhu, Liu, Zhao. Sensors 2024, 24, 2083.
       https://doi.org/10.3390/s24072083

Faithful implementation of Algorithms 1 (initial GBDT-IL) and 2 (full
GBDT-IL with residual-based tree pruning and drift-triggered retrain).
Default hyper-parameters match paper Table 7.

Adaptation for our structured-missingness setting:
  * GBDT-IL assumes a *fixed* feature space. We keep that assumption by
    always using base_features ⊕ ext_features as the feature matrix,
    NaN-imputing ext columns on rows where has_extended==0 (XGBoost
    handles NaN natively; we fall back to training-set column means for
    determinism across baselines).
  * Training "stream" = the shuffled training rows (identical order to
    the AdaptiveXGBoost/PUFE/OCDS/EMLI baselines). No special ordering
    that would bias the evaluation.
  * Prediction simply uses the final pruned ensemble — no per-row split
    on has_extended.

Pseudo-code followed (paper §3.3):
    Input : stream of data windows; hyper-params
    1. Train initial GBDT with M trees on Dchunk_ini.
    2. For each new window Dchunk_slide:
         a. Fit num_inc_tree more trees via warm-start (XGBoost's
            `xgb_model=` continuation).
         b. Compute per-prefix residuals R_m on the window.
         c. I_elastic = argmin_m MA(R_m).
         d. If I_elastic < M → retrain GBDT from scratch on the window
            (drift detected); update M = num_boosted_rounds of new model.
         e. Else → truncate booster to [0..I_elastic] (prune).
"""
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np
import pandas as pd
import xgboost as xgb


@dataclass
class GBDTIL:
    """GBDT-IL classifier for binary problems.

    Parameters
    ----------
    initial_trees : int
        Number of trees in the first GBDT trained on the initial window
        (paper's M, default 250 per Table 7).
    num_inc_tree : int
        Number of trees to add per sliding window (paper's num_inc_tree=25).
    init_size : int
        Size of initial training window (paper's ini_train_size=1000).
    win_size : int
        Sliding window size (paper's win_size=500).
    max_tree : int
        Hard cap on total ensemble size (paper's max_tree=10000).
    learning_rate, max_depth, min_child_weight : XGBoost params.
    prefix_search_step : int
        Evaluate residuals every Kth prefix length (speed-up; paper tries
        all, we sample at this step which is a negligible approximation
        for large ensembles).
    seed : int
    verbose : bool
    """
    initial_trees: int = 250
    num_inc_tree: int = 25
    init_size: int = 1000
    win_size: int = 500
    max_tree: int = 10000
    learning_rate: float = 0.01
    max_depth: int = 10
    min_child_weight: int = 5
    subsample: float = 0.8
    colsample_bytree: float = 1.0
    reg_lambda: float = 1.0
    prefix_search_step: int = 5
    seed: int = 42
    verbose: bool = False

    # --- Fitted state (set during fit) ---
    booster_: Optional[xgb.Booster] = field(default=None, init=False)
    base_mean_: Optional[np.ndarray] = field(default=None, init=False)
    feature_names_: Optional[List[str]] = field(default=None, init=False)
    history_: List[dict] = field(default_factory=list, init=False)

    # ------------------------------------------------------------------
    def _xgb_params(self):
        return {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'min_child_weight': self.min_child_weight,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_lambda': self.reg_lambda,
            'seed': self.seed,
            'verbosity': 0,
        }

    def _impute(self, X):
        if self.base_mean_ is None or not np.isnan(X).any():
            return X
        idx = np.where(np.isnan(X))
        X = X.copy()
        X[idx] = np.take(self.base_mean_, idx[1])
        return X

    def _make_dmat(self, X, y=None):
        X_imp = self._impute(X)
        return xgb.DMatrix(X_imp, label=y, feature_names=self.feature_names_)

    # ------------------------------------------------------------------
    def _train_initial(self, X, y):
        dm = self._make_dmat(X, y)
        booster = xgb.train(
            self._xgb_params(), dm,
            num_boost_round=self.initial_trees,
        )
        return booster

    def _fit_incremental(self, X, y):
        """Add num_inc_tree trees to the current booster (Algorithm 1, step 4-6)."""
        dm = self._make_dmat(X, y)
        self.booster_ = xgb.train(
            self._xgb_params(), dm,
            num_boost_round=self.num_inc_tree,
            xgb_model=self.booster_,
        )

    def _find_best_prefix(self, X, y) -> int:
        """I_elastic = argmin_m MA(R_m) over tree prefixes.

        Iterates through prefix sizes in steps of self.prefix_search_step
        using XGBoost's `iteration_range` (O(steps) predict calls).
        Returns the best number of trees (≥ 1).
        """
        dm = self._make_dmat(X)
        n_trees = self.booster_.num_boosted_rounds()
        step = max(1, self.prefix_search_step)
        best_m, best_err = n_trees, float('inf')
        # Sweep prefix lengths
        ms = list(range(step, n_trees + 1, step))
        if ms[-1] != n_trees:
            ms.append(n_trees)
        if 1 not in ms:
            ms.insert(0, 1)
        for m in ms:
            pred = self.booster_.predict(dm, iteration_range=(0, m))
            err = float(np.mean(np.abs(y - pred)))
            if err < best_err:
                best_err = err
                best_m = m
        return best_m

    def _truncate_booster(self, n_keep: int):
        """Keep only the first n_keep trees of the current booster.

        XGBoost doesn't expose a native `trim` API; we save-and-reload a
        JSON with trees [n_keep:] stripped. In practice, for the volumes
        we deal with (<10K trees), this is fast (~tens of ms per call).
        """
        n_total = self.booster_.num_boosted_rounds()
        if n_keep >= n_total:
            return
        if n_keep < 1:
            n_keep = 1
        cfg = self.booster_.save_config()
        raw = self.booster_.save_raw('json')
        import json
        obj = json.loads(bytearray(raw).decode('utf-8'))
        gbt = obj['learner']['gradient_booster']
        if gbt['name'] == 'gbtree':
            model = gbt['model']
            model['trees'] = model['trees'][:n_keep]
            model['tree_info'] = model['tree_info'][:n_keep]
            model['gbtree_model_param']['num_trees'] = str(n_keep)
            # Some XGBoost versions also track number_of_parallel_tree in
            # gradient_booster — leave unchanged.
        new_raw = json.dumps(obj).encode('utf-8')
        new_booster = xgb.Booster()
        new_booster.load_model(bytearray(new_raw))
        new_booster.load_config(cfg)
        self.booster_ = new_booster

    # ------------------------------------------------------------------
    def fit(self, X, y, feature_names=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if feature_names is None:
            feature_names = [f'f{i}' for i in range(X.shape[1])]
        self.feature_names_ = list(feature_names)

        # Training-set column means for deterministic NaN imputation
        self.base_mean_ = np.nanmean(X, axis=0)
        if np.isnan(self.base_mean_).any():
            self.base_mean_ = np.nan_to_num(self.base_mean_, nan=0.0)

        n = X.shape[0]
        if n < self.init_size + 1:
            # Degenerate small-data case: just train one model on everything.
            self.booster_ = self._train_initial(X, y)
            self.history_.append({'window': 0, 'action': 'initial_only', 'trees': self.booster_.num_boosted_rounds()})
            return self

        # --- Initial window (Algorithm 1 step 2) ---
        X_ini = X[:self.init_size]; y_ini = y[:self.init_size]
        self.booster_ = self._train_initial(X_ini, y_ini)
        M_initial = self.booster_.num_boosted_rounds()
        drift_threshold = M_initial  # paper: retrain if I_elastic < M
        self.history_.append({
            'window': 0, 'action': 'initial', 'trees': M_initial,
        })

        # --- Sliding windows (Algorithm 2) ---
        win = self.win_size
        for w, start in enumerate(range(self.init_size, n, win), start=1):
            end = min(start + win, n)
            X_w = X[start:end]; y_w = y[start:end]
            if len(X_w) < 10:
                continue

            # Step: incremental fit (Algorithm 1 step 3-6 applied each window)
            self._fit_incremental(X_w, y_w)
            if self.booster_.num_boosted_rounds() > self.max_tree:
                self._truncate_booster(self.max_tree)

            # Step: find best prefix on the current window (Algorithm 2 step 3)
            i_elastic = self._find_best_prefix(X_w, y_w)

            if i_elastic < drift_threshold:
                # Drift detected → retrain (Algorithm 2 step 5)
                self.booster_ = self._train_initial(X_w, y_w)
                new_M = self.booster_.num_boosted_rounds()
                drift_threshold = new_M
                self.history_.append({
                    'window': w, 'action': 'drift_retrain',
                    'i_elastic': i_elastic, 'trees': new_M,
                })
            else:
                # Prune (Algorithm 2 step 8)
                self._truncate_booster(i_elastic)
                self.history_.append({
                    'window': w, 'action': 'prune',
                    'i_elastic': i_elastic, 'trees': self.booster_.num_boosted_rounds(),
                })

        if self.verbose:
            n_initial = 1
            n_drift = sum(1 for h in self.history_ if h['action'] == 'drift_retrain')
            n_prune = sum(1 for h in self.history_ if h['action'] == 'prune')
            final_trees = self.booster_.num_boosted_rounds()
            print(f"GBDT-IL: {n_initial} initial + {n_prune} prune + {n_drift} drift retrains; "
                  f"final trees={final_trees}")
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        dm = self._make_dmat(X)
        p1 = self.booster_.predict(dm)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
