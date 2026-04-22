"""AdaptiveXGBoostClassifier (Montiel et al. 2020, IJCNN).

Vendored verbatim from https://github.com/jacobmontiel/AdaptiveXGBoostClassifier
so the baseline does not require installing scikit-multiflow just to pull
this single class.  Only two minor changes:

 * `from skmultiflow.core import BaseSKMObject, ClassifierMixin` — we replace
   BaseSKMObject with a tiny shim so the class works without scikit-multiflow
   being installed.  The shim only needs to provide a `reset()` no-op.
 * ADWIN drift detector is imported lazily and is optional — if
   scikit-multiflow is not installed and `detect_drift=False` (our default),
   the class works fine.
"""
import numpy as np
import xgboost as xgb


class _BaseSKMObject:
    """Minimal shim for skmultiflow.core.BaseSKMObject."""
    def reset(self):
        return self


class _ClassifierMixin:
    pass


class AdaptiveXGBoostClassifier(_BaseSKMObject, _ClassifierMixin):
    """Adaptive XGBoost for Evolving Data Streams (Montiel et al. 2020)."""

    _PUSH_STRATEGY = 'push'
    _REPLACE_STRATEGY = 'replace'
    _UPDATE_STRATEGIES = [_PUSH_STRATEGY, _REPLACE_STRATEGY]

    def __init__(self,
                 n_estimators=30,
                 learning_rate=0.3,
                 max_depth=6,
                 max_window_size=1000,
                 min_window_size=None,
                 detect_drift=False,
                 update_strategy='replace'):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size
        self.detect_drift = detect_drift
        self.update_strategy = update_strategy
        self._init_margin = 0.0
        self._boosting_params = {
            "silent": True,
            "objective": "binary:logistic",
            "eta": learning_rate,
            "max_depth": max_depth,
        }
        if update_strategy not in self._UPDATE_STRATEGIES:
            raise AttributeError("Invalid update_strategy: {}".format(update_strategy))
        self._ensemble = None
        self._X_buffer = None
        self._y_buffer = None
        self._samples_seen = 0
        self._model_idx = 0
        self._first_run = True
        self._drift_detector = None
        if detect_drift:
            try:
                from skmultiflow.drift_detection.adwin import ADWIN
                self._drift_detector = ADWIN()
            except Exception as e:
                raise ImportError(
                    "detect_drift=True requires scikit-multiflow. "
                    "Install with: pip install scikit-multiflow"
                ) from e

    def reset(self):
        self._ensemble = None
        self._X_buffer = None
        self._y_buffer = None
        self._samples_seen = 0
        self._model_idx = 0
        self._first_run = True
        if self._drift_detector is not None:
            self._drift_detector.reset()
        return self

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        for i in range(X.shape[0]):
            self._partial_fit_single(X[i:i + 1], y[i:i + 1])
        return self

    def _partial_fit_single(self, X, y):
        if self._first_run:
            self._X_buffer = np.array([]).reshape(0, X.shape[1])
            self._y_buffer = np.array([])
            self._ensemble = [None] * self.n_estimators
            self._first_run = False
        self._X_buffer = np.concatenate((self._X_buffer, X))
        self._y_buffer = np.concatenate((self._y_buffer, y))
        while self._X_buffer.shape[0] >= self.window_size:
            self._train_on_mini_batch(
                X=self._X_buffer[0:self.window_size, :],
                y=self._y_buffer[0:self.window_size]
            )
            delete_idx = [i for i in range(self.window_size)]
            self._X_buffer = np.delete(self._X_buffer, delete_idx, axis=0)
            self._y_buffer = np.delete(self._y_buffer, delete_idx)
            self._adjust_window_size()
        if self.detect_drift:
            pred = self._predict_proba(X)[0][1]
            incorrect = int(y[0] != (pred > 0.5))
            self._drift_detector.add_element(incorrect)
            if self._drift_detector.detected_change():
                # On drift: reset ensemble but keep buffer
                self._ensemble = [None] * self.n_estimators
                self._model_idx = 0
                self._drift_detector.reset()

    def _adjust_window_size(self):
        if self.min_window_size is None:
            return
        if self.window_size * 2 <= self.max_window_size:
            self._dynamic_window_size = self.window_size * 2

    @property
    def window_size(self):
        if self.min_window_size is None:
            return self.max_window_size
        if not hasattr(self, "_dynamic_window_size"):
            self._dynamic_window_size = self.min_window_size
        return self._dynamic_window_size

    def _train_on_mini_batch(self, X, y):
        if self.update_strategy == self._REPLACE_STRATEGY:
            booster = self._train_booster(X, y, self._model_idx)
            self._ensemble[self._model_idx] = booster
            self._model_idx = (self._model_idx + 1) % self.n_estimators
        else:  # push
            booster = self._train_booster(X, y, self._model_idx)
            if self._model_idx < self.n_estimators:
                self._ensemble[self._model_idx] = booster
                self._model_idx += 1
            else:
                # Pop oldest, push new
                self._ensemble = self._ensemble[1:] + [booster]

    def _train_booster(self, X, y, last_model_idx):
        dtrain = xgb.DMatrix(X, label=y)
        if last_model_idx == 0 or self._ensemble[last_model_idx - 1] is None:
            booster = xgb.train(
                params=self._boosting_params,
                dtrain=dtrain,
                num_boost_round=1,
                verbose_eval=False,
            )
        else:
            margin = self._ensemble[last_model_idx - 1].predict(dtrain, output_margin=True)
            dtrain.set_base_margin(margin=margin)
            booster = xgb.train(
                params=self._boosting_params,
                dtrain=dtrain,
                num_boost_round=1,
                verbose_eval=False,
            )
        return booster

    def _predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if self._first_run or all(m is None for m in self._ensemble or []):
            return np.tile([0.5, 0.5], (X.shape[0], 1))
        dtest = xgb.DMatrix(X)
        trained = [m for m in self._ensemble if m is not None]
        margin = None
        for booster in trained:
            if margin is None:
                margin = booster.predict(dtest, output_margin=True)
            else:
                margin = margin + booster.predict(dtest, output_margin=True)
        proba_1 = 1.0 / (1.0 + np.exp(-margin))
        return np.column_stack([1 - proba_1, proba_1])

    def predict_proba(self, X):
        return self._predict_proba(X)

    def predict(self, X):
        proba = self._predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)
