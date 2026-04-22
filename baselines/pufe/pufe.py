"""PUFE — Prediction with Unpredictable Feature Evolution.

Implementation of Hou, Zhang, Zhou (IEEE TNNLS 2021, arXiv:1904.12171).
Paper: https://arxiv.org/abs/1904.12171

Adaptation for our structured-missingness setting:

  Paper setting                      |   Our setting
  -----------------------------------+-------------------------------------
  Previous feature space P (d1)      |   base_features (always observed)
  Current feature space C (d2)       |   base_features ∪ ext_features
  Overlap period (both observed)     |   rows with has_extended == 1
  Post-overlap (only C)              |   N/A (base is always observed)
  Matrix A (intact P before overlap) |   rows with has_extended == 0
  Matrix M (incomplete P in overlap) |   not incomplete in our setting
  Matrix N (intact C in overlap)     |   ext_features on has_extended == 1

PUFE's matrix-completion step becomes trivial here (ext features are
either fully observed or fully missing per row). It is replaced by a
linear mapping ψ: base → ext learned via least squares on the overlap
period, exactly as the paper's Eq. (5):

    P1 = Σ x_C x_C^T ; P2 = Σ x_C x_P^T ; ψ = P1⁻¹ · P2

At test time we maintain two base predictors and Hedge-ensemble them:

    * w_P   : OGD logistic classifier trained on base_features
              using rows with has_extended == 0 (Algorithm 2).
    * w_C   : OGD logistic classifier trained on (base, ext) features
              using rows with has_extended == 1.
    * ψ     : linear map recovering ext from base (used when has_ext == 0).

    For has_extended == 1:
        p = α·σ(w_C · [x_base, x_ext]) + (1-α)·σ(w_P · x_base)
    For has_extended == 0:
        p = α·σ(w_C · [x_base, ψ(x_base)]) + (1-α)·σ(w_P · x_base)

    α comes from Hedge weights computed on the training stream per
    Algorithm 4: α = exp(-η·L_C) / (exp(-η·L_P)+exp(-η·L_C))
    where L_* are cumulative log losses on the training rows.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class _OGDLogistic:
    """Online gradient descent for logistic regression (one-pass)."""
    n_features: int
    lr: float = 0.1  # step size scaled as lr/sqrt(t) per the paper
    w: np.ndarray = field(init=False)
    b: float = field(init=False, default=0.0)
    t: int = field(init=False, default=0)

    def __post_init__(self):
        self.w = np.zeros(self.n_features, dtype=float)

    def _sigmoid(self, z):
        z = np.clip(z, -30, 30)
        return 1.0 / (1.0 + np.exp(-z))

    def partial_fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        for i in range(X.shape[0]):
            self.t += 1
            eta = self.lr / np.sqrt(self.t)
            z = self.w @ X[i] + self.b
            p = self._sigmoid(z)
            grad_w = (p - y[i]) * X[i]
            grad_b = (p - y[i])
            self.w -= eta * grad_w
            self.b -= eta * grad_b
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        p1 = self._sigmoid(X @ self.w + self.b)
        return np.column_stack([1 - p1, p1])


class PUFEClassifier:
    """PUFE baseline adapted to structured feature missingness.

    Parameters
    ----------
    lr : float
        OGD learning rate (scaled internally as lr/sqrt(t)).
    hedge_eta : float
        Hedge exponential-weights learning rate.
    n_passes : int
        Number of passes over the training stream. The paper is one-pass;
        we allow >1 for small datasets where a single pass underfits.
    ridge : float
        Ridge regularisation for the linear mapping ψ.
    standardize : bool
        If True, standardize features (zero mean, unit variance) using
        training-set statistics. Matches the common LR pre-processing and
        is standard practice for OGD logistic models.
    """

    def __init__(self, lr=0.5, hedge_eta=1.0, n_passes=3,
                 ridge=1e-3, standardize=True, verbose=False):
        self.lr = lr
        self.hedge_eta = hedge_eta
        self.n_passes = n_passes
        self.ridge = ridge
        self.standardize = standardize
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _impute(self, X, mean):
        X = np.asarray(X, dtype=float)
        if np.isnan(X).any():
            idx = np.where(np.isnan(X))
            X = X.copy()
            X[idx] = np.take(mean, idx[1])
        return X

    def _standardize(self, X, mean, std):
        return (X - mean) / np.where(std > 1e-9, std, 1.0)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X_base, X_ext, y, has_extended):
        """Train PUFE on arrays aligned row-wise.

        X_base : (n, d_base)   base features (may contain NaN)
        X_ext  : (n, d_ext)    extended features (NaN where has_extended==0)
        y      : (n,)          binary labels 0/1
        has_extended : (n,)    0 / 1 indicator
        """
        X_base = np.asarray(X_base, dtype=float)
        X_ext = np.asarray(X_ext, dtype=float)
        y = np.asarray(y, dtype=float)
        has_ext = np.asarray(has_extended, dtype=int)
        n, d_base = X_base.shape
        _, d_ext = X_ext.shape

        ext_mask = has_ext == 1
        noext_mask = has_ext == 0

        # --- Imputation statistics from training data ---
        # Base features: imputed using column means of ALL training rows
        self.base_mean_ = np.nanmean(X_base, axis=0)
        Xb = self._impute(X_base, self.base_mean_)
        # Ext features: imputed using column means of rows where observed
        if ext_mask.sum() > 0:
            self.ext_mean_ = np.nanmean(X_ext[ext_mask], axis=0)
            if np.isnan(self.ext_mean_).any():
                self.ext_mean_ = np.nan_to_num(self.ext_mean_, nan=0.0)
        else:
            self.ext_mean_ = np.zeros(d_ext)
        Xe = np.where(np.isnan(X_ext), self.ext_mean_, X_ext)

        # --- Standardization (train-set stats) ---
        if self.standardize:
            self.base_std_mean_ = Xb.mean(axis=0)
            self.base_std_std_ = Xb.std(axis=0)
            Xb = self._standardize(Xb, self.base_std_mean_, self.base_std_std_)
            if ext_mask.sum() > 0:
                self.ext_std_mean_ = Xe[ext_mask].mean(axis=0)
                self.ext_std_std_ = Xe[ext_mask].std(axis=0)
            else:
                self.ext_std_mean_ = np.zeros(d_ext)
                self.ext_std_std_ = np.ones(d_ext)
            Xe = self._standardize(Xe, self.ext_std_mean_, self.ext_std_std_)

        # --- Linear mapping ψ: base → ext (paper Eq. 5) ---
        # Uses only overlap rows (has_ext == 1).
        if ext_mask.sum() > 0:
            Xb_overlap = Xb[ext_mask]
            Xe_overlap = Xe[ext_mask]
            # Ridge-regularised LS: ψ = (X_b^T X_b + λI)^-1 X_b^T X_e
            P1 = Xb_overlap.T @ Xb_overlap + self.ridge * np.eye(d_base)
            P2 = Xb_overlap.T @ Xe_overlap
            self.psi_ = np.linalg.solve(P1, P2)  # shape (d_base, d_ext)
        else:
            self.psi_ = np.zeros((d_base, d_ext))

        # --- Feature construction helpers (full = base ⊕ ext) ---
        def full_features_from_rows(Xb_rows, Xe_rows, has_ext_rows):
            """Return base⊕ext for each row; recover ext via ψ when missing."""
            out = np.empty((Xb_rows.shape[0], d_base + d_ext), dtype=float)
            out[:, :d_base] = Xb_rows
            for i in range(Xb_rows.shape[0]):
                if has_ext_rows[i] == 1:
                    out[i, d_base:] = Xe_rows[i]
                else:
                    out[i, d_base:] = Xb_rows[i] @ self.psi_
            return out

        # --- Train w_P (OGD logistic on base features) ---
        # Paper's Alg. 2 trains on the "previous feature space" (P-only rows).
        # In our setting all rows have base features, so we train on ALL rows
        # to match the spirit (w_P should be a strong P-only predictor).
        self.w_P_ = _OGDLogistic(n_features=d_base, lr=self.lr)
        for _ in range(self.n_passes):
            perm = np.random.permutation(n)
            self.w_P_.partial_fit(Xb[perm], y[perm])

        # --- Train w_C (OGD logistic on base⊕ext, using overlap rows) ---
        self.w_C_ = _OGDLogistic(n_features=d_base + d_ext, lr=self.lr)
        if ext_mask.sum() > 0:
            Xf_ov = full_features_from_rows(Xb[ext_mask], Xe[ext_mask],
                                             has_ext[ext_mask])
            y_ov = y[ext_mask]
            for _ in range(self.n_passes):
                perm = np.random.permutation(len(y_ov))
                self.w_C_.partial_fit(Xf_ov[perm], y_ov[perm])

        # --- Hedge weights via streaming prediction on the training set ---
        # (Alg 4 in the paper.) We compute cumulative log-losses of both
        # base predictors and set α = softmax(-η·L).
        L_P = 0.0
        L_C = 0.0
        # One streaming pass in shuffled order (paper is one-pass)
        perm = np.random.permutation(n)
        for i in perm:
            xb_row = Xb[i]
            p_P = self.w_P_.predict_proba(xb_row.reshape(1, -1))[0, 1]
            if has_ext[i] == 1:
                xf = np.concatenate([xb_row, Xe[i]])
            else:
                xf = np.concatenate([xb_row, xb_row @ self.psi_])
            p_C = self.w_C_.predict_proba(xf.reshape(1, -1))[0, 1]
            yt = y[i]
            eps = 1e-12
            L_P += -(yt * np.log(p_P + eps) + (1 - yt) * np.log(1 - p_P + eps))
            L_C += -(yt * np.log(p_C + eps) + (1 - yt) * np.log(1 - p_C + eps))

        # Use mean per-round log-loss so η stays in a reasonable range
        # regardless of dataset size. Equivalent to cumulative Hedge with
        # η_effective = self.hedge_eta / n (a standard per-round tuning).
        mean_losses = np.array([L_P / max(n, 1), L_C / max(n, 1)])
        logits = -self.hedge_eta * mean_losses
        logits -= logits.max()
        w = np.exp(logits)
        self.hedge_weights_ = w / w.sum()  # [α_P, α_C]

        if self.verbose:
            print(f"PUFE: d_base={d_base}, d_ext={d_ext}, "
                  f"overlap={ext_mask.sum()}/{n}, ridge={self.ridge}")
            print(f"  Cumulative log-loss  L_P={L_P:.2f}  L_C={L_C:.2f}")
            print(f"  Hedge weights        α_P={self.hedge_weights_[0]:.4f}  "
                  f"α_C={self.hedge_weights_[1]:.4f}")
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_proba(self, X_base, X_ext, has_extended):
        X_base = np.asarray(X_base, dtype=float)
        X_ext = np.asarray(X_ext, dtype=float)
        has_ext = np.asarray(has_extended, dtype=int)
        n, d_base = X_base.shape
        _, d_ext = X_ext.shape

        Xb = self._impute(X_base, self.base_mean_)
        Xe = np.where(np.isnan(X_ext), self.ext_mean_, X_ext)
        if self.standardize:
            Xb = self._standardize(Xb, self.base_std_mean_, self.base_std_std_)
            Xe = self._standardize(Xe, self.ext_std_mean_, self.ext_std_std_)

        # Base-only predictions
        p_P = self.w_P_.predict_proba(Xb)[:, 1]

        # Full-feature predictions
        Xf = np.empty((n, d_base + d_ext), dtype=float)
        Xf[:, :d_base] = Xb
        for i in range(n):
            if has_ext[i] == 1:
                Xf[i, d_base:] = Xe[i]
            else:
                Xf[i, d_base:] = Xb[i] @ self.psi_
        p_C = self.w_C_.predict_proba(Xf)[:, 1]

        alpha_P, alpha_C = self.hedge_weights_
        p = alpha_P * p_P + alpha_C * p_C
        return np.column_stack([1 - p, p])

    def predict(self, X_base, X_ext, has_extended):
        return (self.predict_proba(X_base, X_ext, has_extended)[:, 1] > 0.5).astype(int)
