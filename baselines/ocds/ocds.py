"""OCDS — Online Capricious Data Streams (He, Wu, Wu, Beyazit, Chen, Wu).

Paper: IJCAI 2019 — https://www.ijcai.org/proceedings/2019/346

Faithful adaptation of Algorithm 1+2 to our structured-missingness setting:

  Universal space U = base_features ∪ ext_features  (dim D = d_base + d_ext)

  At each streaming iteration t, observe a row x_t whose features
  fall into an arbitrary subset of U — in our data:
    * has_extended == 1  ⇒ all D features observed
    * has_extended == 0  ⇒ only the d_base base features observed

  Graph   G ∈ R^{D×D}   encodes pairwise feature relatednesses.
  Reconstruction ψ(x_t) = (1/d_t) · G_r^T · x_full_zero_padded,
  where  G_r = I_t · G   (shape  d_t × D ),  I_t is the  d_t × D
  indicator matrix selecting observed rows of G.

  Equivalently ψ(x_t)[j] = (1/d_t) · Σ_{i∈obs} G[i, j] · x_t[i].

  Joint bi-convex objective (Eq. 8 of the paper):
      F = (y − w^T ψ)²  +  γ · ||x_obs − R_obs(ψ)||²
        + λ₁ ||w||₁  (+ λ₂ Tr(w^T L w) — we omit the Laplacian)

  Updates (Eqs. 9 / 10) via coordinate gradient descent with
      η_t = sqrt(1/t),
  then soft-threshold on w for the ℓ₁ term.

  Ensemble prediction (Eqs. 11–12):
      ŷ = p · ⟨w_obs, x_obs⟩ + (1 − p) · ⟨w_rec, x̃_rec⟩
      p = exp(-η_h · L_obs̄) / (exp(-η_h · L_obs̄) + exp(-η_h · L_rec̄))
  where L̄ are training-stream cumulative losses (per-round mean scaled for
  numerical stability, same trick as our PUFE baseline).

  Label convention: the paper uses y ∈ {-1, +1} with square loss. We
  convert our {0, 1} labels accordingly and use the raw score ŷ as the
  ranking signal for AUC computation.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class OCDSClassifier:
    gamma: float = 0.5            # scale between class loss and reconstruction loss
    lam1: float = 1e-3            # ℓ₁ regularizer on w
    n_passes: int = 2             # passes over the training stream
    hedge_eta: float = 5.0        # Hedge learning rate (scaled per-round, cf. PUFE)
    ridge_G: float = 1e-3         # ridge ψ-initialisation of G (LS over overlap rows)
    standardize: bool = True
    seed: int = 42
    verbose: bool = False

    # ------------------------------------------------------------------
    def _impute(self, X, col_mean):
        if not np.isnan(X).any():
            return X
        idx = np.where(np.isnan(X))
        X = X.copy()
        X[idx] = np.take(col_mean, idx[1])
        return X

    def _standardize(self, X, mean, std):
        return (X - mean) / np.where(std > 1e-9, std, 1.0)

    # ------------------------------------------------------------------
    def fit(self, X_base, X_ext, y, has_extended):
        """Train OCDS on aligned arrays.

        X_base : (n, d_base)   base features (may contain NaN)
        X_ext  : (n, d_ext)    ext features (NaN where has_extended == 0)
        y      : (n,)          binary labels {0, 1}
        has_extended : (n,)    indicator 0/1
        """
        rng = np.random.RandomState(self.seed)

        X_base = np.asarray(X_base, dtype=float)
        X_ext = np.asarray(X_ext, dtype=float)
        y = np.asarray(y, dtype=float)
        has_ext = np.asarray(has_extended, dtype=int)
        n, d_base = X_base.shape
        _, d_ext = X_ext.shape
        D = d_base + d_ext

        self.d_base_ = d_base
        self.d_ext_ = d_ext

        # --- Imputation + standardisation (training-set stats) ---
        self.base_mean_ = np.nanmean(X_base, axis=0)
        Xb = self._impute(X_base, self.base_mean_)

        ext_mask = has_ext == 1
        if ext_mask.sum() > 0:
            self.ext_mean_ = np.nanmean(X_ext[ext_mask], axis=0)
            if np.isnan(self.ext_mean_).any():
                self.ext_mean_ = np.nan_to_num(self.ext_mean_, nan=0.0)
        else:
            self.ext_mean_ = np.zeros(d_ext)
        Xe = np.where(np.isnan(X_ext), self.ext_mean_, X_ext)

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

        # Universal feature matrix: concat base + ext.
        # For has_extended==0 rows we zero-fill ext (they are "unobserved"
        # during the streaming step; the reconstruction will replace them).
        X_full = np.hstack([Xb, np.where(ext_mask[:, None], Xe, 0.0)])
        y_pm = 2.0 * y - 1.0  # {-1, +1}

        # --- Warm-start G via ridge LS on overlap rows ---
        # G[:d_base, d_base:]  ← base → ext linear map
        # G[d_base:, :d_base]  ← ext  → base linear map
        # Diagonal and intra-block initialised to identity.
        G = np.eye(D)
        if ext_mask.sum() > 0:
            Xb_ov = Xb[ext_mask]
            Xe_ov = Xe[ext_mask]
            # base → ext
            P1 = Xb_ov.T @ Xb_ov + self.ridge_G * np.eye(d_base)
            P2 = Xb_ov.T @ Xe_ov
            G[:d_base, d_base:] = np.linalg.solve(P1, P2)
            # ext → base
            P1e = Xe_ov.T @ Xe_ov + self.ridge_G * np.eye(d_ext)
            P2e = Xe_ov.T @ Xb_ov
            G[d_base:, :d_base] = np.linalg.solve(P1e, P2e)

        w = np.zeros(D, dtype=float)
        t = 0
        L_obs = 0.0    # ⟨w_obs, x_obs⟩
        L_rec = 0.0    # ⟨w_rec, x̃_rec⟩
        n_rec_updates = 0

        for _pass in range(self.n_passes):
            order = rng.permutation(n)
            for i in order:
                t += 1
                eta = 1.0 / np.sqrt(t)
                x_full = X_full[i]
                is_full = bool(has_ext[i] == 1)
                d_obs = D if is_full else d_base
                obs_mask = np.ones(D, dtype=bool) if is_full else np.arange(D) < d_base

                # ψ = (1/d_obs) · G_r^T · x_obs_padded
                # Equivalent to picking the observed rows of G and weighting by x.
                # Compute ψ ∈ R^D
                G_obs = G[obs_mask]                    # (d_obs, D)
                x_obs = x_full[obs_mask]               # (d_obs,)
                psi = (1.0 / d_obs) * (G_obs.T @ x_obs)  # (D,)

                # Residuals
                pred = w @ psi
                r_cls = y_pm[i] - pred                 # scalar
                # Reconstruction residual on observed dims: x_obs − ψ_obs
                r_rec = x_obs - psi[obs_mask]         # (d_obs,)

                # --- Gradients ---
                grad_w = -2.0 * r_cls * psi
                # (Eq. 10) gradient wrt G:
                #   d/dG (y − w^T ψ)^2   = (−2/d_obs) r_cls · I_t^T x_obs · w^T
                #   d/dG γ ‖x_obs − R ψ‖² = (−2γ/d_obs) · I_t^T x_obs · (r_rec)_padded^T · I_t
                grad_G = np.zeros_like(G)
                grad_G[obs_mask, :] += (-2.0 / d_obs) * np.outer(x_obs, w)
                # Reconstruction grad — only updates columns that are observed.
                grad_G[np.ix_(obs_mask, obs_mask)] += \
                    (-2.0 * self.gamma / d_obs) * np.outer(x_obs, r_rec)

                # --- Update ---
                w -= eta * grad_w
                # ℓ₁ proximal / soft-threshold step for w
                if self.lam1 > 0:
                    thresh = eta * self.lam1
                    w = np.sign(w) * np.maximum(np.abs(w) - thresh, 0.0)
                G -= eta * grad_G

                # --- Hedge accumulators ---
                # w_obs is the restriction of w to observed dims; w_rec to the rest.
                # On fully-observed rows there is no "rec" term — skip rec update.
                w_obs = w[obs_mask]
                x_obs_vec = x_obs
                p_obs = w_obs @ x_obs_vec
                L_obs += (y_pm[i] - p_obs) ** 2
                if not is_full:
                    w_rec = w[~obs_mask]
                    x_rec = psi[~obs_mask]
                    p_rec = w_rec @ x_rec
                    L_rec += (y_pm[i] - p_rec) ** 2
                    n_rec_updates += 1

        n_total = n * self.n_passes
        mean_obs = L_obs / max(n_total, 1)
        mean_rec = L_rec / max(n_rec_updates, 1) if n_rec_updates > 0 else float('inf')
        logits = -self.hedge_eta * np.array([mean_obs, mean_rec])
        logits -= logits.max()
        ew = np.exp(logits)
        self.hedge_p_ = ew[0] / ew.sum()  # weight on observed predictor
        self.w_ = w
        self.G_ = G

        if self.verbose:
            print(f"OCDS: D={D} (base={d_base} ext={d_ext}), "
                  f"overlap={int(ext_mask.sum())}/{n}, passes={self.n_passes}")
            print(f"  mean L_obs={mean_obs:.4f}  mean L_rec={mean_rec:.4f}")
            print(f"  Hedge p (weight on observed)={self.hedge_p_:.4f}")
            print(f"  w nonzero fraction={np.mean(np.abs(w) > 1e-12):.2f}")
        return self

    # ------------------------------------------------------------------
    def decision_function(self, X_base, X_ext, has_extended):
        X_base = np.asarray(X_base, dtype=float)
        X_ext = np.asarray(X_ext, dtype=float)
        has_ext = np.asarray(has_extended, dtype=int)
        n = len(X_base)

        Xb = self._impute(X_base, self.base_mean_)
        Xe = np.where(np.isnan(X_ext), self.ext_mean_, X_ext)
        if self.standardize:
            Xb = self._standardize(Xb, self.base_std_mean_, self.base_std_std_)
            Xe = self._standardize(Xe, self.ext_std_mean_, self.ext_std_std_)

        d_base = self.d_base_
        d_ext = self.d_ext_
        D = d_base + d_ext
        w = self.w_
        G = self.G_
        p_h = self.hedge_p_

        scores = np.zeros(n, dtype=float)
        for i in range(n):
            is_full = bool(has_ext[i] == 1)
            if is_full:
                x_obs = np.concatenate([Xb[i], Xe[i]])
                d_obs = D
                obs_mask = np.ones(D, dtype=bool)
            else:
                x_obs = Xb[i]
                d_obs = d_base
                obs_mask = np.arange(D) < d_base

            # ψ using current G
            G_obs = G[obs_mask]
            psi = (1.0 / d_obs) * (G_obs.T @ x_obs)

            s_obs = w[obs_mask] @ x_obs
            if is_full:
                scores[i] = s_obs
            else:
                s_rec = w[~obs_mask] @ psi[~obs_mask]
                scores[i] = p_h * s_obs + (1.0 - p_h) * s_rec
        return scores

    def predict_proba(self, X_base, X_ext, has_extended):
        s = self.decision_function(X_base, X_ext, has_extended)
        # Sigmoid squashing of square-loss score — preserves ranking.
        s = np.clip(s, -30, 30)
        p1 = 1.0 / (1.0 + np.exp(-s))
        return np.column_stack([1 - p1, p1])

    def predict(self, X_base, X_ext, has_extended):
        return (self.predict_proba(X_base, X_ext, has_extended)[:, 1] > 0.5).astype(int)
