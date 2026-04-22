"""EML(I) — Evolving Metric Learning for Incremental and Decremental Features.

Paper: Dong, Cong, Sun, Zhang, Tang, Xu. IEEE T-CSVT 2021.
       https://arxiv.org/abs/2006.15334

Faithful adaptation to our structured-missingness setting:

  Paper stages                        | Our setting
  ------------------------------------+-----------------------------------------
  Vanished features (decremental)     | ext_features (absent when has_extended=0)
  Survived features                   | base_features (always present)
  Augmented features (incremental)    | N/A (no new features appear at test)

Because our problem has no new/augmented features, the Inheriting stage
reduces to "pick the right metric per row":

    has_extended == 1  →  use L_a (base ⊕ ext)
    has_extended == 0  →  use L_s (base only)

Training follows the T-stage objective (paper §III.B, Eq. 4) simplified
to a clean, tractable form:

    min_{L_s, L_a, w_s, w_a}
        TripletMargin(L_s · x_base,  margin=m)            [all rows]
      + TripletMargin(L_a · [x_base; x_ext], margin=m)    [has_ext=1 rows]
      + α · MSE(L_s · x_base,  L_a · [x_base; x_ext])     [has_ext=1 rows]
      + β · ( ||L_s||²_F + ||L_a||²_F )                   (low-rank proxy)
      + BCE(w_s · L_s · x_base + b_s,  y)                 (classifier on s)
      + BCE(w_a · L_a · [x_base;x_ext] + b_a,  y)         (classifier on a)

Triplet margin loss is used as a standard substitute for the smoothed
Wasserstein distance the paper uses between class-signature batches.

Prediction: per row, use the appropriate classifier depending on has_ext.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F

# Prevent PyTorch BLAS thread-pool deadlocks on small matrices, especially
# inside tight triplet-sampling loops on macOS. Cluster jobs set
# OMP_NUM_THREADS through SLURM anyway.
if 'EMLI_TORCH_THREADS' in os.environ:
    torch.set_num_threads(int(os.environ['EMLI_TORCH_THREADS']))
else:
    torch.set_num_threads(1)


class _EMLIModel(torch.nn.Module):
    def __init__(self, d_base, d_ext, k):
        super().__init__()
        self.L_s = torch.nn.Parameter(torch.empty(k, d_base))
        self.L_a = torch.nn.Parameter(torch.empty(k, d_base + d_ext))
        torch.nn.init.kaiming_uniform_(self.L_s, a=5 ** 0.5)
        torch.nn.init.kaiming_uniform_(self.L_a, a=5 ** 0.5)
        self.w_s = torch.nn.Parameter(torch.zeros(k))
        self.b_s = torch.nn.Parameter(torch.zeros(1))
        self.w_a = torch.nn.Parameter(torch.zeros(k))
        self.b_a = torch.nn.Parameter(torch.zeros(1))
        self.d_base = d_base
        self.d_ext = d_ext

    def embed_s(self, x_base):
        return F.linear(x_base, self.L_s)

    def embed_a(self, x_full):
        return F.linear(x_full, self.L_a)

    def logit_s(self, x_base):
        return self.embed_s(x_base) @ self.w_s + self.b_s

    def logit_a(self, x_full):
        return self.embed_a(x_full) @ self.w_a + self.b_a


class EMLIClassifier:
    """EMLI for structured-missingness feature evolution.

    Parameters
    ----------
    k : int
        Output dimensionality of the learned Mahalanobis projection (low-rank).
    margin : float
        Triplet margin.
    lr : float
        Adam learning rate.
    n_epochs : int
    batch_size : int
    triplet_weight, consistency_weight, lowrank_weight : float
        Loss weights corresponding to paper's γ/α/β in Eq. 4.
    cls_weight : float
        Weight on the binary-cross-entropy classifier head.
    standardize : bool
    seed : int
    device : str or None
    """

    def __init__(self, k=16, margin=1.0, lr=1e-3, n_epochs=30, batch_size=256,
                 triplet_weight=1.0, consistency_weight=0.5, lowrank_weight=1e-3,
                 cls_weight=1.0, standardize=True, seed=42, device=None,
                 verbose=False):
        self.k = k
        self.margin = margin
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.triplet_weight = triplet_weight
        self.consistency_weight = consistency_weight
        self.lowrank_weight = lowrank_weight
        self.cls_weight = cls_weight
        self.standardize = standardize
        self.seed = seed
        self.verbose = verbose
        self.device = (device if device is not None
                        else ('cuda' if torch.cuda.is_available() else 'cpu'))

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
    def _triplets_for_batch(self, y_batch, class_pools, rng, n_attempts=4):
        """For each anchor in the batch, pick a positive (same class) and
        negative (different class) index into the full training set.

        class_pools : dict[int -> np.ndarray of row indices]
        Returns (pos_idx, neg_idx) arrays of same length as y_batch.
        None for the anchor if no valid pos/neg can be sampled.
        """
        pos = np.empty(len(y_batch), dtype=np.int64)
        neg = np.empty(len(y_batch), dtype=np.int64)
        valid = np.ones(len(y_batch), dtype=bool)
        all_classes = list(class_pools.keys())
        for i, yi in enumerate(y_batch):
            pool_pos = class_pools.get(int(yi))
            if pool_pos is None or len(pool_pos) == 0:
                valid[i] = False; continue
            pos[i] = rng.choice(pool_pos)
            # pick a different class
            other_classes = [c for c in all_classes if c != int(yi)]
            if not other_classes:
                valid[i] = False; continue
            cneg = rng.choice(other_classes)
            neg[i] = rng.choice(class_pools[cneg])
        return pos, neg, valid

    # ------------------------------------------------------------------
    def fit(self, X_base, X_ext, y, has_extended):
        torch.manual_seed(self.seed)
        rng = np.random.RandomState(self.seed)

        X_base = np.asarray(X_base, dtype=float)
        X_ext = np.asarray(X_ext, dtype=float)
        y = np.asarray(y, dtype=np.int64)
        has_ext = np.asarray(has_extended, dtype=np.int64)
        n, d_base = X_base.shape
        _, d_ext = X_ext.shape
        self.d_base_, self.d_ext_ = d_base, d_ext

        # --- Imputation + standardization (training stats) ---
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

        # Full feature matrix (base ⊕ ext; ext zeroed when has_ext==0)
        X_full = np.hstack([Xb, np.where(ext_mask[:, None], Xe, 0.0)])

        device = self.device
        Xb_t = torch.as_tensor(Xb, dtype=torch.float32, device=device)
        Xf_t = torch.as_tensor(X_full, dtype=torch.float32, device=device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=device)
        ext_t = torch.as_tensor(ext_mask, dtype=torch.bool, device=device)

        # Class pools (for triplet sampling) — over full training set
        class_pools_all = {int(c): np.where(y == c)[0] for c in np.unique(y)}
        # Class pools restricted to has_ext==1 rows (for L_a triplet sampling)
        class_pools_ext = {int(c): np.where((y == c) & ext_mask)[0]
                            for c in np.unique(y)}
        # Drop empty pools for L_a — if one class has 0 ext rows, disable L_a triplet
        ext_triplet_ok = all(len(v) > 0 for v in class_pools_ext.values())

        model = _EMLIModel(d_base, d_ext, self.k).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)

        n_batches = max(1, n // self.batch_size)
        idx_all = np.arange(n)
        idx_ext_only = np.where(ext_mask)[0]

        for epoch in range(self.n_epochs):
            perm = rng.permutation(n)
            epoch_losses = []
            for b in range(n_batches):
                batch_idx = perm[b * self.batch_size:(b + 1) * self.batch_size]
                if len(batch_idx) == 0:
                    continue

                yb = y[batch_idx]
                xb = Xb_t[batch_idx]
                xf = Xf_t[batch_idx]
                yb_t = y_t[batch_idx]
                ext_b = ext_t[batch_idx]

                # Embeddings
                zs = model.embed_s(xb)              # (B, k)
                za = model.embed_a(xf)              # (B, k)

                # --- Classifier BCE on both heads ---
                logit_s = zs @ model.w_s + model.b_s
                logit_a = za @ model.w_a + model.b_a
                bce_s = F.binary_cross_entropy_with_logits(logit_s, yb_t)
                if ext_b.any():
                    bce_a = F.binary_cross_entropy_with_logits(
                        logit_a[ext_b], yb_t[ext_b])
                else:
                    bce_a = torch.tensor(0.0, device=device)
                cls_loss = bce_s + bce_a

                # --- Triplet loss on L_s (all rows) ---
                pos_idx_s, neg_idx_s, valid_s = self._triplets_for_batch(
                    yb, class_pools_all, rng)
                if valid_s.any():
                    sel = np.where(valid_s)[0]
                    zs_pos = model.embed_s(Xb_t[pos_idx_s[sel]])
                    zs_neg = model.embed_s(Xb_t[neg_idx_s[sel]])
                    trip_s = F.triplet_margin_loss(
                        zs[sel], zs_pos, zs_neg, margin=self.margin)
                else:
                    trip_s = torch.tensor(0.0, device=device)

                # --- Triplet loss on L_a (has_ext rows only) ---
                if ext_triplet_ok and ext_b.any():
                    ext_local = torch.where(ext_b)[0].cpu().numpy()
                    yb_ext = yb[ext_local]
                    pos_idx_a, neg_idx_a, valid_a = self._triplets_for_batch(
                        yb_ext, class_pools_ext, rng)
                    if valid_a.any():
                        sel_a = np.where(valid_a)[0]
                        za_pos = model.embed_a(Xf_t[pos_idx_a[sel_a]])
                        za_neg = model.embed_a(Xf_t[neg_idx_a[sel_a]])
                        trip_a = F.triplet_margin_loss(
                            za[ext_local[sel_a]], za_pos, za_neg,
                            margin=self.margin)
                    else:
                        trip_a = torch.tensor(0.0, device=device)
                else:
                    trip_a = torch.tensor(0.0, device=device)

                triplet_loss = trip_s + trip_a

                # --- Consistency loss (has_ext rows) ---
                if ext_b.any():
                    consist_loss = F.mse_loss(zs[ext_b], za[ext_b])
                else:
                    consist_loss = torch.tensor(0.0, device=device)

                # --- Low-rank / Frobenius regulariser ---
                lr_loss = (model.L_s.pow(2).sum() + model.L_a.pow(2).sum())

                loss = (self.cls_weight * cls_loss
                        + self.triplet_weight * triplet_loss
                        + self.consistency_weight * consist_loss
                        + self.lowrank_weight * lr_loss)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                opt.step()
                epoch_losses.append(loss.item())

            if self.verbose and (epoch == 0 or (epoch + 1) % 5 == 0):
                print(f"  EMLI epoch {epoch+1:>3}/{self.n_epochs}  "
                      f"mean loss={np.mean(epoch_losses):.4f}")

        self.model_ = model
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X_base, X_ext, has_extended):
        X_base = np.asarray(X_base, dtype=float)
        X_ext = np.asarray(X_ext, dtype=float)
        has_ext = np.asarray(has_extended, dtype=int)

        Xb = self._impute(X_base, self.base_mean_)
        Xe = np.where(np.isnan(X_ext), self.ext_mean_, X_ext)
        if self.standardize:
            Xb = self._standardize(Xb, self.base_std_mean_, self.base_std_std_)
            Xe = self._standardize(Xe, self.ext_std_mean_, self.ext_std_std_)
        X_full = np.hstack([Xb, np.where(has_ext[:, None] == 1, Xe, 0.0)])

        device = self.device
        Xb_t = torch.as_tensor(Xb, dtype=torch.float32, device=device)
        Xf_t = torch.as_tensor(X_full, dtype=torch.float32, device=device)

        self.model_.eval()
        with torch.no_grad():
            logit_s = self.model_.logit_s(Xb_t).cpu().numpy()
            logit_a = self.model_.logit_a(Xf_t).cpu().numpy()
        # Pick the right head per row
        logit = np.where(has_ext == 1, logit_a, logit_s)
        p1 = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
        return np.column_stack([1 - p1, p1])

    def predict(self, X_base, X_ext, has_extended):
        return (self.predict_proba(X_base, X_ext, has_extended)[:, 1] > 0.5).astype(int)
