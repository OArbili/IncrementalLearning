"""Run PUFE baseline on all 10 datasets.

Uses EXACTLY the same preprocessing, features (base vs extended), and
stratified train/test split (label × has_extended, seed=42, test=20%)
as our incremental framework — via scripts/prepare_datasets.py.

Output: one AUC row per (dataset, config) appended to
    baselines/results/pufe_results_<DS>.csv
"""
import os
import sys
import time
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from scripts.prepare_datasets import DATASET_LOADERS, split_train_test
from baselines.pufe.pufe import PUFEClassifier

SEED = 42


def _to_matrix(df, features):
    """Convert DataFrame slice to float matrix (categoricals → codes, NaNs kept)."""
    X = df[features].copy()
    for c in X.columns:
        if X[c].dtype == 'object' or str(X[c].dtype).startswith('category'):
            X[c] = pd.Categorical(X[c]).codes.astype(float)
            X[c] = X[c].replace(-1, np.nan)
    return X.to_numpy(dtype=float)


def run_once(name, prepare_fn, *,
             lr=0.5, hedge_eta=1.0, n_passes=3, ridge=1e-3,
             standardize=True, seed=SEED):
    print(f"\n{'='*80}\nBaseline: PUFE | Dataset: {name}\n{'='*80}", flush=True)
    t0 = time.time()
    np.random.seed(seed)

    df, label, base_features, ext_features = prepare_fn()
    print(f"Shape: {df.shape} | base={len(base_features)} ext={len(ext_features)}")

    train_df, test_df = split_train_test(df, label, ext_features, seed=seed)
    print(f"Train: {len(train_df)} | Test: {len(test_df)}", flush=True)

    test_with = test_df[test_df['has_extended'] == 1]
    test_without = test_df[test_df['has_extended'] == 0]
    print(f"Test — has_ext: {len(test_with)} | no_ext: {len(test_without)}",
          flush=True)

    # Shuffle training rows (streaming order) — same as AdaptiveXGBoost runner
    train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    y_tr = train_df[label].to_numpy()
    ext_tr = train_df['has_extended'].to_numpy()
    Xb_tr = _to_matrix(train_df, base_features)
    Xe_tr = _to_matrix(train_df, ext_features)

    y_te = test_df[label].to_numpy()
    ext_te = test_df['has_extended'].to_numpy()
    Xb_te = _to_matrix(test_df, base_features)
    Xe_te = _to_matrix(test_df, ext_features)

    clf = PUFEClassifier(lr=lr, hedge_eta=hedge_eta, n_passes=n_passes,
                          ridge=ridge, standardize=standardize, verbose=True)
    clf.fit(Xb_tr, Xe_tr, y_tr, ext_tr)

    p_all = clf.predict_proba(Xb_te, Xe_te, ext_te)[:, 1]
    auc_overall = roc_auc_score(y_te, p_all)

    # Per-population AUCs
    idx_w = np.where(ext_te == 1)[0]
    idx_n = np.where(ext_te == 0)[0]
    auc_with = roc_auc_score(y_te[idx_w], p_all[idx_w]) if len(idx_w) > 0 else float('nan')
    auc_without = roc_auc_score(y_te[idx_n], p_all[idx_n]) if len(idx_n) > 0 else float('nan')

    n_no = int(len(idx_n)); n_ext = int(len(idx_w))
    pop_weighted = (n_no * auc_without + n_ext * auc_with) / (n_no + n_ext)

    elapsed = time.time() - t0
    print(f"\nResults (PUFE, {name}):")
    print(f"  AUC (no_extended):  {auc_without:.6f}  (n={n_no})")
    print(f"  AUC (has_extended): {auc_with:.6f}  (n={n_ext})")
    print(f"  AUC (overall):      {auc_overall:.6f}  (n={n_no+n_ext})")
    print(f"  AUC (pop-weighted): {pop_weighted:.6f}")
    print(f"  Hedge α_P={clf.hedge_weights_[0]:.4f}  α_C={clf.hedge_weights_[1]:.4f}")
    print(f"  Elapsed: {elapsed:.1f}s", flush=True)

    return {
        'dataset': name,
        'baseline': 'PUFE',
        'lr': lr, 'hedge_eta': hedge_eta, 'n_passes': n_passes, 'ridge': ridge,
        'standardize': standardize,
        'auc_no_extended': auc_without,
        'auc_has_extended': auc_with,
        'auc_overall': auc_overall,
        'auc_pop_weighted': pop_weighted,
        'hedge_alpha_P': clf.hedge_weights_[0],
        'hedge_alpha_C': clf.hedge_weights_[1],
        'n_no_extended': n_no,
        'n_has_extended': n_ext,
        'n_train': len(train_df),
        'n_test': len(test_df),
        'elapsed_sec': round(elapsed, 1),
        'seed': seed,
        'date': datetime.now().strftime('%Y-%m-%d'),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default=','.join(DATASET_LOADERS),
                    help=f'Comma-separated list. Default: all ({len(DATASET_LOADERS)}).')
    ap.add_argument('--lr', type=float, default=0.5)
    ap.add_argument('--hedge_eta', type=float, default=1.0)
    ap.add_argument('--n_passes', type=int, default=3)
    ap.add_argument('--ridge', type=float, default=1e-3)
    ap.add_argument('--no_standardize', action='store_true')
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--out', default=os.path.join(ROOT, 'baselines', 'results',
                                                    'pufe_results.csv'))
    args = ap.parse_args()

    to_run = [d.strip() for d in args.datasets.split(',') if d.strip()]
    rows = []
    for ds in to_run:
        if ds not in DATASET_LOADERS:
            print(f"Unknown dataset: {ds}")
            continue
        rows.append(run_once(ds, DATASET_LOADERS[ds],
                              lr=args.lr,
                              hedge_eta=args.hedge_eta,
                              n_passes=args.n_passes,
                              ridge=args.ridge,
                              standardize=not args.no_standardize,
                              seed=args.seed))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df_out = pd.DataFrame(rows)
    if os.path.exists(args.out):
        existing = pd.read_csv(args.out)
        df_out = pd.concat([existing, df_out], ignore_index=True)
    df_out.to_csv(args.out, index=False)
    print(f"\nResults saved to: {args.out}")

    print("\n" + "=" * 80)
    print("BASELINE SUMMARY — PUFE")
    print("=" * 80)
    print(f"{'Dataset':<18} {'AUC no_ext':>12} {'AUC has_ext':>13} {'AUC overall':>13} {'AUC pop-w':>11}")
    for r in rows:
        print(f"{r['dataset']:<18} {r['auc_no_extended']:>12.6f} "
              f"{r['auc_has_extended']:>13.6f} {r['auc_overall']:>13.6f} "
              f"{r['auc_pop_weighted']:>11.6f}")


if __name__ == '__main__':
    main()
