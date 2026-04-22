"""Run AdaptiveXGBoostClassifier baseline on WIDS and ClientRecord.

Uses the SAME data preparation, features (base + extended), and train/test
split as our incremental framework (via scripts/prepare_datasets.py).

AdaptiveXGBoost is a streaming classifier; we feed the training rows in
mini-batches (of size max_window_size) via partial_fit and then evaluate
on the held-out test set, separately on the has_extended and no_extended
populations so numbers are directly comparable to our framework.

Results are appended to baselines/results/baseline_results.csv.
"""
import os
import sys
import time
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Project-root imports
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from scripts.prepare_datasets import DATASET_LOADERS, split_train_test
from baselines.adaptive_xgboost.adaptive_xgboost import AdaptiveXGBoostClassifier

SEED = 42


def _to_matrix(df, features):
    """Convert a DataFrame slice to float matrix with NaN→-1 imputation.

    AdaptiveXGBoost uses plain numpy arrays (via xgb.DMatrix underneath).
    XGBoost supports NaN natively, but the streaming partial_fit path
    goes through np.asarray(..., dtype=float) which keeps NaNs, and
    xgb.train handles them. We still replace any object/categorical
    leftovers with -1 defensively.
    """
    X = df[features].copy()
    for c in X.columns:
        if X[c].dtype == 'object' or str(X[c].dtype).startswith('category'):
            X[c] = pd.Categorical(X[c]).codes
    return X.to_numpy(dtype=float)


def run_once(name, prepare_fn, *,
             n_estimators=30,
             learning_rate=0.3,
             max_depth=6,
             max_window_size=1000,
             min_window_size=None,
             update_strategy='replace',
             detect_drift=False,
             shuffle_stream=True,
             seed=SEED):
    print(f"\n{'='*80}\nBaseline: AdaptiveXGBoost | Dataset: {name}\n{'='*80}")
    t0 = time.time()

    df, label, base_features, ext_features = prepare_fn()
    all_features = base_features + ext_features

    train_df, test_df = split_train_test(df, label, ext_features, seed=seed)
    print(f"Shape: {df.shape} | base={len(base_features)} ext={len(ext_features)}")
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")

    test_with = test_df[test_df['has_extended'] == 1]
    test_without = test_df[test_df['has_extended'] == 0]
    print(f"Test populations — has_extended: {len(test_with)}, no_extended: {len(test_without)}")

    # Stream order: shuffle training rows so the streaming model sees a mixed sequence
    train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    X_train = _to_matrix(train_df, all_features)
    y_train = train_df[label].to_numpy()
    X_test_with = _to_matrix(test_with, all_features)
    X_test_without = _to_matrix(test_without, all_features)
    y_test_with = test_with[label].to_numpy()
    y_test_without = test_without[label].to_numpy()

    clf = AdaptiveXGBoostClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        max_window_size=max_window_size,
        min_window_size=min_window_size,
        detect_drift=detect_drift,
        update_strategy=update_strategy,
    )

    # Feed in mini-batches so progress prints are readable.
    batch = max(max_window_size, 500)
    for start in range(0, len(X_train), batch):
        end = min(start + batch, len(X_train))
        clf.partial_fit(X_train[start:end], y_train[start:end])
        pct = 100 * end / len(X_train)
        print(f"  partial_fit {end}/{len(X_train)} ({pct:.1f}%)", flush=True)

    # Predict and compute AUCs
    auc_with = roc_auc_score(y_test_with, clf.predict_proba(X_test_with)[:, 1]) if len(test_with) > 0 else float('nan')
    auc_without = roc_auc_score(y_test_without, clf.predict_proba(X_test_without)[:, 1]) if len(test_without) > 0 else float('nan')
    auc_overall = roc_auc_score(test_df[label], clf.predict_proba(_to_matrix(test_df, all_features))[:, 1])

    # Population-weighted AUC for direct comparison with our framework
    n_no, n_ext = len(test_without), len(test_with)
    pop_weighted = (n_no * auc_without + n_ext * auc_with) / (n_no + n_ext)

    elapsed = time.time() - t0
    print(f"\nResults (AdaptiveXGBoost, {name}):")
    print(f"  AUC (no_extended):  {auc_without:.6f}  (n={n_no})")
    print(f"  AUC (has_extended): {auc_with:.6f}  (n={n_ext})")
    print(f"  AUC (overall):      {auc_overall:.6f}  (n={n_no+n_ext})")
    print(f"  AUC (pop-weighted): {pop_weighted:.6f}")
    print(f"  Elapsed: {elapsed:.1f}s")

    return {
        'dataset': name,
        'baseline': 'AdaptiveXGBoost',
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'max_window_size': max_window_size,
        'update_strategy': update_strategy,
        'detect_drift': detect_drift,
        'auc_no_extended': auc_without,
        'auc_has_extended': auc_with,
        'auc_overall': auc_overall,
        'auc_pop_weighted': pop_weighted,
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
                    help=f'Comma-separated list. Default: all ({len(DATASET_LOADERS)} datasets)')
    ap.add_argument('--n_estimators', type=int, default=30)
    ap.add_argument('--learning_rate', type=float, default=0.3)
    ap.add_argument('--max_depth', type=int, default=6)
    ap.add_argument('--max_window_size', type=int, default=1000)
    ap.add_argument('--update_strategy', choices=['push', 'replace'], default='replace')
    ap.add_argument('--detect_drift', action='store_true')
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--out', default=os.path.join(ROOT, 'baselines', 'results', 'baseline_results.csv'))
    args = ap.parse_args()

    to_run = [d.strip() for d in args.datasets.split(',') if d.strip()]

    rows = []
    for ds in to_run:
        if ds not in DATASET_LOADERS:
            print(f"Unknown dataset: {ds}. Available: {list(DATASET_LOADERS)}")
            continue
        row = run_once(
            ds, DATASET_LOADERS[ds],
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            max_window_size=args.max_window_size,
            update_strategy=args.update_strategy,
            detect_drift=args.detect_drift,
            seed=args.seed,
        )
        rows.append(row)

    # Append to CSV
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df_out = pd.DataFrame(rows)
    if os.path.exists(args.out):
        existing = pd.read_csv(args.out)
        df_out = pd.concat([existing, df_out], ignore_index=True)
    df_out.to_csv(args.out, index=False)
    print(f"\nResults saved to: {args.out}")

    # Final summary
    print("\n" + "=" * 80)
    print("BASELINE SUMMARY — AdaptiveXGBoost")
    print("=" * 80)
    print(f"{'Dataset':<18} {'AUC no_ext':>12} {'AUC has_ext':>13} {'AUC overall':>13} {'AUC pop-w':>11}")
    for r in rows:
        print(f"{r['dataset']:<18} {r['auc_no_extended']:>12.6f} {r['auc_has_extended']:>13.6f} "
              f"{r['auc_overall']:>13.6f} {r['auc_pop_weighted']:>11.6f}")


if __name__ == '__main__':
    main()
