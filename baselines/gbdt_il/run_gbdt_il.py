"""Run GBDT-IL baseline on all 10 datasets.

Uses EXACTLY the same preprocessing, features, and stratified train/test
split as our framework and prior baselines, via scripts/prepare_datasets.py.
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
from baselines.gbdt_il.gbdt_il import GBDTIL

SEED = 42


def _to_matrix(df, features):
    X = df[features].copy()
    for c in X.columns:
        if X[c].dtype == 'object' or str(X[c].dtype).startswith('category'):
            X[c] = pd.Categorical(X[c]).codes.astype(float)
            X[c] = X[c].replace(-1, np.nan)
    return X.to_numpy(dtype=float)


def run_once(name, prepare_fn, *,
             initial_trees=250, num_inc_tree=25, init_size=1000, win_size=500,
             max_tree=10000, learning_rate=0.01, max_depth=10,
             min_child_weight=5, prefix_search_step=5, seed=SEED):
    print(f"\n{'='*80}\nBaseline: GBDT-IL | Dataset: {name}\n{'='*80}", flush=True)
    t0 = time.time()
    np.random.seed(seed)

    df, label, base_features, ext_features = prepare_fn()
    all_features = base_features + ext_features
    print(f"Shape: {df.shape} | base={len(base_features)} ext={len(ext_features)}",
          flush=True)

    train_df, test_df = split_train_test(df, label, ext_features, seed=seed)
    print(f"Train: {len(train_df)} | Test: {len(test_df)}", flush=True)
    print(f"Test — has_ext: {(test_df['has_extended']==1).sum()} | "
          f"no_ext: {(test_df['has_extended']==0).sum()}", flush=True)

    # Shuffle training rows (streaming order)
    train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    y_tr = train_df[label].to_numpy()
    Xf_tr = _to_matrix(train_df, all_features)
    y_te = test_df[label].to_numpy()
    ext_te = test_df['has_extended'].to_numpy()
    Xf_te = _to_matrix(test_df, all_features)

    clf = GBDTIL(initial_trees=initial_trees, num_inc_tree=num_inc_tree,
                  init_size=init_size, win_size=win_size, max_tree=max_tree,
                  learning_rate=learning_rate, max_depth=max_depth,
                  min_child_weight=min_child_weight,
                  prefix_search_step=prefix_search_step,
                  seed=seed, verbose=True)
    clf.fit(Xf_tr, y_tr, feature_names=all_features)

    proba = clf.predict_proba(Xf_te)[:, 1]
    auc_overall = roc_auc_score(y_te, proba)
    idx_w = np.where(ext_te == 1)[0]
    idx_n = np.where(ext_te == 0)[0]
    auc_with = roc_auc_score(y_te[idx_w], proba[idx_w]) if len(idx_w) else float('nan')
    auc_without = roc_auc_score(y_te[idx_n], proba[idx_n]) if len(idx_n) else float('nan')
    n_no, n_ext = int(len(idx_n)), int(len(idx_w))
    pop_w = (n_no * auc_without + n_ext * auc_with) / (n_no + n_ext)

    # History summary
    n_prune = sum(1 for h in clf.history_ if h['action'] == 'prune')
    n_drift = sum(1 for h in clf.history_ if h['action'] == 'drift_retrain')
    final_trees = clf.booster_.num_boosted_rounds()

    elapsed = time.time() - t0
    print(f"\nResults (GBDT-IL, {name}):")
    print(f"  AUC (no_extended):  {auc_without:.6f}  (n={n_no})")
    print(f"  AUC (has_extended): {auc_with:.6f}  (n={n_ext})")
    print(f"  AUC (overall):      {auc_overall:.6f}")
    print(f"  AUC (pop-weighted): {pop_w:.6f}")
    print(f"  Ensemble: final_trees={final_trees}, "
          f"{n_prune} pruning ops, {n_drift} drift-retrains")
    print(f"  Elapsed: {elapsed:.1f}s", flush=True)

    return {
        'dataset': name,
        'baseline': 'GBDT-IL',
        'initial_trees': initial_trees, 'num_inc_tree': num_inc_tree,
        'init_size': init_size, 'win_size': win_size, 'max_tree': max_tree,
        'learning_rate': learning_rate, 'max_depth': max_depth,
        'min_child_weight': min_child_weight,
        'auc_no_extended': auc_without,
        'auc_has_extended': auc_with,
        'auc_overall': auc_overall,
        'auc_pop_weighted': pop_w,
        'final_trees': int(final_trees),
        'n_prune_ops': n_prune,
        'n_drift_retrains': n_drift,
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
    ap.add_argument('--datasets', default=','.join(DATASET_LOADERS))
    ap.add_argument('--initial_trees', type=int, default=250)   # paper max_iter
    ap.add_argument('--num_inc_tree', type=int, default=25)
    ap.add_argument('--init_size', type=int, default=1000)
    ap.add_argument('--win_size', type=int, default=500)
    ap.add_argument('--max_tree', type=int, default=10000)
    ap.add_argument('--learning_rate', type=float, default=0.01)
    ap.add_argument('--max_depth', type=int, default=10)
    ap.add_argument('--min_child_weight', type=int, default=5)
    ap.add_argument('--prefix_search_step', type=int, default=5)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--out', default=os.path.join(ROOT, 'baselines', 'results',
                                                    'gbdt_il_results.csv'))
    args = ap.parse_args()

    to_run = [d.strip() for d in args.datasets.split(',') if d.strip()]
    rows = []
    for ds in to_run:
        if ds not in DATASET_LOADERS:
            print(f"Unknown dataset: {ds}"); continue
        rows.append(run_once(
            ds, DATASET_LOADERS[ds],
            initial_trees=args.initial_trees,
            num_inc_tree=args.num_inc_tree,
            init_size=args.init_size,
            win_size=args.win_size,
            max_tree=args.max_tree,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            min_child_weight=args.min_child_weight,
            prefix_search_step=args.prefix_search_step,
            seed=args.seed,
        ))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df_out = pd.DataFrame(rows)
    if os.path.exists(args.out):
        existing = pd.read_csv(args.out)
        df_out = pd.concat([existing, df_out], ignore_index=True)
    df_out.to_csv(args.out, index=False)
    print(f"\nResults saved to: {args.out}")

    print("\n" + "=" * 80)
    print("BASELINE SUMMARY — GBDT-IL")
    print("=" * 80)
    print(f"{'Dataset':<18} {'AUC no_ext':>12} {'AUC has_ext':>13} {'AUC overall':>13} {'AUC pop-w':>11}")
    for r in rows:
        print(f"{r['dataset']:<18} {r['auc_no_extended']:>12.6f} "
              f"{r['auc_has_extended']:>13.6f} {r['auc_overall']:>13.6f} "
              f"{r['auc_pop_weighted']:>11.6f}")


if __name__ == '__main__':
    main()
