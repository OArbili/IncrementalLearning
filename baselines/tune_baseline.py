"""Tier-2 generic Optuna driver for the 5 external baselines.

Each baseline gets the SAME budget (default 15 trials) per dataset so that
reviewers cannot claim our framework had an unfair tuning advantage.

Pipeline per (baseline, dataset):
    1. Load via DATASET_LOADERS[dataset]() -> (df, label, base_features, ext_features).
    2. Stratified train/test split (label x has_extended) via split_train_test
       with seed=42 — identical to all other baseline runners.
    3. Internally split train_df into tune_train (80%) / tune_val (20%)
       stratified by label x has_extended.
    4. Optuna TPE search on tune_val population-weighted AUC.  ONLY tune_val
       is shown to Optuna — test_df is untouched until the final eval.
    5. Refit on the full train_df with the best hparams; report
       auc_no_extended, auc_has_extended, auc_pop_weighted on test_df.
    6. Append a single row to <out CSV>.

Usage:
    python -m baselines.tune_baseline --baseline axgb --dataset Weather \
        --n_trials 15 --seed 42 --out baselines/results/tier2_tuned.csv
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from scripts.prepare_datasets import DATASET_LOADERS, split_train_test  # noqa: E402

import optuna  # noqa: E402
from optuna.samplers import TPESampler  # noqa: E402

# Baseline classes
from baselines.adaptive_xgboost.adaptive_xgboost import AdaptiveXGBoostClassifier  # noqa: E402
from baselines.pufe.pufe import PUFEClassifier  # noqa: E402
from baselines.ocds.ocds import OCDSClassifier  # noqa: E402
from baselines.emli.emli import EMLIClassifier  # noqa: E402
from baselines.gbdt_il.gbdt_il import GBDTIL  # noqa: E402

SEED = 42

BASELINES = ('axgb', 'pufe', 'ocds', 'emli', 'gbdt_il')


# ----------------------------------------------------------------------
# Shared matrix conversion (categoricals -> codes; NaNs preserved).
# ----------------------------------------------------------------------

def _to_matrix(df, features, keep_nan=True):
    X = df[features].copy()
    for c in X.columns:
        if X[c].dtype == 'object' or str(X[c].dtype).startswith('category'):
            X[c] = pd.Categorical(X[c]).codes.astype(float)
            if keep_nan:
                X[c] = X[c].replace(-1, np.nan)
    return X.to_numpy(dtype=float)


def _stratified_internal_split(train_df, label, seed, val_size=0.2):
    """Split train_df into tune_train/tune_val stratified by label x has_extended."""
    strat = train_df[label].astype(str) + '_' + train_df['has_extended'].astype(str)
    tr_idx, val_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=val_size,
        random_state=seed,
        stratify=strat,
    )
    return train_df.iloc[tr_idx].reset_index(drop=True), train_df.iloc[val_idx].reset_index(drop=True)


def _pop_weighted_auc(y_true, y_score, has_ext):
    """Population-weighted AUC. has_ext is per-row 0/1."""
    has_ext = np.asarray(has_ext)
    idx_w = np.where(has_ext == 1)[0]
    idx_n = np.where(has_ext == 0)[0]
    auc_w = roc_auc_score(y_true[idx_w], y_score[idx_w]) if len(idx_w) > 0 and len(np.unique(y_true[idx_w])) > 1 else float('nan')
    auc_n = roc_auc_score(y_true[idx_n], y_score[idx_n]) if len(idx_n) > 0 and len(np.unique(y_true[idx_n])) > 1 else float('nan')
    n_w, n_n = len(idx_w), len(idx_n)
    # If a population has only one class, fall back to the other (rare).
    if np.isnan(auc_w) and np.isnan(auc_n):
        return float('nan'), float('nan'), float('nan'), n_n, n_w
    if np.isnan(auc_w):
        pop = auc_n
    elif np.isnan(auc_n):
        pop = auc_w
    else:
        pop = (n_n * auc_n + n_w * auc_w) / (n_n + n_w)
    return auc_n, auc_w, pop, n_n, n_w


# ----------------------------------------------------------------------
# Per-baseline: search space + fit/score helpers.
# Each helper takes (params, tune_train, tune_val, label, base_features,
# ext_features, seed) and returns the validation pop-weighted AUC.
# A separate "final_fit_predict" helper trains on full train_df and
# returns scores on test_df.
# ----------------------------------------------------------------------

# ---- AXGB -------------------------------------------------------------

def _axgb_space(trial):
    return dict(
        n_estimators=trial.suggest_categorical('n_estimators', [30, 50, 100]),
        learning_rate=trial.suggest_float('learning_rate', 0.05, 0.3, log=True),
        max_depth=trial.suggest_categorical('max_depth', [3, 5, 7, 9]),
        max_window_size=trial.suggest_categorical('max_window_size', [500, 1000, 2000, 5000]),
        update_strategy=trial.suggest_categorical('update_strategy', ['push', 'replace']),
    )


def _axgb_fit_score(params, train_df, eval_df, label, base_features, ext_features, seed):
    all_features = base_features + ext_features
    X_tr = _to_matrix(train_df, all_features)
    y_tr = train_df[label].to_numpy()
    X_ev = _to_matrix(eval_df, all_features)
    y_ev = eval_df[label].to_numpy()
    has_ev = eval_df['has_extended'].to_numpy()

    clf = AdaptiveXGBoostClassifier(
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        max_window_size=params['max_window_size'],
        update_strategy=params['update_strategy'],
        detect_drift=False,
    )
    batch = max(params['max_window_size'], 500)
    for s in range(0, len(X_tr), batch):
        e = min(s + batch, len(X_tr))
        clf.partial_fit(X_tr[s:e], y_tr[s:e])
    proba = clf.predict_proba(X_ev)[:, 1]
    return y_ev, proba, has_ev


# ---- PUFE -------------------------------------------------------------

def _pufe_space(trial):
    return dict(
        lr=trial.suggest_float('lr', 0.05, 1.0, log=True),
        hedge_eta=trial.suggest_float('hedge_eta', 0.5, 5.0, log=True),
        n_passes=trial.suggest_categorical('n_passes', [1, 3, 5]),
        ridge=trial.suggest_float('ridge', 1e-4, 1e-1, log=True),
    )


def _pufe_fit_score(params, train_df, eval_df, label, base_features, ext_features, seed):
    Xb_tr = _to_matrix(train_df, base_features)
    Xe_tr = _to_matrix(train_df, ext_features)
    y_tr = train_df[label].to_numpy()
    ext_tr = train_df['has_extended'].to_numpy()

    Xb_ev = _to_matrix(eval_df, base_features)
    Xe_ev = _to_matrix(eval_df, ext_features)
    y_ev = eval_df[label].to_numpy()
    has_ev = eval_df['has_extended'].to_numpy()

    np.random.seed(seed)
    clf = PUFEClassifier(
        lr=params['lr'], hedge_eta=params['hedge_eta'],
        n_passes=params['n_passes'], ridge=params['ridge'],
        standardize=True, verbose=False,
    )
    clf.fit(Xb_tr, Xe_tr, y_tr, ext_tr)
    proba = clf.predict_proba(Xb_ev, Xe_ev, has_ev)[:, 1]
    return y_ev, proba, has_ev


# ---- OCDS -------------------------------------------------------------

def _ocds_space(trial):
    return dict(
        gamma=trial.suggest_float('gamma', 0.1, 1.0),
        lam1=trial.suggest_float('lam1', 1e-4, 1e-1, log=True),
        n_passes=trial.suggest_categorical('n_passes', [1, 2, 3]),
        hedge_eta=trial.suggest_float('hedge_eta', 1.0, 10.0, log=True),
        ridge_G=trial.suggest_float('ridge_G', 1e-4, 1e-1, log=True),
    )


def _ocds_fit_score(params, train_df, eval_df, label, base_features, ext_features, seed):
    Xb_tr = _to_matrix(train_df, base_features)
    Xe_tr = _to_matrix(train_df, ext_features)
    y_tr = train_df[label].to_numpy()
    ext_tr = train_df['has_extended'].to_numpy()

    Xb_ev = _to_matrix(eval_df, base_features)
    Xe_ev = _to_matrix(eval_df, ext_features)
    y_ev = eval_df[label].to_numpy()
    has_ev = eval_df['has_extended'].to_numpy()

    np.random.seed(seed)
    clf = OCDSClassifier(
        gamma=params['gamma'], lam1=params['lam1'],
        n_passes=params['n_passes'], hedge_eta=params['hedge_eta'],
        ridge_G=params['ridge_G'], standardize=True, seed=seed, verbose=False,
    )
    clf.fit(Xb_tr, Xe_tr, y_tr, ext_tr)
    scores = clf.decision_function(Xb_ev, Xe_ev, has_ev)
    return y_ev, scores, has_ev


# ---- EMLI -------------------------------------------------------------

def _emli_space(trial):
    return dict(
        k=trial.suggest_categorical('k', [8, 16, 32, 64]),
        margin=trial.suggest_float('margin', 0.1, 2.0),
        lr=trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        n_epochs=trial.suggest_categorical('n_epochs', [10, 20, 30]),
        triplet_weight=trial.suggest_float('triplet_weight', 0.5, 2.0),
        consistency_weight=trial.suggest_float('consistency_weight', 0.1, 1.0),
    )


def _emli_fit_score(params, train_df, eval_df, label, base_features, ext_features, seed):
    Xb_tr = _to_matrix(train_df, base_features)
    Xe_tr = _to_matrix(train_df, ext_features)
    y_tr = train_df[label].to_numpy()
    ext_tr = train_df['has_extended'].to_numpy()

    Xb_ev = _to_matrix(eval_df, base_features)
    Xe_ev = _to_matrix(eval_df, ext_features)
    y_ev = eval_df[label].to_numpy()
    has_ev = eval_df['has_extended'].to_numpy()

    np.random.seed(seed)
    clf = EMLIClassifier(
        k=params['k'], margin=params['margin'], lr=params['lr'],
        n_epochs=params['n_epochs'], batch_size=256,
        triplet_weight=params['triplet_weight'],
        consistency_weight=params['consistency_weight'],
        lowrank_weight=1e-3, cls_weight=1.0,
        standardize=True, seed=seed, verbose=False,
    )
    clf.fit(Xb_tr, Xe_tr, y_tr, ext_tr)
    proba = clf.predict_proba(Xb_ev, Xe_ev, has_ev)[:, 1]
    return y_ev, proba, has_ev


# ---- GBDT-IL ----------------------------------------------------------

def _gbdtil_space(trial):
    return dict(
        initial_trees=trial.suggest_categorical('initial_trees', [100, 250, 500]),
        num_inc_tree=trial.suggest_categorical('num_inc_tree', [15, 25, 50]),
        win_size=trial.suggest_categorical('win_size', [300, 500, 1000]),
        learning_rate=trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        max_depth=trial.suggest_categorical('max_depth', [6, 8, 10]),
    )


def _gbdtil_fit_score(params, train_df, eval_df, label, base_features, ext_features, seed):
    all_features = base_features + ext_features
    Xf_tr = _to_matrix(train_df, all_features)
    y_tr = train_df[label].to_numpy()
    Xf_ev = _to_matrix(eval_df, all_features)
    y_ev = eval_df[label].to_numpy()
    has_ev = eval_df['has_extended'].to_numpy()

    np.random.seed(seed)
    clf = GBDTIL(
        initial_trees=params['initial_trees'],
        num_inc_tree=params['num_inc_tree'],
        init_size=1000,
        win_size=params['win_size'],
        max_tree=10000,
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        min_child_weight=5,
        prefix_search_step=5,
        seed=seed, verbose=False,
    )
    clf.fit(Xf_tr, y_tr, feature_names=all_features)
    proba = clf.predict_proba(Xf_ev)[:, 1]
    return y_ev, proba, has_ev


# ---- Registry ---------------------------------------------------------

REGISTRY = {
    'axgb':    {'name': 'AdaptiveXGBoost', 'space': _axgb_space,    'fit_score': _axgb_fit_score},
    'pufe':    {'name': 'PUFE',            'space': _pufe_space,    'fit_score': _pufe_fit_score},
    'ocds':    {'name': 'OCDS',            'space': _ocds_space,    'fit_score': _ocds_fit_score},
    'emli':    {'name': 'EMLI',            'space': _emli_space,    'fit_score': _emli_fit_score},
    'gbdt_il': {'name': 'GBDT-IL',         'space': _gbdtil_space,  'fit_score': _gbdtil_fit_score},
}


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------

def tune_one(baseline, dataset, n_trials, seed, out_path):
    if baseline not in REGISTRY:
        raise SystemExit(f"Unknown baseline '{baseline}'. Choose from: {list(REGISTRY)}")
    if dataset not in DATASET_LOADERS:
        raise SystemExit(f"Unknown dataset '{dataset}'. Choose from: {list(DATASET_LOADERS)}")

    info = REGISTRY[baseline]
    print(f"\n{'='*80}")
    print(f"TIER-2 TUNE | baseline={info['name']} | dataset={dataset} | n_trials={n_trials}")
    print(f"{'='*80}", flush=True)
    t0 = time.time()

    df, label, base_features, ext_features = DATASET_LOADERS[dataset]()
    train_df, test_df = split_train_test(df, label, ext_features, seed=seed)

    # Shuffle training rows once (matches existing baselines' streaming order).
    train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Internal stratified split for Optuna evaluation.
    tune_train, tune_val = _stratified_internal_split(train_df, label, seed=seed, val_size=0.2)
    print(f"train_full={len(train_df)} | tune_train={len(tune_train)} | "
          f"tune_val={len(tune_val)} | test={len(test_df)}", flush=True)
    print(f"tune_val populations: has_ext={(tune_val['has_extended']==1).sum()} | "
          f"no_ext={(tune_val['has_extended']==0).sum()}", flush=True)

    space_fn = info['space']
    fit_score_fn = info['fit_score']

    def objective(trial):
        params = space_fn(trial)
        try:
            y_ev, scores, has_ev = fit_score_fn(
                params, tune_train, tune_val, label,
                base_features, ext_features, seed,
            )
        except Exception as exc:
            print(f"  trial {trial.number} FAILED: {exc}", flush=True)
            raise optuna.TrialPruned()
        _, _, pop, _, _ = _pop_weighted_auc(y_ev, scores, has_ev)
        if np.isnan(pop):
            raise optuna.TrialPruned()
        print(f"  trial {trial.number}: pop_auc={pop:.4f} | {params}", flush=True)
        return pop

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_val_auc = study.best_value
    print(f"\nBest tune_val pop-weighted AUC: {best_val_auc:.6f}")
    print(f"Best params: {best_params}", flush=True)

    # Final refit on the full train_df, evaluate on test_df.
    print("\nRefitting on full train_df with best params...", flush=True)
    y_te, scores_te, has_te = fit_score_fn(
        best_params, train_df, test_df, label,
        base_features, ext_features, seed,
    )
    auc_no, auc_has, auc_pop, n_no, n_has = _pop_weighted_auc(y_te, scores_te, has_te)
    elapsed = time.time() - t0

    print(f"\nTEST: auc_no_extended={auc_no:.6f} | auc_has_extended={auc_has:.6f} | "
          f"auc_pop_weighted={auc_pop:.6f}")
    print(f"Elapsed: {elapsed:.1f}s", flush=True)

    row = {
        'dataset': dataset,
        'baseline': info['name'],
        'baseline_key': baseline,
        'n_trials': n_trials,
        'best_hparams': json.dumps(best_params),
        'best_val_pop_auc': best_val_auc,
        'auc_no_extended': auc_no,
        'auc_has_extended': auc_has,
        'auc_pop_weighted': auc_pop,
        'n_no_extended': n_no,
        'n_has_extended': n_has,
        'n_train': len(train_df),
        'n_test': len(test_df),
        'elapsed_sec': round(elapsed, 1),
        'seed': seed,
        'date': datetime.now().strftime('%Y-%m-%d'),
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df_out = pd.DataFrame([row])
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        df_out = pd.concat([existing, df_out], ignore_index=True)
    df_out.to_csv(out_path, index=False)
    print(f"\nAppended row to: {out_path}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', required=True, choices=list(BASELINES))
    ap.add_argument('--dataset', required=True, choices=list(DATASET_LOADERS))
    ap.add_argument('--n_trials', type=int, default=15)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--out', default=os.path.join(ROOT, 'baselines', 'results', 'tier2_tuned.csv'))
    args = ap.parse_args()

    tune_one(args.baseline, args.dataset, args.n_trials, args.seed, args.out)


if __name__ == '__main__':
    main()
