"""Aggregate Tier-2 tuned baseline results into LaTeX-ready tables.

Reads baselines/results/tier2_tuned.csv (5 baselines x 10 datasets = 50 rows)
and prints three pivot tables (Base AUC = no_extended, Ext AUC = has_extended,
Pop-weighted AUC) in both plain and LaTeX-row formats.

Usage:
    python baselines/tier2_aggregate.py
    python baselines/tier2_aggregate.py --csv path/to/tier2_tuned.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from scripts.prepare_datasets import DATASET_LOADERS  # noqa: E402

DEFAULT_CSV = os.path.join(ROOT, 'baselines', 'results', 'tier2_tuned.csv')

# Display order for tables (rows = methods, cols = datasets).
METHOD_ORDER = ['AdaptiveXGBoost', 'PUFE', 'OCDS', 'EMLI', 'GBDT-IL']
DATASET_ORDER = list(DATASET_LOADERS)


def _pivot(df, value_col):
    out = df.pivot_table(index='baseline', columns='dataset', values=value_col, aggfunc='mean')
    # Reorder rows / cols to canonical layout (missing entries left as NaN).
    out = out.reindex(index=METHOD_ORDER, columns=DATASET_ORDER)
    return out


def _print_table(title, pivot):
    print(f"\n{'='*100}\n{title}\n{'='*100}")
    with pd.option_context('display.float_format', '{:.4f}'.format,
                           'display.width', 200,
                           'display.max_columns', 20):
        print(pivot)


def _print_latex_rows(title, pivot):
    """Emit one LaTeX row per method (baseline). Caller wires into the table."""
    print(f"\n--- LaTeX rows: {title} ---")
    for method in pivot.index:
        cells = []
        for ds in pivot.columns:
            v = pivot.loc[method, ds]
            cells.append('--' if pd.isna(v) else f"{v:.4f}")
        print(f"{method:<18} & " + " & ".join(cells) + r" \\")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=DEFAULT_CSV, help=f'Default: {DEFAULT_CSV}')
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f"CSV not found: {args.csv}\nRun the SLURM array first "
                 f"(baselines/submit_tier2.sh).")

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    print(f"Baselines: {sorted(df['baseline'].unique())}")
    print(f"Datasets:  {sorted(df['dataset'].unique())}")

    expected = len(METHOD_ORDER) * len(DATASET_ORDER)
    if len(df) < expected:
        print(f"WARNING: expected {expected} rows (5 methods x 10 datasets), "
              f"have {len(df)}. Some array tasks may not have completed.")

    base_pivot = _pivot(df, 'auc_no_extended')
    ext_pivot = _pivot(df, 'auc_has_extended')
    pop_pivot = _pivot(df, 'auc_pop_weighted')

    _print_table('Base AUC (no_extended population) — Table 6', base_pivot)
    _print_table('Ext AUC (has_extended population) — Table 7', ext_pivot)
    _print_table('Population-weighted AUC — Table 8', pop_pivot)

    _print_latex_rows('Base AUC (Table 6)', base_pivot)
    _print_latex_rows('Ext AUC (Table 7)', ext_pivot)
    _print_latex_rows('Pop-weighted AUC (Table 8)', pop_pivot)

    # Per-method mean (handy summary for paper text).
    print("\n--- Per-method mean across the 10 datasets ---")
    summary = pd.DataFrame({
        'mean_base_auc': base_pivot.mean(axis=1),
        'mean_ext_auc': ext_pivot.mean(axis=1),
        'mean_pop_auc': pop_pivot.mean(axis=1),
    }).reindex(METHOD_ORDER)
    with pd.option_context('display.float_format', '{:.4f}'.format):
        print(summary)


if __name__ == '__main__':
    main()
