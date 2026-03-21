#!/usr/bin/env python3
"""Run all dataset experiments sequentially and generate a unified summary.

Usage:
    python run_all_experiments.py [N_TRIALS] [DATASET_FILTER]

Examples:
    python run_all_experiments.py 30                    # all datasets, 30 trials
    python run_all_experiments.py 30 Weather,WIDS       # only Weather and WIDS
"""
import sys
import os
import subprocess
import time
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

PYTHON = sys.executable
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPTS_DIR, '..', 'results')

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
DATASET_FILTER = sys.argv[2].split(',') if len(sys.argv) > 2 else None

# Dataset registry: (name, script_file, extra_args)
# Each script accepts N_TRIALS as first arg
DATASETS = [
    ('BankLoanSta',     'run_augmented_combos.py',        []),
    ('Weather',         'run_weather.py',                  []),
    ('DiabetesRecord',  'run_diabetes.py',                 []),
    ('HRAnalytics',     'run_hr_analytics.py',             []),
    ('ClientRecordAug', 'run_client_record_augmented.py',  []),
    ('MovieAugV2',      'run_movie_augmented_v2.py',       []),
    ('WeatherAUS',      'run_weatheraus.py',               []),
    ('WIDS',            'run_wids.py',                     []),
    ('FlightDelay',     'run_airline.py',                  []),
]

print("=" * 100)
print(f"RUNNING ALL EXPERIMENTS — N_TRIALS={N_TRIALS}")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
if DATASET_FILTER:
    print(f"Filter: {DATASET_FILTER}")
print("=" * 100)

all_results = {}

for ds_name, script_file, extra_args in DATASETS:
    if DATASET_FILTER and ds_name not in DATASET_FILTER:
        print(f"\nSkipping {ds_name} (not in filter)")
        continue

    script_path = os.path.join(SCRIPTS_DIR, script_file)
    if not os.path.exists(script_path):
        print(f"\nWARNING: Script not found: {script_path}, skipping {ds_name}")
        continue

    print(f"\n{'#' * 100}")
    print(f"# DATASET: {ds_name} — {script_file} (n_trials={N_TRIALS})")
    print(f"# Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'#' * 100}")
    sys.stdout.flush()

    # Build command
    cmd = [PYTHON, script_path, str(N_TRIALS)] + extra_args

    # Run and stream output
    start_time = time.time()
    result_file = os.path.join(RESULTS_DIR, ds_name, f'{ds_name.lower()}_results_t{N_TRIALS}.txt')
    os.makedirs(os.path.join(RESULTS_DIR, ds_name), exist_ok=True)

    try:
        proc = subprocess.run(
            cmd,
            cwd=os.path.join(SCRIPTS_DIR, '..'),
            capture_output=False,
            text=True,
        )
        elapsed = time.time() - start_time
        exit_code = proc.returncode
    except Exception as e:
        elapsed = time.time() - start_time
        exit_code = -2
        print(f"\nERROR: {ds_name} failed with {e}")

    status = "OK" if exit_code == 0 else f"FAILED (exit={exit_code})"
    print(f"\n>>> {ds_name}: {status} ({elapsed:.0f}s)")
    all_results[ds_name] = {
        'status': status,
        'elapsed': elapsed,
        'exit_code': exit_code,
    }
    sys.stdout.flush()

# --- Final report ---
print(f"\n{'=' * 100}")
print(f"ALL EXPERIMENTS COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'=' * 100}")

print(f"\n{'Dataset':<20} {'Status':<20} {'Time':<10}")
print("-" * 50)
for ds_name, info in all_results.items():
    elapsed_str = f"{info['elapsed']:.0f}s"
    print(f"{ds_name:<20} {info['status']:<20} {elapsed_str:<10}")

n_ok = sum(1 for v in all_results.values() if v['exit_code'] == 0)
n_total = len(all_results)
print(f"\nSuccess: {n_ok}/{n_total}")

# --- Parse results from individual output files and build summary ---
print(f"\n{'=' * 100}")
print("COLLECTING BEST COMBOS FROM EACH DATASET")
print(f"{'=' * 100}")

import re

summary_lines = []
summary_lines.append("=" * 100)
summary_lines.append(f"INCREMENTAL LEARNING EXPERIMENT SUMMARY (N_TRIALS={N_TRIALS})")
summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
summary_lines.append("=" * 100)
summary_lines.append("")
summary_lines.append("Objective = (combined_with_ext_auc - extended_auc) + (combined_no_ext_auc - base_auc)")
summary_lines.append("Negative objective = incremental learning approach outperforms combined model.")
summary_lines.append("")

# Scan each dataset results directory for the latest results file
best_per_dataset = []

for ds_name, script_file, extra_args in DATASETS:
    if DATASET_FILTER and ds_name not in DATASET_FILTER:
        continue
    if ds_name not in all_results or all_results[ds_name]['exit_code'] != 0:
        continue

    ds_dir = os.path.join(RESULTS_DIR, ds_name)
    if not os.path.isdir(ds_dir):
        continue

    # Find result text files
    txt_files = sorted([f for f in os.listdir(ds_dir) if f.endswith('.txt')],
                       key=lambda x: os.path.getmtime(os.path.join(ds_dir, x)),
                       reverse=True)

    best_obj = 999
    best_combo = None

    for tf in txt_files:
        with open(os.path.join(ds_dir, tf)) as fh:
            content = fh.read()

        # Find all RESULT lines
        for match in re.finditer(
            r'Combo \d+/\d+:\s*(.+?)\n.*?RESULT:\s*objective\s*=\s*([-\d.]+).*?'
            r'Base AUC:\s*([\d.]+).*?Extended AUC:\s*([\d.]+).*?'
            r'Combined \(no ext\) AUC:\s*([\d.]+).*?Combined \(with ext\) AUC:\s*([\d.]+)',
            content, re.DOTALL
        ):
            obj = float(match.group(2))
            if obj < best_obj:
                best_obj = obj
                best_combo = {
                    'name': match.group(1).strip(),
                    'objective': obj,
                    'base_auc': float(match.group(3)),
                    'ext_auc': float(match.group(4)),
                    'comb_no': float(match.group(5)),
                    'comb_ext': float(match.group(6)),
                }
        # Only check the most recent file
        break

    if best_combo:
        best_per_dataset.append((ds_name, best_combo))
        summary_lines.append(f"--- {ds_name} ---")
        summary_lines.append(f"  Best combo: {best_combo['name']} (objective: {best_combo['objective']:.6f})")
        summary_lines.append(f"  Base AUC: {best_combo['base_auc']:.4f}, Ext AUC: {best_combo['ext_auc']:.4f}")
        summary_lines.append(f"  Comb-no: {best_combo['comb_no']:.4f}, Comb+ext: {best_combo['comb_ext']:.4f}")
        winner = "Incremental" if best_combo['objective'] < 0 else "Combined"
        summary_lines.append(f"  Winner: {winner}")
        summary_lines.append("")

# Overall stats
n_increm = sum(1 for _, c in best_per_dataset if c['objective'] < 0)
n_total_ds = len(best_per_dataset)
summary_lines.append("=" * 100)
summary_lines.append(f"OVERALL: Incremental wins {n_increm}/{n_total_ds} datasets")
summary_lines.append("=" * 100)

# Print and save summary
summary_text = "\n".join(summary_lines)
print(summary_text)

summary_path = os.path.join(RESULTS_DIR, f'summary_t{N_TRIALS}.txt')
with open(summary_path, 'w') as f:
    f.write(summary_text)
print(f"\nSummary saved to: {summary_path}")
print("Done!")
