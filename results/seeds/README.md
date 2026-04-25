# Seed-sweep artifacts (Table 4 robustness check)

Generated 2026-04-25 from a 5-seed (42–46) sweep on the BGU cluster
using `scripts/run_ablation_pruning.py 30 <DS> 30` with
`IL_SEED=<S>` and `IL_RESULTS_DIR=results/seeds/seed_<S>`.

## What is in here

* `table4_seedsweep.csv` (long format)
  – 5 seeds × 9 datasets × 3 pruning modes = 135 rows.
    Columns: `seed, dataset, mode, objective, base_auc, ext_auc,
    comb_no_auc, comb_ext_auc, n_no, n_ext`.

* `table4_meanstd.csv`
  – Per-dataset mean ± std across 5 seeds (optuna mode only) for
    Our base / Our ext / BL base / BL ext / weighted improvement.

* `seed_<S>/ablation/<DS>/{base,combined,extended_*}.json`
  – Trained model artefacts for every (seed, dataset) cell that
    completed warm-start successfully.

* `seed_<S>_summary.txt`
  – Per-seed ablation summary (only the last finishing dataset's
    rows; full data is in the long-format CSV).

## Why MovieAugV2 is missing

MovieAugV2 has a 1.9% positive class. The locked-in 6-feature ext
combo combined with the structured null injection yields a
(label × has_extended) stratum below the >100-sample threshold that
`core/RunData.py` enforces, so the dataset is skipped on every seed.
Its single-seed value is retained in the published Table 4.

## Headline finding

Paired Wilcoxon (Ours - Baseline Combined weighted improvement,
one-sided) over 45 (seed, dataset) cells: **W = 523, p = 0.48**.
Mean weighted improvement: **+0.0010 ± 0.0052**. 5 of 9 datasets
have positive mean, 4 have negative; only BankLoanSta (+0.011 ± 0.010)
has a mean larger than its own seed std. The conclusion is parity
with the data-sharing baseline, not a strict improvement.

External-baseline gaps (GBDT-IL +0.041, PUFE +0.053, EMLI +0.054,
AdaptiveXGBoost +0.079, OCDS +0.140 in mean pop-weighted AUC) are
an order of magnitude larger than this seed std and therefore
remain robust.

These artefacts are not used as the source-of-truth for any number
in the paper. They exist as reviewer-facing evidence that the
parity claim is the honest reading of the results.
