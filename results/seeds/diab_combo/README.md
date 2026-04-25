# DiabetesRecord candidate-combo screen (rebuttal item)

Pre-registered (see commit history) on 2026-04-25 *before* any results
were inspected. Six ext_features candidates were chosen by non-performance
criteria — natural-missingness rate and clinical interpretability — to
avoid fishing-for-significance.

## Setup

* 6 candidates × 5 seeds (42–46) = 30 SLURM jobs.
* Each job uses `IL_EXT_FEATURES`, `IL_SEED`, and `IL_RESULTS_DIR`
  to keep outputs sandboxed; canonical `results/` is untouched.
* Framework configuration identical to Table 4: `n_trials=30` for
  base + extended + combined, `pruning_mode=optuna`.

## Pre-registered decision rule

> A candidate qualifies to replace the current Table 4 combo iff the
> lower bound of its two-sided 95 % CI on weighted improvement
> (across the 5 seeds) is strictly positive.

Statistic: `wimp_mean - 2.776·wimp_std/sqrt(5) > 0`
(t_{4,0.975} ≈ 2.776).

## Result

Two candidates pass:

| Combo            | Wimp mean ± std    | 95% CI               | Decision |
|------------------|--------------------|---------------------|----------|
| c3_Glu           | +0.00244 ± 0.00160 | (+0.00045, +0.00443) | PASS    |
| c6_A1CplusGlu    | +0.00143 ± 0.00033 | (+0.00102, +0.00184) | PASS    |
| c1_A1Conly       | +0.00045 ± 0.00125 | (−0.00111, +0.00200) | fail    |
| c2_MSonly        | −0.00144 ± 0.00137 | (−0.00314, +0.00026) | fail    |
| c5_MSplusA1C *   | −0.00181 ± 0.00202 | (−0.00432, +0.00070) | fail    |
| c4_Payer         | −0.00201 ± 0.00087 | (−0.00308, −0.00093) | fail    |

\* current Table 4 combo for DiabetesRecord.

## Artifacts

* `diabetes_combo_screen.csv` — long format (one row per combo×seed).
* `diabetes_combo_summary.csv` — per-combo aggregate with CI bounds.
* `<combo>/seed_<S>/ablation/DiabetesRecord/` — model artefacts per cell.

## Pending decision

Two candidates pass; user has not yet chosen whether (and which) to
swap into the published Table 4. The screen is filed for transparency
regardless of that decision.
