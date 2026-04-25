# Rebuttal Item #4 — Multi-Seed Robustness Analysis: Plan

> *"Many reported gains over a strong baseline are small and lack
> robustness analysis across seeds."* — Reviewer

This plan converts every single-point AUC in the paper into a
**mean ± std** estimate across multiple data splits, so we can argue
that our gains are signal, not noise.

---

## 1. Scope and Goal

**Goal.** Replace every single-seed AUC in the paper (Tables 6, 7, 8,
the ablation table, and all baseline comparison tables) with
mean ± std across `S = 5` independent seeds.

**Scope (what is recomputed):**

| Component | How many seeds | Why |
|---|---|---|
| Our framework (Phase 1 + Phase 2 + Combined) | 5 | core claim |
| Pruning ablation (Optuna / NoPrune / Fixed-50%) | 5 | §6.2 |
| AdaptiveXGBoost | 5 | baseline fairness |
| PUFE | 5 | baseline fairness |
| OCDS | 5 | baseline fairness |
| EMLI | 5 | baseline fairness |
| GBDT-IL | 5 | baseline fairness |
| Missingness validation tests | 1 | already deterministic |

Total cells: 10 datasets × 7 methods × 5 seeds = **350 runs** per method-set.

---

## 2. Seed Definition

A single integer "seed" $s$ controls **everything stochastic** inside
one trial, downstream of a fixed dataset:

1. **Train/test split** (`split_train_test`, seed=$s$, stratified by
   $label \times has\_extended$).
2. **Structured-null injection** for augmented datasets
   (`np.random.RandomState(s)`).
3. **Optuna sampler seed** = $s$ for every Bayesian search.
4. **XGBoost `seed` param** = $s$ for every booster.
5. **Numpy / torch global seed** = $s$ at process start.

`S = {42, 43, 44, 45, 46}`. The baseline of `42` matches all current paper numbers, so the existing CSVs become seed-42 rows.

**What does NOT vary across seeds (intentionally):**
- The dataset itself (raw rows are fixed).
- The base/extended feature partition for each dataset (per §4.3 it's
  determined upstream of the BO loop).
- The Optuna search budget (`n_trials = 30`).

---

## 3. Code Changes

### 3.1. Plumb `seed` through every entry point

| File | Change |
|---|---|
| `scripts/prepare_datasets.py` | `split_train_test(..., seed=s)` already accepts seed; structured-null injection in `_load_*` reads `SEED` constant — refactor to accept seed argument. |
| `scripts/run_all_experiments.py` | Add `--seed` CLI flag; pass through to dataset loaders, train/test split, Optuna sampler, XGBoost. |
| `scripts/run_ablation_pruning.py` | Same. |
| `core/RunData.py` | `RunPipeline(seed=s)` already accepts seed; verify it propagates to Optuna sampler in `XGBoostModel.train`. |
| `core/XGBoostModel.py` | `seed` param already exists; verify propagated to `optuna.samplers.TPESampler(seed=s)` and to `xgb.train` `params['seed']`. |
| `baselines/*/run_*.py` | Each runner already takes a `--seed` arg; submit jobs varying it. |

### 3.2. Output schema

Every result CSV gains a `seed` column. Per-dataset CSVs become a long
table: one row per (combo, pruning_mode, seed). The aggregator script
groups by (dataset, combo, pruning_mode, method) and computes mean,
std, n.

### 3.3. New aggregator script

`scripts/aggregate_seeds.py` that reads:
- `results/all_results_<DS>.csv`
- `baselines/results/{baseline,pufe,ocds,emli,gbdt_il}_results_<DS>.csv`

…and writes:
- `results/mean_std_per_dataset.csv` (mean, std, n_seeds for each
  dataset × method × population × pruning_mode).
- `results/framework_comparison_meanstd.txt` (rebuilt comparison file
  with `0.7944 ± 0.0028` style cells).
- `results/wilcoxon_paired.txt` (paired Wilcoxon Ours-vs-each-method
  with $n = 50$ pairs from 10 datasets × 5 seeds).

---

## 4. Cluster Compute Estimate

Existing single-seed times (from cluster logs):

| Method | Mean per-dataset wall-clock | × 10 datasets × 5 seeds | Parallel on 32-CPU node |
|---|---|---|---|
| Our framework + ablation | ~4 min | 200 min | ~10 min (50-way parallel) |
| AdaptiveXGBoost | 30 s | 25 min | <5 min |
| PUFE | 25 s | 21 min | <5 min |
| OCDS | 35 s | 29 min | <5 min |
| EMLI | 80 s | 67 min | ~10 min |
| GBDT-IL | 30 s | 25 min | <5 min |

If we submit each (dataset, method, seed) as its own SLURM job
(50 jobs × 6 method-sets = **300 jobs**, all 32 CPU), the bottleneck
is queue time, not compute. Total wall-clock: **2-4 hours** including queue.

Storage: ~50 KB per CSV × 300 jobs ≈ 15 MB.

---

## 5. Statistical Analysis

For each (dataset × method) cell:
- $\bar{\mathrm{AUC}}$ = mean over 5 seeds
- $s_{\mathrm{AUC}}$ = sample std
- 95 % CI = $\bar{\mathrm{AUC}} \pm t_{4,0.975} \cdot s_{\mathrm{AUC}} / \sqrt{5}$

For each (dataset × method-pair) cell:
- Paired difference $d_s = \mathrm{AUC}^{\text{Ours}}_{d,s} - \mathrm{AUC}^{\text{M}}_{d,s}$
- Aggregate over $50 = 10 \times 5$ pairs.
- **Paired Wilcoxon signed-rank test, one-sided** ($H_1$: Ours > M).
- Holm–Bonferroni correction across the 6 comparisons (BL Combined,
  AdaXGB, PUFE, OCDS, EMLI, GBDT-IL).

For the headline "10 / 10 wins" claim: replace with
"Ours' mean exceeds each baseline's mean on K / 10 datasets" and report
how many of those K beats are robust (mean gap > 2 × max(std)).

---

## 6. Paper Updates

### Sections to edit

1. **§5.2 Experimental Setup** — add a paragraph: *"All experiments
   are repeated for `S = 5` seeds (42–46); each seed jointly controls
   the train/test split, Optuna sampler, XGBoost RNG, and structured-
   null injection. Reported AUCs are mean ± std over the 5 seeds."*

2. **Tables 6 (results), 7 (ablation), and §6.3 baseline tables** —
   replace each cell with `mean ± std` (4 decimal places). Bold the
   best mean per row; tie-break by lower std.

3. **§6.1 main results** — update Wilcoxon line: was *"Wilcoxon
   $W = 55$, $p = 0.001$"* on 10 datasets, becomes *"paired Wilcoxon
   on $50 = 10 \times 5$ seed-pairs, $W = ?$, $p = ?$, Holm-corrected
   significant against k baselines"*.

4. **§6.3 baseline summary table** — `Δ vs Ours` becomes mean of the
   per-pair gap with its own std.

5. **Abstract & Conclusion** — replace "+0.041 over closest competitor"
   with the seed-averaged version, e.g. *"+0.041 ± 0.008"*. Drop the
   "10 / 10" universal-win claim if it doesn't survive seed variance.

### Sentences likely to change phrasing

- *"All 10 datasets confirm successful cross-phase knowledge transfer"*
  → may become *"K / 10 datasets… with the remaining within seed-noise"*.
- *"Our framework wins on every single dataset against every baseline"*
  → *"Ours' mean exceeds every baseline's mean on K / 10 datasets;
  paired tests significant for J / 6 method comparisons"*.

We should NOT update these sentences in advance — wait for the actual
seed numbers and rewrite from data.

---

## 7. Phased Execution

| Phase | Task | Owner | Effort |
|---|---|---|---|
| 1 | Verify `seed` plumbing in `core/RunData.py`, `XGBoostModel.py`, all runners. Patch any hard-coded `42`. | code | 1 hour |
| 2 | Submit 50 SLURM jobs for our framework (10 datasets × 5 seeds). | cluster | 2 hours queue + run |
| 3 | Submit 5 × 50 = 250 baseline jobs in parallel. | cluster | 2 hours queue + run |
| 4 | Pull all CSVs, write `aggregate_seeds.py`, generate the mean±std comparison file. | code | 2 hours |
| 5 | Re-compute Wilcoxon + Holm correction. | code | 30 min |
| 6 | Update paper tables and prose with the new numbers. | paper | 2 hours |
| 7 | Commit, push, regenerate PDF. | git | 15 min |

**Total wall-clock: 1 working day** if everything runs cleanly.

---

## 8. Risks and Mitigations

1. **Seed plumbing leaks** — some sub-routine ignores the seed
   argument and gives the same result every time. *Mitigation:* before
   running 250 jobs, run our framework on one dataset with seeds
   {42, 43} and confirm the test AUCs differ by more than $10^{-4}$.

2. **A baseline's tuning is so sensitive that the std swallows the
   mean gap to ours.** This is actually the reviewer's prediction.
   *Mitigation:* if it happens, it is the honest finding — we report
   it. We may then bundle this with item #5 (small Optuna sweep per
   baseline) so the gap-to-tuned-baseline is the comparison that
   matters.

3. **Cluster queue contention** — 300 jobs at once may saturate the
   shared queue. *Mitigation:* batch by method, throttle to ~50
   jobs at a time.

4. **The 10/10 universal-win claim may not survive.** That is fine —
   the paper becomes more honest. We replace it with a robust
   "K-of-10" formulation supported by Wilcoxon.

---

## 9. Decision Points (need user confirmation before launching)

- [ ] Confirm `S = 5`. (Could also do 3 for quick turnaround or 10 for
  more statistical power.)
- [ ] Confirm we keep the per-dataset feature allocation fixed across
  seeds (the BO that picked it ran at seed 42; varying it across seeds
  would mix up the selection-bias issue).
- [ ] Confirm the aggregator output format (`mean ± std` cells, or
  `mean (std)` notation).
- [ ] Do we re-tune baselines under Optuna in the same sweep
  (combining items #4 and #5), or seed-only first then tuning?

Once these are confirmed I can start with Phase 1 (verify seed
plumbing) and submit jobs.
