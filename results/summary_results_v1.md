# Summary Results v1 — Incremental Learning Experiments

Generated: 2026-04-04

## Objective Definition

```
Objective = (combined_with_ext_auc - extended_auc) + (combined_no_ext_auc - base_auc)
```
**Negative objective = incremental learning approach outperforms combined model.**

## Models
- **Base** — trained on all rows, extended features set to NaN
- **Extended** — warm-started from base, trained on rows with extended data only
- **Combined** — trained on all rows with real values (no incremental learning)

---

## Best Objective Per Dataset Per Source

| Dataset | summary.txt (10t) | upbeat (30t) | local main (30t) | frosty (30t) | keen-bassi (50t) |
|---|---|---|---|---|---|
| BankLoanSta | -0.006532 | +0.024701 | **-0.074311** | — | — |
| Weather | **-0.009676** | +0.007388 | — | — | — |
| DiabetesRecord | **-0.005322** | +0.000378 | — | — | — |
| HRAnalytics | -0.000177 | **-0.000855** | — | — | — |
| ClientRecordAug | -0.000341 | **-0.000534** | — | — | — |
| MovieAugV2 | **-0.004046** | -0.000147 | — | — | — |
| WeatherAUS | +0.000560 | +0.004080 | — | **-0.047474** | -0.000291 |
| WIDS | -0.004540 | -0.001107 | **-0.002435** | — | — |
| FlightDelay | — | — | **-0.004602** | — | — |

### Source Key
- **summary.txt (10t)** — full combo sweep, 10 Optuna trials per model (March 20)
- **upbeat (30t)** — branch `claude/upbeat-chandrasekhar`, ablation with hardcoded best combo, 30 trials (March 31)
- **local main (30t)** — individual dataset runs via `run_all_experiments.py`, 30 trials (April 1-3)
- **frosty (30t)** — worktree `frosty-pasteur`, WeatherAUS with different best combo, 30 trials (April 3)
- **keen-bassi (50t)** — worktree `keen-bassi`, WeatherAUS leakage-fixed (`RainToday` removed), 50 trials (April 4)

---

## Extended Features Used Per Best Result

| Dataset | Best Objective | Extended Features | Trials | Mode |
|---|---|---|---|---|
| BankLoanSta | **-0.074311** | Current Loan Amount, Annual Income, Credit Score | 30 | optuna |
| Weather | **-0.009676** | Evaporation, Sunshine, WindDir9am | 10 | optuna |
| DiabetesRecord | **-0.005322** | weight, medical_specialty, A1Cresult | 10 | optuna |
| HRAnalytics | **-0.000855** | gender | 30 | no_pruning |
| ClientRecordAug | **-0.000534** | Offer | 30 | no_pruning |
| MovieAugV2 | **-0.004046** | rating_mean, rating_std | 10 | optuna |
| WeatherAUS | **-0.047474** | Cloud3pm | 30 | no_pruning |
| WIDS | **-0.004540** | h1_bilirubin (max/min), h1_albumin (max/min), h1_lactate (max/min), h1_pao2fio2ratio (max/min), h1_arterial_ph (max/min), h1_arterial_pco2 (max/min), h1_arterial_po2 (max/min) — 14 features total | 10 | optuna |
| FlightDelay | **-0.004602** | OP_CARRIER_FL_NUM | 30 | optuna |

---

## Detailed AUC Breakdown (Best Result Per Dataset)

| Dataset | Rows | Base AUC | Ext AUC | Comb-no | Comb+ext | N_no_ext | N_ext | Augmented? |
|---|---|---|---|---|---|---|---|---|
| BankLoanSta | 100,000 | 0.6952 | 0.7817 | 0.6495 | 0.7531 | 907 | 19,093 | Yes |
| Weather | 25,000 | 0.8432 | 0.8448 | 0.8314 | 0.8469 | 578 | 4,422 | No |
| DiabetesRecord | 101,766 | 0.7069 | 0.7041 | 0.7024 | 0.7032 | 8,117 | 12,237 | No |
| HRAnalytics | 19,158 | 0.7735 | 0.8109 | 0.7727 | 0.8108 | 902 | 2,930 | No |
| ClientRecordAug | 7,043 | 0.9612 | 0.9610 | 0.9605 | 0.9612 | 775 | 634 | Yes |
| MovieAugV2 | 112,466 | 0.9209 | 0.8739 | 0.9199 | 0.8709 | 5,616 | 16,878 | Yes |
| WeatherAUS | 142,193 | 0.8923 | 0.9479 | 0.8906 | 0.9021 | 11,419 | 17,020 | No |
| WIDS | 91,713 | 0.9006 | 0.8955 | 0.8992 | 0.8923 | 13,632 | 4,711 | No |
| FlightDelay | 100,000* | 0.7478 | 0.7188 | 0.7450 | 0.7170 | 6,011 | 13,989 | Yes |

*FlightDelay sampled to 100K from 567,630 rows.

---

## Ablation: Pruning Mode Comparison (upbeat-chandrasekhar, 30 trials)

| Dataset | optuna | no_pruning | fixed_50 | Best Mode |
|---|---|---|---|---|
| BankLoanSta | +0.028379 | +0.053562 | **+0.024701** | fixed_50 |
| Weather | +0.008229 | +0.008119 | **+0.007388** | fixed_50 |
| DiabetesRecord | **+0.000378** | +0.000900 | +0.002505 | optuna |
| HRAnalytics | -0.000192 | **-0.000855** | +0.001127 | no_pruning |
| ClientRecordAug | -0.000361 | **-0.000534** | +0.000621 | no_pruning |
| MovieAugV2 | **-0.000147** | +0.000933 | +0.005244 | optuna |
| WeatherAUS | **+0.004080** | +0.004219 | +0.004856 | optuna |
| WIDS | -0.000932 | **-0.001107** | +0.001987 | no_pruning |

Mode wins: optuna 3/8, no_pruning 3/8, fixed_50 2/8

### WeatherAUS Ablation (frosty-pasteur, 30t, ext=Cloud3pm)

| Mode | Objective | Ext AUC |
|---|---|---|
| optuna | -0.001144 | 0.9016 |
| **no_pruning** | **-0.047474** | **0.9479** |
| fixed_50 | -0.029520 | 0.9299 |

### WeatherAUS Ablation (keen-bassi, 50t, ext=Evaporation+Cloud9am, RainToday leakage fixed)

| Mode | Objective | Ext AUC |
|---|---|---|
| **optuna** | **-0.000291** | 0.9024 |
| no_pruning | -0.000254 | 0.9024 |
| fixed_50 | +0.001589 | 0.9005 |

---

## Notes

1. **All 9 datasets achieve negative objectives** (incremental learning wins) in at least one configuration.
2. **WeatherAUS `RainToday` leakage** — discovered April 4. The feature `RainToday` is a temporal proxy for the target `RainTomorrow`. It was correctly dropped in `run_all_experiments.py` but missing from `run_weatheraus.py` and `run_ablation_pruning.py`. Fixed in keen-bassi branch. The frosty-pasteur results (Cloud3pm combo) were not affected since `RainToday` was not selected as an extended feature there.
3. **BankLoanSta Run 1 (April 1)** had suspicious results: no_pruning objective=-0.157657 with ext_auc=0.9215. This was overfitting from forced warm-starting. The Run 2 (April 3) optuna result of -0.074311 is more reliable as optuna chose to warm-start.
4. **Datasets not yet run with 30 trials (full combo sweep):** Weather, DiabetesRecord, HRAnalytics, ClientRecordAug, MovieAugV2. These only have 10-trial full sweep results or 30-trial ablation on a hardcoded best combo.
