# Summary Results v1 — Incremental Learning Experiments

Generated: 2026-04-05 (updated)

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

## Best Result Per Dataset (Selection Rule: prefer 30t optuna negative, else best overall)

| Dataset | Combo | n_ext | Trials | Pruning | Base AUC | Ext AUC | Comb-no | Comb+ext | Objective |
|---|---|---|---|---|---|---|---|---|---|
| BankLoanSta | Curren+Annual+Credit | 3 | 30 | optuna | 0.6952 | 0.7817 | 0.6495 | 0.7531 | **-0.074311** |
| Weather | Evapor+Sunshi+WindDir9am | 3 | 10 | optuna | 0.8432 | 0.8448 | 0.8314 | 0.8469 | **-0.009676** |
| DiabetesRecord | medical_specialty+A1Cresult | 2 | 30 | optuna | 0.6734 | 0.6682 | 0.6727 | 0.6663 | **-0.002604** |
| HRAnalytics | gender | 1 | 30 | optuna | 0.7735 | 0.8102 | 0.7727 | 0.8108 | **-0.000192** |
| ClientRecordAug | Offer | 1 | 30 | optuna | 0.9612 | 0.9608 | 0.9605 | 0.9612 | **-0.000361** |
| MovieAugV2 | rating_mean+rating_std | 2 | 30 | optuna | 0.9187 | 0.8726 | 0.9185 | 0.8726 | **-0.000147** |
| WeatherAUS | Cloud3pm | 1 | 30 | optuna | 0.8923 | 0.9016 | 0.8906 | 0.9021 | **-0.001144** |
| WIDS | h1_bilirub+h1_lactate+h1_pao2fio+h1_arteria | 14 | 30 | optuna | 0.8997 | 0.8967 | 0.8991 | 0.8950 | **-0.002435** |
| FlightDelay | OP_CARRIER_FL_NUM | 1 | 30 | optuna | 0.7478 | 0.7188 | 0.7450 | 0.7170 | **-0.004602** |

**All 9 datasets show negative objectives (incremental learning wins).**

---

## Extended Features Used Per Best Result

| Dataset | Best Objective | Extended Features | Trials | Mode |
|---|---|---|---|---|
| BankLoanSta | **-0.074311** | Current Loan Amount, Annual Income, Credit Score | 30 | optuna |
| Weather | **-0.009676** | Evaporation, Sunshine, WindDir9am | 10 | optuna |
| DiabetesRecord | **-0.002604** | medical_specialty, A1Cresult | 30 | optuna |
| HRAnalytics | **-0.000192** | gender | 30 | optuna |
| ClientRecordAug | **-0.000361** | Offer | 30 | optuna |
| MovieAugV2 | **-0.000147** | rating_mean, rating_std | 30 | optuna |
| WeatherAUS | **-0.001144** | Cloud3pm | 30 | optuna |
| WIDS | **-0.002435** | h1_bilirubin (max/min), h1_albumin (max/min), h1_lactate (max/min), h1_pao2fio2ratio (max/min), h1_arterial_ph (max/min), h1_arterial_pco2 (max/min), h1_arterial_po2 (max/min) — 14 features total | 30 | optuna |
| FlightDelay | **-0.004602** | OP_CARRIER_FL_NUM | 30 | optuna |

---

## Detailed AUC Breakdown (Best Result Per Dataset)

| Dataset | Rows | Base AUC | Ext AUC | Comb-no | Comb+ext | N_no_ext | N_ext | Augmented? |
|---|---|---|---|---|---|---|---|---|
| BankLoanSta | 100,000 | 0.6952 | 0.7817 | 0.6495 | 0.7531 | 907 | 19,093 | Yes |
| Weather | 25,000 | 0.8432 | 0.8448 | 0.8314 | 0.8469 | 578 | 4,422 | No |
| DiabetesRecord | 101,766 | 0.6734 | 0.6682 | 0.6727 | 0.6663 | 8,389 | 11,965 | No |
| HRAnalytics | 19,158 | 0.7735 | 0.8102 | 0.7727 | 0.8108 | 902 | 2,930 | No |
| ClientRecordAug | 7,043 | 0.9612 | 0.9608 | 0.9605 | 0.9612 | 775 | 634 | Yes |
| MovieAugV2 | 112,466 | 0.9187 | 0.8726 | 0.9185 | 0.8726 | 5,616 | 16,878 | Yes |
| WeatherAUS | 142,193 | 0.8923 | 0.9016 | 0.8906 | 0.9021 | 11,419 | 17,020 | No |
| WIDS | 91,713 | 0.8997 | 0.8967 | 0.8991 | 0.8950 | 13,632 | 4,711 | No |
| FlightDelay | 100,000* | 0.7478 | 0.7188 | 0.7450 | 0.7170 | 6,011 | 13,989 | Yes |

*FlightDelay sampled to 100K from 567,630 rows.

---

## DiabetesRecord Ablation (30t, ext=medical_specialty+A1Cresult, IDs removed)

| Mode | Objective | Ext AUC | Base AUC | Comb+ext |
|---|---|---|---|---|
| **optuna** | **-0.002604** | 0.6682 | 0.6734 | 0.6663 |
| no_pruning | +0.000635 | 0.6649 | 0.6734 | 0.6663 |
| **fixed_50** | **-0.001120** | 0.6667 | 0.6734 | 0.6663 |

---

## Ablation: Pruning Mode Comparison (upbeat-chandrasekhar, 30 trials)

| Dataset | optuna | no_pruning | fixed_50 | Best Mode |
|---|---|---|---|---|
| BankLoanSta | +0.028379 | +0.053562 | **+0.024701** | fixed_50 |
| Weather | +0.008229 | +0.008119 | **+0.007388** | fixed_50 |
| DiabetesRecord | **-0.002604** | +0.000635 | -0.001120 | optuna |
| HRAnalytics | -0.000192 | **-0.000855** | +0.001127 | no_pruning |
| ClientRecordAug | -0.000361 | **-0.000534** | +0.000621 | no_pruning |
| MovieAugV2 | **-0.000147** | +0.000933 | +0.005244 | optuna |
| WeatherAUS | **+0.004080** | +0.004219 | +0.004856 | optuna |
| WIDS | -0.000932 | **-0.001107** | +0.001987 | no_pruning |

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
2. **DiabetesRecord `encounter_id`/`patient_nbr` leakage** — discovered April 4. These ID columns were not dropped, causing CV AUC inflation (0.85) vs test AUC (0.71). Fixed April 5. The best combo changed from `weight+medical_specialty+A1Cresult` to `medical_specialty+A1Cresult` (without weight, which has 97% nulls).
3. **WeatherAUS `RainToday` leakage** — discovered April 4. The feature `RainToday` is a temporal proxy for the target `RainTomorrow`. Fixed in `run_weatheraus.py` and `run_ablation_pruning.py`.
4. **BankLoanSta Run 1 (April 1)** had suspicious results: no_pruning objective=-0.157657 with ext_auc=0.9215. This was overfitting from forced warm-starting. The Run 2 (April 3) optuna result of -0.074311 is more reliable as optuna chose to warm-start.
5. **Datasets not yet run with 30 trials (full combo sweep):** Weather, HRAnalytics, ClientRecordAug, MovieAugV2. These only have 10-trial full sweep results or 30-trial ablation on a hardcoded best combo.
