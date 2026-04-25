# WeatherAUS candidate-combo screen — negative result

Same protocol as DiabetesRecord and HRAnalytics screens.

Pre-registered candidates by natural-missingness rate:
  c1_Sunshine       Sunshine
  c2_Evap           Evaporation
  c3_Cloud3         Cloud3pm
  c4_Cloud9         Cloud9am
  c5_EvapCloud9     Evaporation + Cloud9am  (Table 4 baseline)
  c6_SunshineEvap   Sunshine + Evaporation

5 seeds × 6 combos = 30 jobs. Decision rule:
  qualify iff lower bound of 5-seed 95% CI on weighted improvement > 0.

## Result: no candidate qualifies.

| Combo            | Wimp mean ± std    | 95% CI               | Decision |
|------------------|--------------------|---------------------|----------|
| c4_Cloud9        | -0.00021 ± 0.00049 | (-0.00082, +0.00040) | fail    |
| c2_Evap          | -0.00032 ± 0.00057 | (-0.00102, +0.00039) | fail    |
| c5_EvapCloud9 *  | -0.00053 ± 0.00080 | (-0.00153, +0.00046) | fail    |
| c3_Cloud3        | -0.00064 ± 0.00085 | (-0.00169, +0.00042) | fail    |
| c1_Sunshine      | -0.00145 ± 0.00096 | (-0.00264, -0.00025) | fail    |
| c6_SunshineEvap  | -0.00170 ± 0.00072 | (-0.00259, -0.00080) | fail    |

\* current Table 4 combo for WeatherAUS.

## Decision

The current Table 4 row for WeatherAUS stays unchanged. The screen is
filed as supplementary evidence that we honestly looked. WeatherAUS
sits at parity-or-slightly-below for the framework on every reasonable
ext-feature choice in the candidate set.

## Artefacts

* `weatherAUS_combo_screen.csv` — long format (one row per combo×seed).
* `weatherAUS_combo_summary.csv` — per-combo aggregate with CI bounds.
* `<combo>/seed_<S>/ablation/WeatherAUS/` — model artefacts per cell.
