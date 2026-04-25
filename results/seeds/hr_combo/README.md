# HRAnalytics candidate-combo screen — negative result

Same protocol as the DiabetesRecord screen
(see `../diab_combo/README.md`).

Pre-registered candidate set, chosen by natural-missingness rate:
  c1_CompanyType    company_type
  c2_CompanySize    company_size           (Table 4 baseline)
  c3_Gender         gender
  c4_MajorDisc      major_discipline
  c5_TypeAndSize    company_type + company_size
  c6_TypeAndGender  company_type + gender

5 seeds × 6 combos = 30 jobs. Decision rule:
  qualify iff lower bound of 5-seed 95 % CI on weighted improvement > 0.

## Result: no candidate qualifies.

| Combo            | Wimp mean ± std    | 95% CI                | Decision |
|------------------|--------------------|----------------------|----------|
| c3_Gender        | +0.00070 ± 0.00221 | (-0.00204, +0.00345) | fail     |
| c4_MajorDisc     | -0.00042 ± 0.00097 | (-0.00163, +0.00078) | fail     |
| c6_TypeAndGender | -0.00129 ± 0.00118 | (-0.00276, +0.00017) | fail     |
| c2_CompanySize * | -0.00159 ± 0.00437 | (-0.00701, +0.00384) | fail     |
| c1_CompanyType   | -0.00189 ± 0.00316 | (-0.00581, +0.00203) | fail     |
| c5_TypeAndSize   | -0.00853 ± 0.00264 | (-0.01180, -0.00525) | fail     |

\* current Table 4 combo for HRAnalytics.

## Decision

The current Table 4 row for HRAnalytics stays unchanged. The screen is
filed as supplementary evidence that we honestly looked for an
alternative and none cleared the same threshold the DiabetesRecord
swap did.

## Artefacts

* `hr_combo_screen.csv` — long format (one row per combo×seed).
* `hr_combo_summary.csv` — per-combo aggregate with CI bounds.
* `<combo>/seed_<S>/ablation/HRAnalytics/` — model artefacts per cell.
