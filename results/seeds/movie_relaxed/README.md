# MovieAugV2 5-seed sweep with IL_MIN_STRATUM=50

## Why a separate sweep

MovieAugV2 was the only one of the 10 datasets the original 5-seed
Table-4 sweep deterministically skipped (the 1.9%-positive class
crossed with the locked-in 6-feature ext combo gives a (label×has_ext)
stratum below 100 samples on every seed). To produce a multi-seed
estimate for the row, we relax the per-stratum sample floor in
`core/RunData.py` from 100 to 50 via the `IL_MIN_STRATUM` env var
(commit f6dd1d8). All five seeds (42–46) then complete normally.

## Artefacts

* `movie_seedsweep.csv` — long format (one row per seed × mode).
* `movie_meanstd.csv` — per-mode aggregate.
* `seed_<S>/ablation/MovieAugV2/{base,combined,extended_*}.json` —
  trained model artefacts per seed.

## 5-seed result (optuna mode)

|              | mean ± std         |
|--------------|--------------------|
| Our base     | 0.7277 ± 0.0114    |
| BL base      | 0.7276 ± 0.0072    |
| Our ext      | 0.9306 ± 0.0042    |
| BL ext       | 0.9316 ± 0.0049    |
| Wimp         | -0.0007 ± 0.0017   |
| n_no avg     | 5340               |
| n_ext avg    | 17154              |

The mean Wimp is −0.0007 (slightly negative), with std 0.0017 —
parity with the data-sharing baseline, consistent with the rest of
the seed-aware Table 4. The has_extended split here (n_no=5340,
n_ext=17154 → 23.7% Null) differs from the historical single-seed
run (10830/11664 → 48.1% Null) because the structured null-injection
RNG sequence resolves differently after the seed-plumbing refactor.
