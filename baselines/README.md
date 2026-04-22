# Baselines

External baselines used to compare against our incremental learning framework.

Each subfolder is a single baseline, self-contained except for two shared pieces
taken from the main project:

1. **Data preparation** — via `scripts/prepare_datasets.py::prepare_wids()` and
   `prepare_client_record()`. Same preprocessing, feature engineering, and
   structured null injection as our framework.
2. **Train/test split** — via `scripts/prepare_datasets.py::split_train_test()`.
   Same stratified split by `(label, has_extended)` as `core/RunData.py`.

The same `base_features` and `ext_features` are used for every baseline, so AUC
on `test_with_extended` and `test_without_extended` is directly comparable to
our framework's `ext_auc` and `base_auc`.

## Baselines

| Folder | Paper / Source | Method |
| --- | --- | --- |
| `adaptive_xgboost/` | Montiel et al., [Adaptive XGBoost for Evolving Data Streams](https://github.com/jacobmontiel/AdaptiveXGBoostClassifier), IJCNN 2020 | Streaming XGBoost ensemble with replace/push window, optional ADWIN drift detection |
| `pufe/` | Hou, Zhang, Zhou, [Prediction with Unpredictable Feature Evolution](https://arxiv.org/abs/1904.12171), IEEE TNNLS 2021 | Two OGD logistic predictors (base-only, base⊕ext with recovered ext) Hedge-ensembled by cumulative log-loss |

## Running

From the project root:

```bash
# AdaptiveXGBoost on both datasets (default config)
python baselines/adaptive_xgboost/run_adaptive_xgboost.py

# Just WIDS, bigger ensemble
python baselines/adaptive_xgboost/run_adaptive_xgboost.py \
    --datasets WIDS --n_estimators 100 --max_window_size 2000
```

Results are appended to `baselines/results/baseline_results.csv`.
