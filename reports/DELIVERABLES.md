# Deliverables index (v2 — daily regression)

Mapped to the Project-2 requirements.

## Primary deliverables

| item | path |
|---|---|
| **Jupyter notebook (the program)** | [../notebooks/APS1052_BTC_regression.ipynb](../notebooks/APS1052_BTC_regression.ipynb) |
| Slides (32 pages, Marp, with speaker notes) | [SLIDES.md](SLIDES.md) |
| Final report | [FINAL_REPORT.md](FINAL_REPORT.md) |
| Data dictionary / feature table | [../outputs/tables/data_dictionary.csv](../outputs/tables/data_dictionary.csv) |

## Data required to run

- Fixed data base: `raw_data/` (teammate export, read-only).
- Assembled modelling dataset: `data/processed/btc_daily_dataset.parquet` (+ .csv)
  — rebuilt by `aps.datasets.assemble()`.

## Per-stage reports & artifacts

| stage | report | key artifacts |
|---|---|---|
| p1 audit + EDA | [P1_DATA.md](../outputs/eda/P1_DATA.md) | data_dictionary.csv, corr, summary_stats |
| p2 models | [P2_MODELS.md](../outputs/models/P2_MODELS.md) | p2_val_metrics.csv, feature_ranking.csv, val_predictions.parquet, selected_model.json |
| p3 test + SHAP | [P3_TEST_SHAP.md](../outputs/models/P3_TEST_SHAP.md) | p3_test_metrics.csv, p3_shap_importance.csv, test_predictions.parquet |
| p4 trading | [P4_TRADING.md](../outputs/backtest/P4_TRADING.md) | p4_test_trade_metrics.csv, p4_test_ledger_base.csv, p4_diagnostics.json |

## Required metrics — where to find them

| requirement | location |
|---|---|
| MAE, Spearman RHO, directional accuracy by quantile (val + test) | p2/p3 reports, `p3_test_quantile_diracc.csv` |
| SHAP feature importance (test) | `outputs/figures/p3_shap_importance.png`, `p3_shap_importance.csv` |
| Test equity curve vs buy-and-hold | `outputs/figures/p4_test_equity.png` |
| White Reality Check p-value | `p4_diagnostics.json` (`reality_check`) |
| Monte-Carlo permutation p-value (Profit Factor) | `p4_diagnostics.json` (`mc_permutation_pf`) |
| Sharpe / Profit Factor / CAGR | `p4_test_trade_metrics.csv`, P4 report |

## Environment

- [../requirements.txt](../requirements.txt): Scikit-Learn 1.6.1, TensorFlow 2.16.2 /
  Keras 3.10, LightGBM 4.6.0, SHAP 0.49.1 (no PyTorch, no XGBoost).
- Python 3.9.10, global seed 1052 (`aps.config.SEED`).

## Code

- Library `src/aps/`: `config, datasets, data, scaling, features, models,
  nn_keras, evaluate, backtest, stats_tests, runner, plotting`.
- Pipelines `pipelines/p1_data … p4_trading` (one entry point per stage).
- Run order: `p1_data → p2_models → p3_test_shap → p4_trading`; or run the notebook.

## Archived

- v1 classification framework: git tag `v1-classification-archive`.
