# Deliverables index

Everything the APS1052 project produces, mapped to the plan's deliverable list.

## Written deliverables

| item | path |
|---|---|
| Final report | [FINAL_REPORT.md](FINAL_REPORT.md) |
| Slides (30 pages, Marp) | [SLIDES.md](SLIDES.md) |
| Data dictionary | [../outputs/tables/data_dictionary.csv](../outputs/tables/data_dictionary.csv) |
| Feature / label definitions | [../docs/feature_selection.md](../docs/feature_selection.md) |
| Project plan | [../docs/project_plan.md](../docs/project_plan.md) |

## Per-stage reports (`outputs/`)

| stage | report |
|---|---|
| 1 audit | [audit/STAGE1_AUDIT.md](../outputs/audit/STAGE1_AUDIT.md) |
| 1b EDA | [eda/STAGE1B_EDA.md](../outputs/eda/STAGE1B_EDA.md) |
| 2 baselines | [models/STAGE2_BASELINES.md](../outputs/models/STAGE2_BASELINES.md) |
| 3 models | [models/STAGE3_MODELS.md](../outputs/models/STAGE3_MODELS.md) |
| 4 selection | [models/STAGE4_SELECTION.md](../outputs/models/STAGE4_SELECTION.md) |
| 5 signals | [backtest/STAGE5_SIGNALS.md](../outputs/backtest/STAGE5_SIGNALS.md) |
| 6 backtest | [backtest/STAGE6_BACKTEST.md](../outputs/backtest/STAGE6_BACKTEST.md) |
| 7 statistics | [stats/STAGE7_STATS.md](../outputs/stats/STAGE7_STATS.md) |
| 7b barrier sensitivity | [backtest/STAGE7B_BARRIER_SENSITIVITY.md](../outputs/backtest/STAGE7B_BARRIER_SENSITIVITY.md) |
| 8 final test | [backtest/STAGE8_FINAL_TEST.md](../outputs/backtest/STAGE8_FINAL_TEST.md) |

## Data / result files

| item | path |
|---|---|
| Model predictions (per barrier) | `outputs/models/predictions_{0.03,0.02,0.01}.parquet` |
| Classification metrics | `outputs/models/s2_baseline_metrics.csv`, `s3_model_metrics.csv` |
| Trade ledgers (base cost) | `outputs/backtest/s6_ledger_{train,val}_base.csv` |
| Backtest metrics | `outputs/backtest/s6_backtest_metrics.csv` |
| Barrier sensitivity | `outputs/backtest/barrier_sensitivity.csv` |
| Final test summary | `outputs/backtest/s8_final_test.csv` |
| Frozen strategy config | `outputs/backtest/selected_config.json` |
| Statistical-test outputs | `outputs/stats/s7_*.csv` |
| 1-min path tables | `outputs/backtest/path_table_{...}.parquet` |
| Figures | `outputs/figures/*.png` |

## Environment

| item | path |
|---|---|
| Pinned requirements | [../requirements.txt](../requirements.txt) |
| Package config (editable install) | [../pyproject.toml](../pyproject.toml) |
| Python | 3.9.10 |
| Global random seed | 1052 (`aps.config.SEED`) |

## Code

- Library: `src/aps/` — `config, data, audit, eda, evaluate, models, nn_models,
  signals, pathdata, backtest, stats_tests, experiment, plotting`.
- Pipelines: `pipelines/s1_audit … s8_final_test` (one entry point per stage).
- Frozen data pipeline: `scripts/build_dataset.py`.
