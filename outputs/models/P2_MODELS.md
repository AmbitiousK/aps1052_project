# Pipeline 2 — Models & Validation Selection

Common index 971 rows; split 679/146/146. Tabular models: StandardScaler+SelectKBest inside TimeSeriesSplit grid search. Keras MLP: rolling-scaled features. Selection by validation Spearman RHO.

## Validation metrics (ranked by Spearman RHO)

| model         |   cv_spearman |   spearman_rho |    mae |   dir_acc |   top_q_dir_acc |   top_q_edge |   bot_q_dir_acc |   bot_q_edge |
|:--------------|--------------:|---------------:|-------:|----------:|----------------:|-------------:|----------------:|-------------:|
| linear        |        0.1231 |         0.1824 | 0.0211 |    0.4521 |          0.3793 |      -0.1724 |          0.6333 |       0.2667 |
| keras_mlp     |        0.1058 |         0.1224 | 0.0198 |    0.5205 |          0.4828 |       0      |          0.6333 |       0.2667 |
| random_forest |        0.0396 |         0.0461 | 0.0202 |    0.5411 |          0.4828 |       0      |          0.5    |       0      |
| lightgbm      |        0.0737 |        -0.0045 | 0.0253 |    0.4452 |          0.4828 |       0      |          0.5    |       0      |
| svr           |        0.0818 |        -0.0589 | 0.0326 |    0.4726 |          0.4483 |       0      |          0.4    |      -0.2    |


**Selected model: `linear`.**


## Feature ranking (train; f_regression + mutual information)

| feature      |   f_score |   mutual_info |   f_rank |   mi_rank |   avg_rank |
|:-------------|----------:|--------------:|---------:|----------:|-----------:|
| weekday      |    2.1392 |        0.0636 |        2 |         2 |        2   |
| ret_1d       |    1.0053 |        0.0744 |        4 |         1 |        2.5 |
| month_sin    |    0.8907 |        0.0472 |        5 |         5 |        5   |
| month_cos    |    1.6934 |        0.021  |        3 |         8 |        5.5 |
| rsi_14       |    0.3078 |        0.0277 |        8 |         6 |        7   |
| nvt          |    0.4832 |        0.0174 |        6 |         9 |        7.5 |
| mvrv_zscore  |    0.154  |        0.0252 |       10 |         7 |        8.5 |
| puell        |    4.0781 |        0      |        1 |        16 |        8.5 |
| ret_5d       |    0.0298 |        0.0574 |       15 |         3 |        9   |
| ntv_ratio    |    0.351  |        0.0034 |        7 |        12 |        9.5 |
| dvol         |    0.0165 |        0.0501 |       17 |         4 |       10.5 |
| bb_position  |    0.0512 |        0.0141 |       14 |        10 |       12   |
| sopr         |    0.2905 |        0      |        9 |        16 |       12.5 |
| natr_14      |    0.0229 |        0.013  |       16 |        11 |       13.5 |
| cot_net_frac |    0.1405 |        0      |       11 |        16 |       13.5 |
| ret_20d      |    0.0713 |        0      |       12 |        16 |       14   |
| nupl         |    0.0567 |        0      |       13 |        16 |       14.5 |
| macd_hist    |    0.0148 |        0      |       18 |        16 |       17   |
| funding_rate |    0.0021 |        0      |       19 |        16 |       17.5 |


Weakest features by average rank: **macd_hist, funding_rate** — SelectKBest inside the CV already drops weak features per model (tuned `k`); these two are the first candidates to discard.
