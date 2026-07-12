# Pipeline 3 — Test Metrics & SHAP

Selected model **linear** retrained on train; evaluated on validation and the sealed test set.

## Regression metrics

| split   |    mae |   spearman_rho |   dir_acc |   top_q_dir_acc |   top_q_edge |   bot_q_dir_acc |   bot_q_edge |
|:--------|-------:|---------------:|----------:|----------------:|-------------:|----------------:|-------------:|
| val     | 0.0211 |         0.1824 |    0.4521 |          0.3793 |      -0.1724 |          0.6333 |       0.2667 |
| test    | 0.0185 |         0.1414 |    0.5616 |          0.5517 |       0.1034 |          0.6333 |       0      |


## Test directional accuracy by predicted-return quantile

|   quantile |   n |   mean_pred |   mean_actual |   up_rate |   model_dir_acc |   baseline_dir_acc |   edge_vs_baseline |
|-----------:|----:|------------:|--------------:|----------:|----------------:|-------------------:|-------------------:|
|          1 |  30 |     -0.0041 |       -0.0054 |    0.3667 |          0.6333 |             0.6333 |             0      |
|          2 |  29 |     -0.0004 |        0.0039 |    0.5517 |          0.6207 |             0.4483 |             0.1724 |
|          3 |  29 |      0.0016 |       -0.0099 |    0.3103 |          0.3103 |             0.6897 |            -0.3793 |
|          4 |  29 |      0.004  |        0.0069 |    0.6897 |          0.6897 |             0.3103 |             0.3793 |
|          5 |  29 |      0.0079 |        0.0032 |    0.5517 |          0.5517 |             0.4483 |             0.1034 |


## SHAP feature importance (test)

| feature      |   mean_abs_shap |
|:-------------|----------------:|
| nupl         |         0.00794 |
| cot_net_frac |         0.00706 |
| rsi_14       |         0.00505 |
| ret_1d       |         0.00281 |
| puell        |         0.00258 |
| ret_20d      |         0.00239 |
| month_sin    |         0.00227 |
| ntv_ratio    |         0.00212 |
| bb_position  |         0.00182 |
| nvt          |         0.00179 |
| natr_14      |         0.00141 |
| mvrv_zscore  |         0.00118 |
| weekday      |         0.00103 |
| dvol         |         0.00079 |
| macd_hist    |         0.0007  |
| ret_5d       |         0.00063 |
| funding_rate |         0.0006  |
| month_cos    |         0.00054 |
| sopr         |         0.00011 |
