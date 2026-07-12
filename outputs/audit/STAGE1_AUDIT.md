# Stage 1 — Data Integrity Audit

Main threshold: **±3%** · features: **10** · seed: **1052**

## 1. Data dictionary

| column            | role                        | dtype   |   n_missing |          min |        max |       mean |        std | description                                 |
|:------------------|:----------------------------|:--------|------------:|-------------:|-----------:|-----------:|-----------:|:--------------------------------------------|
| close             | price (backtest only)       | float64 |           0 | 9946         |  1.26e+05  |  5.388e+04 |  2.968e+04 |                                             |
| ret_1h            | feature                     | float64 |           0 |   -0.09381   |  0.1161    |  2.554e-05 |  0.006291  | Past 1h return                              |
| ret_24h           | feature                     | float64 |           0 |   -0.2281    |  0.2096    |  0.0007228 |  0.0304    | Past 24h return                             |
| close_vs_sma24    | feature                     | float64 |           0 |   -0.1472    |  0.1255    |  0.0003957 |  0.01691   | Close deviation from 24h SMA (%)            |
| rsi_14            | feature                     | float64 |           0 |    5.561     | 91.84      | 50.63      | 12.12      | RSI(14) on hourly bars                      |
| rvol_24h          | feature                     | float64 |           0 |    0.0004546 |  0.03449   |  0.005386  |  0.003255  | Realized vol: std of last 24 hourly returns |
| range_24h         | feature                     | float64 |           0 |    0.002773  |  0.3855    |  0.04432   |  0.03064   | (24h high - 24h low) / close                |
| vol_ratio_24h     | feature                     | float64 |           0 |    0.08402   | 13.11      |  1.023     |  0.7195    | Last-1h volume / 24h average volume         |
| taker_buy_frac_1h | feature                     | float64 |           0 |    0.1131    |  0.8169    |  0.4893    |  0.05849   | Taker buy volume fraction over last 1h      |
| oi_chg_4h         | feature                     | float64 |           0 |   -0.359     |  0.162     |  0.0002712 |  0.01992   | Futures open-interest 4h change rate        |
| funding_rate      | feature                     | float64 |           0 |   -0.001192  |  0.00249   |  0.0001021 |  0.0001948 | Latest funding rate (8h)                    |
| ret_fwd_1h        | forward stat (relabel only) | float64 |           0 |   -0.08954   |  0.1232    |  4.664e-05 |  0.006291  |                                             |
| max_up_1h         | forward stat (relabel only) | float64 |           0 |   -0.000179  |  0.1343    |  0.004117  |  0.005051  |                                             |
| max_dn_1h         | forward stat (relabel only) | float64 |           0 |   -0.1643    |  0.0002747 | -0.004292  |  0.005679  |                                             |
| label             | target                      | int64   |           0 |   -1         |  1         | -0.002296  |  0.1058    |                                             |

## 2. Timestamp audit

|   threshold |   n_rows | start               | end                 | is_monotonic_increasing   | has_duplicates   |   n_duplicates | all_on_the_hour   |   expected_rows_if_no_gaps |   n_missing_hours |   coverage_pct |   n_gaps |   max_gap_hours |
|------------:|---------:|:--------------------|:--------------------|:--------------------------|:-----------------|---------------:|:------------------|---------------------------:|------------------:|---------------:|---------:|----------------:|
|        0.03 |    50522 | 2020-09-01 04:00:00 | 2026-06-20 02:00:00 | True                      | False            |              0 | True              |                      50831 |               309 |          99.39 |       18 |              30 |
|        0.02 |    50522 | 2020-09-01 04:00:00 | 2026-06-20 02:00:00 | True                      | False            |              0 | True              |                      50831 |               309 |          99.39 |       18 |              30 |
|        0.01 |    50522 | 2020-09-01 04:00:00 | 2026-06-20 02:00:00 | True                      | False            |              0 | True              |                      50831 |               309 |          99.39 |       18 |              30 |


Top gaps (main ±3%, hours): 30, 30, 30, 28, 28, 28, 28, 27, 27, 27

## 3. Missing-value audit (main split)

| feature           |   train_missing |   val_missing |   test_missing |   total_missing |
|:------------------|----------------:|--------------:|---------------:|----------------:|
| ret_1h            |               0 |             0 |              0 |               0 |
| ret_24h           |               0 |             0 |              0 |               0 |
| close_vs_sma24    |               0 |             0 |              0 |               0 |
| rvol_24h          |               0 |             0 |              0 |               0 |
| range_24h         |               0 |             0 |              0 |               0 |
| vol_ratio_24h     |               0 |             0 |              0 |               0 |
| rsi_14            |               0 |             0 |              0 |               0 |
| taker_buy_frac_1h |               0 |             0 |              0 |               0 |
| oi_chg_4h         |               0 |             0 |              0 |               0 |
| funding_rate      |               0 |             0 |              0 |               0 |


Total missing across all features/splits: **0**

## 4. Class-distribution audit

|   threshold | split   |     n |   class_-1 |   class_0 |   class_+1 |   share_-1 |   share_0 |   share_+1 |   extreme_ratio |
|------------:|:--------|------:|-----------:|----------:|-----------:|-----------:|----------:|-----------:|----------------:|
|      0.0300 | train   | 35365 |        309 |     34853 |        203 |     0.0087 |    0.9855 |     0.0057 |          0.0145 |
|      0.0300 | val     |  7578 |         18 |      7545 |         15 |     0.0024 |    0.9956 |     0.0020 |          0.0044 |
|      0.0300 | test    |  7579 |         14 |      7558 |          7 |     0.0018 |    0.9972 |     0.0009 |          0.0028 |
|      0.0200 | train   | 35365 |        903 |     33731 |        731 |     0.0255 |    0.9538 |     0.0207 |          0.0462 |
|      0.0200 | val     |  7578 |         75 |      7439 |         64 |     0.0099 |    0.9817 |     0.0084 |          0.0183 |
|      0.0200 | test    |  7579 |         60 |      7474 |         45 |     0.0079 |    0.9861 |     0.0059 |          0.0139 |
|      0.0100 | train   | 35365 |       3772 |     28135 |       3458 |     0.1067 |    0.7956 |     0.0978 |          0.2044 |
|      0.0100 | val     |  7578 |        423 |      6748 |        407 |     0.0558 |    0.8905 |     0.0537 |          0.1095 |
|      0.0100 | test    |  7579 |        416 |      6813 |        350 |     0.0549 |    0.8989 |     0.0462 |          0.1011 |

## 5. Label-integrity audit

|   threshold |     n |   violations_pos_label_no_up_touch |   violations_neg_label_no_dn_touch |   violations_flat_exactly_one_touch |   total_violations |   same_minute_double_touch |
|------------:|------:|-----------------------------------:|-----------------------------------:|------------------------------------:|-------------------:|---------------------------:|
|        0.03 | 50522 |                                  0 |                                  0 |                                   0 |                  0 |                          0 |
|        0.02 | 50522 |                                  0 |                                  0 |                                   0 |                  0 |                          2 |
|        0.01 | 50522 |                                  0 |                                  0 |                                   0 |                  0 |                          3 |


Label integrity: **PASS — 0 violations**

## 6. Chronological split boundaries (main dataset)

| split   |     n | start               | end                 |
|:--------|------:|:--------------------|:--------------------|
| train   | 35365 | 2020-09-01 04:00:00 | 2024-09-26 13:00:00 |
| val     |  7578 | 2024-09-26 14:00:00 | 2025-08-08 07:00:00 |
| test    |  7579 | 2025-08-08 08:00:00 | 2026-06-20 02:00:00 |
