# Stage 1b — Exploratory Data Analysis

Dataset: main ±3% · statistics on **train** (35365 rows) unless noted.

## 1. Summary statistics (train)

| feature           |   count |    mean |     std |     min |      1% |      5% |     25% |     50% |     75% |     95% |     99% |     max |    skew |   kurtosis |
|:------------------|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|-----------:|
| ret_1h            |   35365 |  0      |  0.0068 | -0.0938 | -0.0205 | -0.0097 | -0.0025 |  0.0001 |  0.0026 |  0.0098 |  0.0198 |  0.1161 | -0.239  |    15.1875 |
| ret_24h           |   35365 |  0.001  |  0.033  | -0.2281 | -0.0983 | -0.051  | -0.0134 |  0.0008 |  0.0161 |  0.0536 |  0.0944 |  0.2096 | -0.1677 |     3.9809 |
| close_vs_sma24    |   35365 |  0.0006 |  0.0183 | -0.1472 | -0.0545 | -0.0284 | -0.007  |  0.0004 |  0.0084 |  0.0297 |  0.0539 |  0.1255 | -0.1638 |     4.9651 |
| rvol_24h          |   35365 |  0.0058 |  0.0035 |  0.0005 |  0.001  |  0.0017 |  0.0035 |  0.0051 |  0.0071 |  0.0124 |  0.0188 |  0.0345 |  1.9889 |     7.055  |
| range_24h         |   35365 |  0.0482 |  0.0332 |  0.0028 |  0.0077 |  0.0135 |  0.027  |  0.0403 |  0.0596 |  0.109  |  0.1803 |  0.3855 |  2.3966 |    10.06   |
| vol_ratio_24h     |   35365 |  1.0195 |  0.679  |  0.1671 |  0.2912 |  0.3938 |  0.6114 |  0.833  |  1.1957 |  2.2656 |  3.6759 | 13.1056 |  3.171  |    19.3826 |
| rsi_14            |   35365 | 50.7484 | 12.1783 |  5.5606 | 22.4477 | 30.4092 | 42.9855 | 50.6981 | 58.3387 | 71.481  | 80.346  | 91.8422 |  0.0559 |     0.1308 |
| taker_buy_frac_1h |   35365 |  0.491  |  0.0448 |  0.1726 |  0.3708 |  0.4157 |  0.4661 |  0.4931 |  0.5158 |  0.5637 |  0.6023 |  0.7335 | -0.167  |     1.5489 |
| oi_chg_4h         |   35365 |  0.0003 |  0.0222 | -0.359  | -0.0662 | -0.0322 | -0.009  |  0.0007 |  0.0103 |  0.0333 |  0.059  |  0.162  | -1.2704 |    17.7205 |
| funding_rate      |   35365 |  0.0001 |  0.0002 | -0.0012 | -0.0002 | -0.0001 |  0      |  0.0001 |  0.0001 |  0.0005 |  0.0012 |  0.0025 |  3.6542 |    20.4428 |

## 2. Feature correlation (train)

Top-10 most correlated feature pairs (Pearson):

| feat_a         | feat_b            |   abs_corr |   corr |
|:---------------|:------------------|-----------:|-------:|
| range_24h      | rvol_24h          |      0.882 |  0.882 |
| close_vs_sma24 | rsi_14            |      0.868 |  0.868 |
| close_vs_sma24 | ret_24h           |      0.861 |  0.861 |
| ret_24h        | rsi_14            |      0.792 |  0.792 |
| ret_1h         | taker_buy_frac_1h |      0.392 |  0.392 |
| close_vs_sma24 | ret_1h            |      0.351 |  0.351 |
| ret_1h         | rsi_14            |      0.33  |  0.33  |
| rsi_14         | taker_buy_frac_1h |      0.222 |  0.222 |
| ret_1h         | ret_24h           |      0.207 |  0.207 |
| funding_rate   | rvol_24h          |      0.195 |  0.195 |


Highest absolute pairwise correlation: **0.882**. 3 pair(s) exceed 0.8, all within a single economic dimension — a *volatility* cluster (`range_24h`↔`rvol_24h`) and a *trend/stretch* cluster (`close_vs_sma24`↔`rsi_14`↔`ret_24h`). These are the same information seen through different lenses, not meaningless duplication; each feature still has a distinct one-sentence meaning, so no feature is dropped. Interpretation caveat noted for the linear model.

## 3. Redundancy & informativeness (train)

VIF (variance inflation factor):

| feature           |   VIF |
|:------------------|------:|
| close_vs_sma24    | 6.584 |
| range_24h         | 4.781 |
| rvol_24h          | 4.701 |
| rsi_14            | 4.362 |
| ret_24h           | 4.361 |
| ret_1h            | 1.36  |
| taker_buy_frac_1h | 1.212 |
| funding_rate      | 1.074 |
| oi_chg_4h         | 1.031 |
| vol_ratio_24h     | 1.017 |


Mutual information with 3-class label:

| feature           |   mutual_info |
|:------------------|--------------:|
| rvol_24h          |       0.01533 |
| range_24h         |       0.01446 |
| close_vs_sma24    |       0.01229 |
| ret_24h           |       0.01122 |
| ret_1h            |       0.00902 |
| funding_rate      |       0.00888 |
| rsi_14            |       0.0045  |
| oi_chg_4h         |       0.00373 |
| vol_ratio_24h     |       0.00266 |
| taker_buy_frac_1h |       4e-05   |


Max VIF: **6.58** (inspect).

## 4. Feature means by label class (train)

| feature           |   mean_label_-1 |   mean_label_0 |   mean_label_1 |
|:------------------|----------------:|---------------:|---------------:|
| ret_1h            |        -0.00483 |        9e-05   |       -0.00105 |
| ret_24h           |        -0.02875 |        0.00145 |       -0.02483 |
| close_vs_sma24    |        -0.0175  |        0.00078 |       -0.01106 |
| rvol_24h          |         0.012   |        0.00572 |        0.01283 |
| range_24h         |         0.11056 |        0.04729 |        0.11837 |
| vol_ratio_24h     |         1.49642 |        1.01343 |        1.33863 |
| rsi_14            |        43.708   |       50.8244  |       48.4255  |
| taker_buy_frac_1h |         0.48431 |        0.49099 |        0.49447 |
| oi_chg_4h         |        -0.00549 |        0.00038 |        0.00189 |
| funding_rate      |         0.00029 |        0.00012 |        0.00015 |

## 5. Distribution shift (KS statistic vs train)

| feature           |   ks_train_val |   ks_train_test |
|:------------------|---------------:|----------------:|
| funding_rate      |         0.1932 |          0.4886 |
| rvol_24h          |         0.2103 |          0.2551 |
| range_24h         |         0.193  |          0.219  |
| taker_buy_frac_1h |         0.1711 |          0.1686 |
| oi_chg_4h         |         0.0961 |          0.1259 |
| ret_24h           |         0.0579 |          0.0852 |
| vol_ratio_24h     |         0.0902 |          0.0818 |
| close_vs_sma24    |         0.0498 |          0.0798 |
| ret_1h            |         0.0439 |          0.0582 |
| rsi_14            |         0.0288 |          0.0501 |


Largest train→test shift: **funding_rate** (KS=0.489).

## 6. Extreme-event time clustering

Monthly extreme-event ratio: min=0.0000, max=0.1075, mean=0.0112. Non-stationarity is visible in `s1b_extreme_timeline.png`.
