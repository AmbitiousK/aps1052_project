# Pipeline 1 — Data Audit & EDA (daily)

**1060 days**, 2022-06-11 → 2025-05-05, coverage 100.0%. 19 features (12 not from the target OHLCV bar), 0 missing, 0 duplicate dates.

## Data dictionary

| feature      | from_target_ohlcv   | group       |        min |       max |     mean |      std | description                                                 |
|:-------------|:--------------------|:------------|-----------:|----------:|---------:|---------:|:------------------------------------------------------------|
| ret_1d       | True                | price_trend |    -0.167  |    0.1123 |   0.0011 |   0.0272 | 1-day log return                                            |
| ret_5d       | True                | price_trend |    -0.3317 |    0.2195 |   0.0054 |   0.0618 | 5-day log return                                            |
| ret_20d      | True                | price_trend |    -0.4404 |    0.3666 |   0.0211 |   0.1245 | 20-day log return                                           |
| rsi_14       | True                | momentum    |    18.6355 |   89.3983 |  53.0933 |  13.4219 | RSI(14)                                                     |
| macd_hist    | True                | momentum    | -1438.37   | 1865.82   |  13.3703 | 440.459  | MACD histogram (asinh-normalized by close)                  |
| natr_14      | True                | volatility  |     1.726  |   10.5974 |   3.9307 |   1.1831 | Normalized ATR(14)                                          |
| bb_position  | True                | volatility  |    -1.6973 |    1.5543 |   0.084  |   0.6458 | Position within Bollinger Bands                             |
| mvrv_zscore  | False               | onchain     |    -0.3599 |    3.3543 |   1.1719 |   1.0136 | On-chain MVRV Z-score                                       |
| nupl         | False               | onchain     |    -0.3168 |    0.6402 |   0.3239 |   0.2497 | On-chain net unrealized P/L                                 |
| nvt          | False               | onchain     |    74.42   |  312.57   | 190.342  |  44.077  | On-chain NVT ratio                                          |
| puell        | False               | onchain     |     0.347  |    2.434  |   1.0279 |   0.3809 | On-chain Puell multiple                                     |
| sopr         | False               | onchain     |     0.871  |    1.1728 |   1.0064 |   0.0178 | On-chain SOPR                                               |
| ntv_ratio    | False               | flow        |    -0.0842 |    0.071  |  -0.0038 |   0.0241 | Net taker volume imbalance                                  |
| funding_rate | False               | leverage    |    -0.0008 |    0.0007 |   0.0001 |   0.0001 | Perp funding rate (daily mean)                              |
| cot_net_frac | False               | positioning |    -0.8979 |   -0.0919 |  -0.5821 |   0.2078 | COT leveraged-money net position fraction (weekly, ffilled) |
| dvol         | False               | options     |    32.38   |  119.65   |  57.6679 |  12.8934 | Deribit implied volatility (DVOL)                           |
| month_sin    | False               | calendar    |    -1      |    1      |  -0.017  |   0.7135 | Calendar month (sin)                                        |
| month_cos    | False               | calendar    |    -1      |    1      |   0.0287 |   0.7005 | Calendar month (cos)                                        |
| weekday      | False               | calendar    |     0      |    6      |   3.0019 |   2.0033 | Day of week (0=Mon)                                         |


## Split boundaries

- train: 742 rows, 2022-06-11 → 2024-06-21
- val: 159 rows, 2024-06-22 → 2024-11-27
- test: 159 rows, 2024-11-28 → 2025-05-05


## Target (y_logret_fwd1)

mean 0.00116, std 0.02716, skew -0.173, min -0.1670, max 0.1123.
