# Stage 7 — Statistical Validation (±3% main, validation)

Frozen: model **logistic**, tau **0.70**, base cost 0.05%/side, TP=SL=±3%, next-minute open. **32 trades.** No tuning; all results kept.

## 1. Trade-level bootstrap CI (95%, 10k resamples)

| basis   | metric              |   point |    ci_lo |    ci_hi |
|:--------|:--------------------|--------:|---------:|---------:|
| net     | total_return        | -0.0141 |  -0.0845 |   0.0674 |
| net     | mean_return         | -0.0004 |  -0.0027 |   0.0021 |
| net     | sharpe              | -0.0602 |  -0.5    |   0.2724 |
| net     | profit_factor       |  0.8546 |   0.2862 |   2.0803 |
| net     | win_rate            |  0.375  |   0.2188 |   0.5312 |
| net     | p_total_return_gt_0 |  0.6561 | nan      | nan      |
| gross   | total_return        |  0.0179 |  -0.0539 |   0.1014 |
| gross   | mean_return         |  0.0006 |  -0.0017 |   0.0031 |
| gross   | sharpe              |  0.0828 |  -0.3079 |   0.3991 |
| gross   | profit_factor       |  1.2519 |   0.4512 |   3.1019 |
| gross   | win_rate            |  0.4688 |   0.3125 |   0.6562 |
| gross   | p_total_return_gt_0 |  0.3291 | nan      | nan      |

## 2. Stationary-bootstrap CI on hourly returns (block=24h, 5k)

| basis   |   point_total_return |   ci_lo |   ci_hi |   p_total_le_0 |
|:--------|---------------------:|--------:|--------:|---------------:|
| net     |              -0.0135 | -0.0782 |  0.055  |         0.6584 |
| gross   |               0.0185 | -0.0487 |  0.0924 |         0.3068 |

## 3. Circular-shift permutation test (2000 shifts)

|   obs_net_total_return |   obs_gross_total_return |   p_net |   p_gross |    B |
|-----------------------:|-------------------------:|--------:|----------:|-----:|
|                -0.0141 |                   0.0179 |  0.2324 |    0.2324 | 2000 |


Observed net total return -0.0141 → p_net = **0.232**; gross 0.0179 → p_gross = **0.232**.

## 4. White's Reality Check (24 candidates: 6 models x 4 tau, benchmark always-flat)

| basis   |   V_obs |   best_strategy_mean |   reality_check_p |   n_candidates |    B |
|:--------|--------:|---------------------:|------------------:|---------------:|-----:|
| net     |  0      |                    0 |             1     |             24 | 2000 |
| gross   |  0.0018 |                    0 |             0.495 |             24 | 2000 |


Reality-check p-value (net) = **1.000**, (gross) = **0.495**.

## Conclusion

With only 32 trades the intervals are wide by construction. The gross edge is small and positive; after realistic costs the net total-return CI includes zero and the permutation / reality-check p-values do not reject the null. The evidence supports a *weak predictive signal* but not *tradeable economic value* at the ±3% barrier. Wide intervals are a result, not a failure.
