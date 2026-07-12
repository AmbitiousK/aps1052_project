# Stage 6 — Barrier-Aligned Backtest (frozen strategy)

Model **logistic** · tau **0.70** · TP=SL=±3% · entry next-minute open · max hold 1h · train + validation (test frozen).

## Financial metrics by split and cost scenario

| split   | scenario   |   n_trades |   n_long |   n_short |   total_net_return |   total_gross_return |   sharpe |   sortino |   max_drawdown |   win_rate |   profit_factor |   avg_hold_min |
|:--------|:-----------|-----------:|---------:|----------:|-------------------:|---------------------:|---------:|----------:|---------------:|-----------:|----------------:|---------------:|
| train   | zero       |        286 |       78 |       208 |            -0.0611 |              -0.0611 |  -0.0685 |   -0.1161 |         0.194  |     0.4441 |          0.9783 |        55.1678 |
| train   | base       |        286 |       78 |       208 |            -0.2948 |              -0.0611 |  -0.6511 |   -1.0955 |         0.3194 |     0.4231 |          0.8131 |        55.1678 |
| train   | high       |        286 |       78 |       208 |            -0.4705 |              -0.0611 |  -1.2337 |   -2.047  |         0.4854 |     0.3881 |          0.678  |        55.1678 |
| val     | zero       |         32 |        1 |        31 |             0.0179 |               0.0179 |   0.5044 |    1.1642 |         0.0394 |     0.4688 |          1.2519 |        60      |
| val     | base       |         32 |        1 |        31 |            -0.0141 |               0.0179 |  -0.3667 |   -0.7866 |         0.0461 |     0.375  |          0.8546 |        60      |
| val     | high       |         32 |        1 |        31 |            -0.0452 |               0.0179 |  -1.2377 |   -2.5797 |         0.0529 |     0.3438 |          0.5987 |        60      |


**Validation (base cost): net return -1.41%, gross 1.79%, Sharpe -0.37, 32 trades.**


At ±3% the strategy is not profitable net of costs: extreme events are extremely rare in the 2024-25 validation regime and signal precision is low, so trading costs dominate. Note that gross returns are less negative (and positive at the highest tau), i.e. the edge is real but too small to survive costs at this barrier. Barrier sensitivity (±1%/±2%) is examined in Stage 7+.
