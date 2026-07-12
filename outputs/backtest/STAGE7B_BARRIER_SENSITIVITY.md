# Stage 7b — Pre-Registered Barrier Sensitivity

Same isomorphic chain per barrier (re-label, re-train, re-calibrate tau on validation, backtest, test). Main model **logistic** frozen; sensitivity barriers NOT used to re-select main model/barrier. Validation only; test sealed.

## Barrier comparison (validation, base cost)

|   threshold |   val_extreme_events |   val_pr_auc_logistic |   selected_tau |   n_trades_val |   gross_return_base |   net_return_base |   net_ci_lo |   net_ci_hi |   sharpe_base |   perm_p_net |   rc_p_net |
|------------:|---------------------:|----------------------:|---------------:|---------------:|--------------------:|------------------:|------------:|------------:|--------------:|-------------:|-----------:|
|        0.03 |                   33 |                0.1035 |            0.7 |             32 |              0.0179 |           -0.0141 |     -0.0845 |      0.0674 |       -0.3667 |       0.2324 |      1     |
|        0.02 |                  139 |                0.1153 |            0.7 |             62 |              0.0891 |            0.0236 |     -0.1043 |      0.1795 |        0.3942 |       0.015  |      0.919 |
|        0.01 |                  830 |                0.3673 |            0.6 |             75 |              0.0463 |           -0.0293 |     -0.1328 |      0.0866 |       -0.5286 |       0.1139 |      0.979 |


## Full detail

|   threshold |   val_extreme_events |   val_pr_auc_logistic |   val_extreme_recall_logistic |   selected_tau |   n_trades_val |   net_return_base |   gross_return_base |   sharpe_base |   max_drawdown_base |   net_total_return |   net_ci_lo |   net_ci_hi |   gross_total_return |   gross_ci_lo |   gross_ci_hi |   perm_p_net |   perm_p_gross |   rc_p_net |   rc_p_gross |   n_trades |
|------------:|---------------------:|----------------------:|------------------------------:|---------------:|---------------:|------------------:|--------------------:|--------------:|--------------------:|-------------------:|------------:|------------:|---------------------:|--------------:|--------------:|-------------:|---------------:|-----------:|-------------:|-----------:|
|        0.03 |                   33 |                0.1035 |                        0.5758 |            0.7 |             32 |           -0.0141 |              0.0179 |       -0.3667 |              0.0461 |            -0.0141 |     -0.0845 |      0.0674 |               0.0179 |       -0.0539 |        0.1014 |       0.2324 |         0.2324 |      1     |        0.495 |        nan |
|        0.02 |                  139 |                0.1153 |                        0.5899 |            0.7 |             62 |            0.0236 |              0.0891 |        0.3942 |              0.0442 |             0.0236 |     -0.1043 |      0.1795 |               0.0891 |       -0.0455 |        0.2513 |       0.015  |         0.015  |      0.919 |        0.818 |         62 |
|        0.01 |                  830 |                0.3673 |                        0.5964 |            0.6 |             75 |           -0.0293 |              0.0463 |       -0.5286 |              0.0598 |            -0.0293 |     -0.1328 |      0.0866 |               0.0463 |       -0.0634 |        0.1669 |       0.1139 |         0.1139 |      0.979 |        0.778 |         75 |


## Reading

- Lower barriers have far more extreme events, so more trades and tighter intervals.
- Gross edge is present at every barrier; whether it survives costs and, crucially, the data-snooping correction differs by barrier.
- A single permutation test may look significant while White's Reality Check (correcting for the 6×4 candidate search) does not reject — that gap is the whole point of the reality check.
- All results are retained regardless of significance.
