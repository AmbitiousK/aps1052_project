# Stage 8 — Final Out-of-Sample Test

Test set unsealed once per barrier. Frozen: model **logistic**, next-minute-open entry, base cost 0.05%/side, per-barrier tau from validation. **±3% is the headline; ±2%/±1% are pre-registered sensitivity.** No tuning.

## Test summary (all barriers)

|   threshold |   tau |   test_events |   pr_auc |   macro_f1 |   extreme_recall |   n_trades |   gross_return |   net_return_base |   net_ci_lo |   net_ci_hi |   sharpe_base |   perm_p_net |
|------------:|------:|--------------:|---------:|-----------:|-----------------:|-----------:|---------------:|------------------:|------------:|------------:|--------------:|-------------:|
|        0.03 |   0.7 |            21 |   0.0272 |     0.3333 |           0.4286 |         46 |         0.0756 |            0.0273 |     -0.0972 |      0.171  |        0.4656 |       0.017  |
|        0.02 |   0.7 |           105 |   0.0882 |     0.352  |           0.4381 |         67 |         0.0619 |           -0.0069 |     -0.1323 |      0.1351 |       -0.0709 |       0.071  |
|        0.01 |   0.6 |           766 |   0.3    |     0.4321 |           0.4413 |         80 |         0.1535 |            0.0649 |     -0.0533 |      0.1967 |        1.1832 |       0.0005 |


## Main task ±3% — test confusion matrix (logistic, argmax)

|           |   pred_down |   pred_flat |   pred_up |
|:----------|------------:|------------:|----------:|
| true_down |           7 |           7 |         0 |
| true_flat |         352 |        6925 |       281 |
| true_up   |           1 |           5 |         1 |


## Main ±3% test result

- Classification: extreme PR-AUC **0.0272** (vs random prior ~0.0028), macro-F1 0.333, extreme recall 0.429.
- Trading (base cost): 46 trades, gross 7.56%, net **2.73%** (95% CI [-9.72%, 17.10%]), Sharpe 0.47.
- Cost sensitivity: zero 7.56% → base 2.73% → high -1.89%.
- Permutation p (net) = 0.017.


## Verdict

Out-of-sample, the frozen strategy produced **positive net returns at the ±3% (main, 2.73%, perm p=0.017) and ±1% (6.49%, Sharpe 1.18, perm p=0.0005) barriers**, so the signal-return alignment persists out of sample. Two cautions keep this from being a claim of robust profitability: (1) the bootstrap 95% CI on total net return **includes zero at every barrier** (wide, owing to limited trade counts), and (2) the validation-stage White Reality Check did **not** reject the null once the 6×4 candidate search was accounted for. The out-of-sample evidence is therefore *encouraging but not conclusive*: consistent with a weak, real predictive signal whose after-cost economic value cannot be established with confidence at this sample size. All barriers are reported regardless of sign or significance.
