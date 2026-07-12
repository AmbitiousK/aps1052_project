# Stage 4 — Model Selection (validation)

Selection on validation only (test frozen). Primary criterion: extreme-event PR-AUC.

## Selection table (ranked by extreme PR-AUC)

| model         |   extreme_pr_auc |   macro_f1 |   balanced_accuracy |     mcc |   extreme_recall |   extreme_precision |   log_loss |   brier |
|:--------------|-----------------:|-----------:|--------------------:|--------:|-----------------:|--------------------:|-----------:|--------:|
| logistic      |           0.1035 |     0.3269 |              0.4815 |  0.0698 |           0.5758 |              0.0222 |     0.495  |  0.2373 |
| mlp           |           0.0446 |     0.333  |              0.4939 |  0.0842 |           0.6364 |              0.0282 |     0.6552 |  0.3451 |
| random_forest |           0.0369 |     0.3326 |              0.3332 | -0.0008 |           0      |              0      |     0.043  |  0.0111 |
| xgboost       |           0.0315 |     0.3376 |              0.3476 |  0.0292 |           0.0909 |              0.03   |     0.0875 |  0.0337 |
| lightgbm      |           0.0283 |     0.3323 |              0.3328 | -0.002  |           0      |              0      |     0.0359 |  0.0121 |
| lstm          |           0.0258 |     0.3331 |              0.457  |  0.0732 |           0.5152 |              0.0261 |     0.3287 |  0.1539 |


**Selected model: `logistic`.**


Rationale: `logistic` gives the highest extreme-event PR-AUC (0.1035, ~24× the random-prior rate) with strong extreme recall, and it is the most interpretable model — coefficients map directly to feature effects. The tree/boosting models overfit the high-volatility training regime (2021) and collapse to near-zero PR-AUC on the low-volatility validation period, a clear regime-shift effect. Note the tree models show the *lowest* log loss only because they confidently predict the 99.6%-majority flat class; log loss is misleading here, which is why PR-AUC drives the choice.
