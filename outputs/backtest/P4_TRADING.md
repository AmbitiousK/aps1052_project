# Pipeline 4 — Quantile Trading, Test Equity & Diagnostics

Selected model **linear**. Long top quantile (Q5), short bottom (Q1); edges from train predictions. Test set, base cost 0.05%/side.

## Test trade metrics by cost scenario

| scenario   |   n_trades |   n_long |   n_short |   total_return |   cagr |   sharpe |   profit_factor |   max_drawdown |   win_rate |
|:-----------|-----------:|---------:|----------:|---------------:|-------:|---------:|----------------:|---------------:|-----------:|
| zero       |         65 |       23 |        19 |         0.1355 | 0.3739 |   1.3122 |          1.4235 |         0.1035 |     0.5238 |
| base       |         65 |       23 |        19 |         0.0965 | 0.2591 |   0.9899 |          1.2977 |         0.1105 |     0.5238 |
| high       |         65 |       23 |        19 |         0.0589 | 0.1538 |   0.6657 |          1.1874 |         0.1203 |     0.5238 |


## Statistical diagnostics (test, base cost)

- Sharpe **0.99**, Profit Factor **1.30**, CAGR **25.91%**.

- Bootstrap 95% CI total return [-14.59%, 40.58%] (point 9.65%); P(total<=0)=0.241.

- Bootstrap 95% CI profit factor [0.69, 2.64].

- **Monte-Carlo permutation p (Profit Factor) = 0.194** (PF_obs 1.30).

- **White's Reality Check p = 0.171** (5 candidate models).
