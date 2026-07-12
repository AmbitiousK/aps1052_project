# Stage 5 — Trading Signals & Confidence-Threshold Calibration

Selected model: **logistic** · base cost 0.05%/side · entry = next-minute open · TP=SL=barrier · calibrated on validation.

## Validation tau sweep

| signal   |   n_signals |   coverage |   n_long |   n_short |   sig_precision |   sig_recall |   net_return |   gross_return |   sharpe |   win_rate |   profit_factor |   n_trades |
|:---------|------------:|-----------:|---------:|----------:|----------------:|-------------:|-------------:|---------------:|---------:|-----------:|----------------:|-----------:|
| argmax   |         856 |     0.113  |      461 |       395 |          0.0105 |       0.2727 |      -0.7    |        -0.2932 |  -5.112  |     0.396  |          0.6376 |        856 |
| tau=0.40 |         752 |     0.0992 |      400 |       352 |          0.012  |       0.2727 |      -0.6513 |        -0.2598 |  -4.846  |     0.3843 |          0.6308 |        752 |
| tau=0.50 |         317 |     0.0418 |      145 |       172 |          0.0095 |       0.0909 |      -0.4178 |        -0.2004 |  -3.7071 |     0.3785 |          0.5755 |        317 |
| tau=0.60 |         107 |     0.0141 |       33 |        74 |          0      |       0      |      -0.1455 |        -0.0489 |  -2.3183 |     0.3364 |          0.5552 |        107 |
| tau=0.70 |          32 |     0.0042 |        1 |        31 |          0      |       0      |      -0.0141 |         0.0179 |  -0.3667 |     0.375  |          0.8546 |         32 |


**Frozen operating point: tau = 0.70** (max validation net return with >= 20 trades).


The probabilities are over-confident (Stage 4), so tau is an empirical operating point, not a literal probability. It is frozen here and reused unchanged on the test set in Stage 8.
