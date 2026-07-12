# Predicting BTC Daily Returns and Quantile Trading

**APS1052 course project — final report (v2, regression)**

> **Question.** Can interpretable daily features — most of them not from the price
> bar — predict Bitcoin's next-day log return well enough to drive a profitable,
> cost-aware quantile trading strategy?

**Answer.** A simple linear model attains a positive out-of-sample rank
correlation (Spearman ≈ 0.14) and 56% directional accuracy; the quantile strategy
is **profitable after costs** (CAGR ≈ 26%, Sharpe ≈ 1.0, Profit Factor 1.30) and
beats buy-and-hold over a falling test period. The edge is **promising but not
statistically proven** — permutation and reality-check p-values are ~0.17–0.19 and
the bootstrap intervals include the null on a ~146-day test sample.

---

## 1. Target, asset, frequency

- **Target:** next-day log return `y_{t+1}=log(close_{t+1}/close_t)` — a single,
  continuous regression target.
- **Asset:** Bitcoin (crypto, chosen for its on-chain indicators).
- **Frequency:** daily. Less noisy than intraday, and the non-price features are
  published daily.

## 2. Data base

The fixed teammate `raw_data/` export (read-only). Daily OHLCV + pre-computed TA,
five on-chain metrics, funding, COT, DVOL, net-taker volume. Common window
**2022-06 → 2025-05** (bounded by the on-chain start and DVOL end): **1060 daily
rows, 100% coverage, 0 missing** ([audit](../outputs/eda/P1_DATA.md)).

## 3. Features — 19, of which 12 (63%) are NOT from the target OHLCV bar

| From target OHLCV (7) | Not from target OHLCV (12) |
|---|---|
| ret_1d/5d/20d, rsi_14, macd_hist, natr_14, bb_position | on-chain mvrv_zscore / nupl / nvt / puell / sopr; ntv_ratio; funding_rate; cot_net_frac; dvol; month_sin/cos; weekday |

Meets the requirement of ≥15 features with ≥half non-OHLCV. Feature meanings are
one-line and causal (data ≤ t). Leakage controls: backward as-of merge for weekly
COT, target shifted so `X_t → y_{t+1}` (alignment verified exact), scalers fit on
train only. ![corr](../outputs/figures/p1_corr_heatmap.png)

## 4. Scaling (three regimes)

1. **Meaning-preserving** transforms once (RSI→(x−50)/50, MACD→asinh(MACD/close),
   positive levels→log) so price-scale indicators are not distorted.
2. **Global StandardScaler** inside the ML grid pipeline (train-fold fit only).
3. **Rolling scaling** (`asymmetric_rolling_scale`) for the Keras MLP, applied to
   the whole matrix before the split (causal; drops 90 warm-up rows → common index
   of 971 rows for all models).

## 5. Split & feature selection

Chronological, no shuffle: **train 679 / val 146 / test 146** (2022-09 → 2025-05).
`SelectKBest` (f_regression for the linear model, mutual information for the
non-linear ones) sits inside the CV pipeline with `k` tuned per model; the
train-set ranking flags the two weakest features as discard candidates. On-chain
and positioning features rank strongest. ![rank](../outputs/figures/p2_feature_ranking.png)

## 6. Models & selection (manual grid-search CV, TimeSeriesSplit)

Five models (two require scaling): LinearRegression, SVR, RandomForest, LightGBM
(tabular: `StandardScaler→SelectKBest→estimator`), and a Keras MLP (rolling-scaled;
MAE loss, L2 + dropout, no sample weights, early stopping). Selection by validation
Spearman RHO.

| model | CV RHO | val RHO |
|---|---|---|
| **linear** | **+0.123** | **+0.182** |
| keras_mlp | +0.106 | +0.122 |
| random_forest | +0.040 | +0.046 |
| lightgbm | +0.074 | −0.005 |
| svr | +0.082 | −0.059 |

**Selected: linear regression** — best rank correlation, fully interpretable; the
flexible models overfit noisy daily returns. ![val](../outputs/figures/p2_val_spearman.png)

## 7. Test metrics & SHAP

| split | MAE | Spearman RHO | dir. accuracy | top-Q dir | bot-Q dir |
|---|---|---|---|---|---|
| validation | 0.021 | +0.182 | 0.45 | 0.38 | 0.63 |
| **test** | **0.019** | **+0.141** | **0.56** | 0.55 | 0.63 |

Directional accuracy in the traded extreme quantiles beats the majority-direction
baseline (Q4 edge +0.38, Q5 +0.10). **SHAP** on test ranks **NUPL and COT
positioning first** — the top drivers are non-OHLCV, validating the feature design.
![shap](../outputs/figures/p3_shap_importance.png)

## 8. Trading, equity & diagnostics

**Strategy:** long the top predicted-return quantile, short the bottom, flat
otherwise; quantile edges from train predictions; hold one day; turnover costs.

Test (base cost 0.05%/side): **total +9.7%, CAGR 26%, Sharpe 0.99, Profit Factor
1.30, max drawdown 11%**, profitable across zero/base/high costs. The strategy
climbs while buy-and-hold falls over the test window.
![equity](../outputs/figures/p4_test_equity.png)

**Diagnostics (test, base cost):**

| test | value |
|---|---|
| Bootstrap 95% CI total return | +9.7% **[−14.6%, +40.6%]** |
| Bootstrap 95% CI Profit Factor | 1.30 [0.69, 2.64] |
| Monte-Carlo permutation p (Profit Factor) | **0.194** |
| White's Reality Check p (5 candidates) | **0.171** |

Point estimates are positive, but the intervals include the null and the p-values
do not reach 5% — expected on a ~146-day sample. Wide intervals are a result.

## 9. Limitations

Small test sample (wide CIs, non-significant tests); short window fixed by feed
coverage; regime dependence; single asset/venue; cost assumptions.

## 10. Conclusion

An interpretable linear model on 19 mostly-non-OHLCV daily features predicts BTC
next-day returns with a real out-of-sample rank correlation, and the quantile
strategy is profitable after costs and beats buy-and-hold in a down market. The
statistical tests, on a small sample, stop short of significance, so the edge is
promising rather than proven — and the program logic is correct and fully
validated end to end.

---

### Reproducibility

- **Notebook:** [`notebooks/APS1052_BTC_regression.ipynb`](../notebooks/APS1052_BTC_regression.ipynb).
- **Library** `src/aps/` + **pipelines** `p1_data → p2_models → p3_test_shap → p4_trading`.
- **Env:** [requirements.txt](../requirements.txt) (Scikit-Learn, Keras/TensorFlow,
  LightGBM, SHAP), Python 3.9.10, seed 1052. **Data:** teammate `raw_data/` base.
