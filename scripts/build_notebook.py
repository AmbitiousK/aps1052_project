"""Build the deliverable Jupyter notebook (nbformat) from the aps library.

The notebook is a runnable program: it imports the `aps` package and drives the
full pipeline (assemble -> audit/EDA -> split -> scaling -> feature selection ->
models with manual grid-search CV -> selection -> test + SHAP -> quantile trading
-> diagnostics), with markdown narrative between steps.

Run:  python scripts/build_notebook.py   (writes notebooks/APS1052_BTC_regression.ipynb)
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# ---------------------------------------------------------------- title
md(r"""
# BTC Daily Return Prediction & Quantile Trading — APS1052

**Target:** single asset (BTC), regression of the **next-day log return**
`y_{t+1} = log(close_{t+1}/close_t)`. **Frequency:** daily.

**Models (5, two require scaling):** LinearRegression, SVR, RandomForest, LightGBM
(tabular, `StandardScaler`+`SelectKBest` inside a `TimeSeriesSplit` grid search),
and a **Keras** MLP (rolling-scaled inputs). *PyTorch is not used.*

**Features (19, 12 not from the target OHLCV bar):** 7 OHLCV/TA features and 12
non-OHLCV features (on-chain MVRV/NUPL/NVT/Puell/SOPR, net-taker-volume, funding,
COT positioning, DVOL, calendar). The data base is the teammate `raw_data/` export.

**Deliverables produced here:** validation & test metrics (MAE, Profit Factor,
Spearman RHO, directional accuracy by quantile), SHAP feature importance, the test
equity curve vs buy-and-hold, and the equity diagnostics (White's Reality Check,
Monte-Carlo permutation p-value on the Profit Factor, Sharpe / Profit Factor / CAGR).
""")

code(r"""
import warnings, os
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, pandas as pd, matplotlib.pyplot as plt

from aps import config as C
from aps import datasets, features as F, models as M, evaluate as E
from aps import backtest as BT, stats_tests as ST
from aps.data import chronological_split
from aps.nn_keras import KerasMLPRegressor, manual_grid_search_nn
from aps.runner import fit_predict
np.random.seed(C.SEED)
print("features:", len(C.FEATURES), "| non-OHLCV:", len(C.NON_OHLCV_FEATURES))
""")

# ---------------------------------------------------------------- data
md(r"""
## 1. Data assembly (from the fixed `raw_data/` base)

Each feed is read read-only and merged on a daily date index. Weekly COT is merged
as-of backward (last report ≤ t). The target is strictly future (t → t+1), so
features at t predict `y_{t+1}` with no look-ahead.
""")
code(r"""
ds = datasets.assemble(save=True)
print(ds.shape, "|", ds.index[0].date(), "->", ds.index[-1].date())
ds.head()
""")

md(r"""
## 2. Audit & EDA

Daily coverage, missing values, target and feature distributions, and the feature
correlation structure.
""")
code(r"""
print("rows:", len(ds), "| missing:", int(ds.isna().sum().sum()),
      "| duplicate dates:", int(ds.index.duplicated().sum()))
ds[C.FEATURES + [C.TARGET]].describe().T.round(4)
""")
code(r"""
corr = ds[C.FEATURES].corr()
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(C.FEATURES))); ax.set_yticks(range(len(C.FEATURES)))
ax.set_xticklabels(C.FEATURES, rotation=90, fontsize=7); ax.set_yticklabels(C.FEATURES, fontsize=7)
fig.colorbar(im, fraction=0.046, pad=0.04); ax.set_title("Feature correlation"); ax.grid(False)
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- split + features
md(r"""
## 3. Chronological split, scaling & feature representations

No shuffling. Two feature representations on ONE common index (so both pipelines
are comparable and share the same test set): meaning-preserving features for the
tabular ML models (StandardScaler is applied inside the grid pipeline) and
rolling-scaled features for the Keras MLP. Rolling scaling drops the first 90
warm-up rows, which defines the common index.
""")
code(r"""
prep = F.prepare(ds)
split_ml = chronological_split(prep.X_ml.assign(**{C.TARGET: prep.y, C.COL_CLOSE: prep.close}))
split_nn = chronological_split(prep.X_nn.assign(**{C.TARGET: prep.y, C.COL_CLOSE: prep.close}))
y = {p: split_ml.parts[p][C.TARGET] for p in ("train","val","test")}
for p in ("train","val","test"):
    a,b = split_ml.parts[p].index[0].date(), split_ml.parts[p].index[-1].date()
    print(f"{p:5s} n={len(split_ml.parts[p]):4d}  {a} -> {b}")
""")

md(r"""
### Feature selection ranking (train)

`SelectKBest` scores (f_regression + mutual information) rank the features; the two
weakest are the first candidates to discard. Inside the grid search, `k` is tuned
so each model can drop weak features on its own.
""")
code(r"""
rank = F.feature_ranking(split_ml.train[C.FEATURES], y["train"])
display(rank.round(4))
print("weakest two:", rank.tail(2)["feature"].tolist())
""")

# ---------------------------------------------------------------- models
md(r"""
## 4. Models — manual grid-search CV (TimeSeriesSplit)

Four tabular models (StandardScaler + SelectKBest + estimator) plus a Keras MLP.
Selection is by validation Spearman RHO (rank correlation — trading-relevant and
robust to the trivial near-zero predictor that MAE would reward on noisy returns).
*This cell runs the full grid search and takes a few minutes.*
""")
code(r"""
results, val_pred, cv = [], {}, []
Xtr, Xval = split_ml.train[C.FEATURES], split_ml.val[C.FEATURES]
for name in ["linear","svr","random_forest","lightgbm"]:
    gs = M.manual_grid_search(Xtr, y["train"], name, n_splits=5)
    pipe = M.fit_best(Xtr, y["train"], name, gs["best_params"])
    pred = pipe.predict(Xval); val_pred[name] = pred
    results.append(E.regression_metrics(y["val"], pred, model=name, split="val"))
    cv.append({"model": name, "cv_spearman": gs["best_cv_spearman"], "best_params": gs["best_params"]})

Xtr_nn, Xval_nn = split_nn.train[C.FEATURES], split_nn.val[C.FEATURES]
gs_nn = manual_grid_search_nn(Xtr_nn, y["train"], n_splits=3)
mlp = KerasMLPRegressor(**gs_nn["best_params"]).fit(Xtr_nn, y["train"])
pred_nn = mlp.predict(Xval_nn); val_pred["keras_mlp"] = pred_nn
results.append(E.regression_metrics(y["val"], pred_nn, model="keras_mlp", split="val"))
cv.append({"model":"keras_mlp","cv_spearman":gs_nn["best_cv_spearman"],"best_params":gs_nn["best_params"]})

val_metrics = pd.DataFrame(results).merge(pd.DataFrame(cv), on="model").sort_values("spearman_rho", ascending=False)
selected = val_metrics.iloc[0]["model"]; selected_params = val_metrics.iloc[0]["best_params"]
display(val_metrics[["model","cv_spearman","spearman_rho","mae","dir_acc","top_q_dir_acc","top_q_edge"]].round(4))
print("SELECTED:", selected)
""")

# ---------------------------------------------------------------- test + shap
md(r"""
## 5. Final out-of-sample test + SHAP feature importance

The selected model is retrained on train and evaluated once on the sealed test set.
SHAP quantifies which features drive the test-set predictions.
""")
code(r"""
fp = fit_predict(selected, selected_params, ds, parts=("train","val","test"))
test_rows = [E.regression_metrics(fp["y"][p], fp["pred"][p].to_numpy(), model=selected, split=p) for p in ("val","test")]
test_metrics = pd.DataFrame(test_rows)
display(test_metrics[["split","mae","spearman_rho","dir_acc","top_q_dir_acc","bot_q_dir_acc"]].round(4))
display(E.directional_accuracy_by_quantile(fp["y"]["test"], fp["pred"]["test"].to_numpy()).round(3))
""")
code(r"""
import shap
if selected in ("linear","svr","random_forest","lightgbm"):
    pipe = fp["model"]; sc, se, est = pipe.named_steps["scaler"], pipe.named_steps["select"], pipe.named_steps["model"]
    feat = np.array(C.FEATURES)[se.get_support()]
    Xtr_t = se.transform(sc.transform(fp["split_ml"].train[C.FEATURES]))
    Xte_t = se.transform(sc.transform(fp["split_ml"].test[C.FEATURES]))
    expl = shap.LinearExplainer(est, Xtr_t) if selected=="linear" else shap.TreeExplainer(est)
    sv = expl.shap_values(Xte_t)
else:
    feat = np.array(C.FEATURES)
    expl = shap.KernelExplainer(fp["model"].predict, shap.sample(fp["split_nn"].train[C.FEATURES].to_numpy(),50,random_state=C.SEED))
    sv = np.array(expl.shap_values(fp["split_nn"].test[C.FEATURES].to_numpy()[:100], nsamples=100))
imp = pd.DataFrame({"feature":feat,"mean_abs_shap":np.abs(sv).mean(0)}).sort_values("mean_abs_shap")
fig, ax = plt.subplots(figsize=(8,5)); ax.barh(imp["feature"], imp["mean_abs_shap"], color="#8e44ad")
ax.set_xlabel("mean |SHAP|"); ax.set_title(f"SHAP feature importance on test ({selected})"); plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- trading
md(r"""
## 6. Quantile trading, test equity curve & diagnostics

**Strategy (English):** each day, rank the model's predicted next-day return
against quantile edges learned on the training predictions. Go **long** the top
quantile (largest predicted up-move), **short** the bottom quantile, otherwise
flat. Hold one day; costs are charged on turnover. This focuses trading on the
largest predicted moves.
""")
code(r"""
actual = fp["y"]["test"]
edges = BT.quantile_thresholds(fp["pred"]["train"])
pos = BT.quantile_signal(fp["pred"]["test"], edges)
rows, equities = [], {}
for scen, cost in C.COST_SCENARIOS.items():
    bt = BT.run_backtest(pos, actual.to_numpy(), cost, index=actual.index)
    m = BT.equity_metrics(bt); m.update({"scenario":scen}); rows.append(m); equities[scen]=bt["equity_net"]
bh = BT.buy_and_hold(actual.to_numpy(), index=actual.index)
trade_metrics = pd.DataFrame(rows)
display(trade_metrics[["scenario","n_trades","total_return","cagr","sharpe","profit_factor","max_drawdown","win_rate"]].round(4))
""")
code(r"""
fig, ax = plt.subplots(figsize=(11,5))
for scen in ["zero","base","high"]:
    ax.plot(equities[scen].index, equities[scen].values, lw=1.3, label=f"strategy ({scen})")
ax.plot(bh.index, bh.values, "k:", lw=1.3, label="buy & hold"); ax.axhline(1, color="gray", lw=0.6)
ax.set_ylabel("equity (start=1)"); ax.set_title(f"Test equity — {selected} quantile strategy"); ax.legend(frameon=False); plt.show()
""")
md(r"""
### Equity diagnostics

White's Reality Check over the 5-model candidate family, the Monte-Carlo
permutation p-value using the **Profit Factor** as the statistic, and bootstrap
confidence intervals.
""")
code(r"""
# candidate family: trade every model on test (base cost)
cand = {}
for name in ["linear","svr","random_forest","lightgbm","keras_mlp"]:
    p = val_metrics.set_index("model").loc[name, "best_params"]
    f2 = fit_predict(name, p, ds, parts=("train","test"))
    e2 = BT.quantile_thresholds(f2["pred"]["train"]); pos2 = BT.quantile_signal(f2["pred"]["test"], e2)
    cand[name] = BT.run_backtest(pos2, f2["y"]["test"].to_numpy(), C.BASE_COST)["net_ret"].to_numpy()
net_base = cand[selected]
boot = ST.bootstrap_ci(net_base, B_iter=5000, seed=C.SEED)
perm = ST.mc_permutation_pf(pos, actual.to_numpy(), C.BASE_COST, B_iter=2000, seed=C.SEED)
rc = ST.whites_reality_check(np.column_stack([cand[m] for m in ["linear","svr","random_forest","lightgbm","keras_mlp"]]), B_iter=2000, seed=C.SEED)
print(f"Sharpe {trade_metrics.set_index('scenario').loc['base','sharpe']:.2f} | "
      f"Profit Factor {trade_metrics.set_index('scenario').loc['base','profit_factor']:.2f} | "
      f"CAGR {trade_metrics.set_index('scenario').loc['base','cagr']:.2%}")
print(f"Bootstrap 95% CI total return [{boot['total_return']['ci_lo']:.2%}, {boot['total_return']['ci_hi']:.2%}]")
print(f"MC permutation p (Profit Factor) = {perm['p_value']:.3f}")
print(f"White's Reality Check p = {rc['reality_check_p']:.3f} ({rc['n_candidates']} candidates)")
""")

# ---------------------------------------------------------------- conclusion
md(r"""
## 7. Conclusion

- The **linear** model gives the best validation rank correlation and generalizes
  out of sample (positive test Spearman RHO and >50% directional accuracy),
  consistent with complex models overfitting noisy daily returns.
- **SHAP** shows the most influential features are **non-OHLCV** (on-chain NUPL and
  COT positioning lead), supporting the value of information beyond the price bar.
- The **quantile strategy is profitable out of sample** net of costs (positive
  CAGR, Sharpe ≈ 1, Profit Factor > 1), and beats buy-and-hold over a falling test
  period — but with only ~150 test days the permutation and reality-check p-values
  do **not** reach significance, so the edge is promising yet not conclusive.
- The program logic (leak-free features, chronological CV, cost-aware backtest,
  data-snooping-corrected diagnostics) is the deliverable; all results are reported
  regardless of significance.
""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python",
                                 "name": "python3"},
                  "language_info": {"name": "python", "version": "3.9.10"}}
import pathlib
out = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "APS1052_BTC_regression.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, str(out))
print("wrote", out)
