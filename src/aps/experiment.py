"""End-to-end single-barrier experiment chain (isomorphic across thresholds).

`run_threshold(thr)` runs the identical pipeline for any barrier: train the six
models on that barrier's labels, evaluate on validation, calibrate the confidence
threshold tau for the frozen main model (logistic) on validation trading
performance, backtest across cost scenarios, and run the statistical tests. This
is the SAME code for ±3% (main) and the pre-registered ±1% / ±2% sensitivity
runs, so no barrier gets special treatment and none is used to re-select the main
model or main barrier. TEST IS NEVER TOUCHED here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import evaluate as E
from . import signals as S
from . import backtest as B
from . import stats_tests as T
from .data import load_split
from .models import (make_logistic, make_random_forest, make_lightgbm,
                     make_xgboost, predict_proba_aligned)
from .nn_models import MLPClassifier, LSTMClassifier
from .pathdata import get_path_table

MODELS = ["logistic", "random_forest", "xgboost", "lightgbm", "mlp", "lstm"]
MAIN_MODEL = "logistic"          # frozen from the ±3% Stage-4 selection
MIN_TRADES = 20                  # same tau-selection floor as Stage 5
PROBA_COLS = [f"p_{c}" for c in C.CLASSES]

_TAB_BUILDERS = {
    "logistic": make_logistic,
    "random_forest": make_random_forest,
    "xgboost": make_xgboost,
    "lightgbm": make_lightgbm,
    "mlp": lambda seed=C.SEED: MLPClassifier(seed=seed),
}


def _pred_frame(model_name, split_name, ts, proba, y) -> pd.DataFrame:
    df = pd.DataFrame(proba, columns=PROBA_COLS, index=ts)
    df["pred"] = [C.CLASSES[i] for i in proba.argmax(1)]
    df["y"] = np.asarray(y)
    df.insert(0, "model", model_name)
    df.insert(1, "split", split_name)
    df.index.name = "ts"
    return df.reset_index()


def train_all_models(split, splits=("train", "val")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train the six models; return (predictions_long, val_metrics)."""
    X_tr, y_tr = split.X("train"), split.y("train")
    frames, metrics = [], []
    for name, builder in _TAB_BUILDERS.items():
        model = builder(seed=C.SEED)
        model.fit(X_tr, y_tr)
        for part in splits:
            X, y = split.X(part), split.y(part)
            proba = predict_proba_aligned(model, X)
            pred = np.array([C.CLASSES[i] for i in proba.argmax(1)])
            frames.append(_pred_frame(name, part, X.index, proba, y))
            metrics.append(E.classification_metrics(y, pred, proba, model=name, split=part))
    lstm = LSTMClassifier(seed=C.SEED)
    lstm.fit(split.train, y_tr)
    for part in splits:
        pdf = split.parts[part]
        proba, end_idx = lstm.predict_proba_df(pdf)
        y = pdf.loc[end_idx, C.COL_LABEL]
        pred = np.array([C.CLASSES[i] for i in proba.argmax(1)])
        frames.append(_pred_frame("lstm", part, end_idx, proba, y))
        metrics.append(E.classification_metrics(y, pred, proba, model="lstm", split=part))
    return pd.concat(frames, ignore_index=True), E.metrics_frame(metrics)


def select_tau(preds, model, thr, cost, split) -> tuple[float, pd.DataFrame]:
    """Calibrate tau on validation trading performance (max net return, >=20 trades)."""
    pv = S.model_proba(preds, model, "val")
    path_val = get_path_table(thr, pv.index)
    rows = []
    for t in C.TAU_GRID:
        direction = S.confidence_signals(pv, t)
        led = B.simulate(direction, path_val, thr, cost)
        met = B.trade_metrics(led, pv.index)
        rows.append({"tau": t, "n_trades": met["n_trades"],
                     "net_return": met["total_net_return"],
                     "gross_return": met["total_gross_return"],
                     "sharpe": met["sharpe"]})
    sweep = pd.DataFrame(rows)
    elig = sweep[sweep["n_trades"] >= MIN_TRADES]
    pick = elig if len(elig) else sweep
    tau = float(pick.loc[pick["net_return"].idxmax(), "tau"])
    return tau, sweep


def backtest_scenarios(preds, model, tau, thr, split, part="val") -> pd.DataFrame:
    pv = S.model_proba(preds, model, part)
    path_tbl = get_path_table(thr, pv.index)
    direction = S.confidence_signals(pv, tau)
    rows = []
    for scen, cost in C.COST_SCENARIOS.items():
        led = B.simulate(direction, path_tbl, thr, cost)
        met = B.trade_metrics(led, pv.index)
        met.update({"scenario": scen, "cost_per_side": cost})
        rows.append(met)
    return pd.DataFrame(rows)


def run_stats(preds, model, tau, thr, cost, B_perm=1000, B_rc=1000) -> dict:
    """Bootstrap CI + permutation + White RC on validation (as in Stage 7)."""
    pv = S.model_proba(preds, model, "val")
    path_val = get_path_table(thr, pv.index)
    direction = S.confidence_signals(pv, tau)
    ledger = B.simulate(direction, path_val, thr, cost)

    tb = T.trade_bootstrap_ci(ledger, B_iter=5000, seed=C.SEED)
    def _ci(basis, metric):
        r = tb[(tb.basis == basis) & (tb.metric == metric)]
        return (float(r["point"].iloc[0]), float(r["ci_lo"].iloc[0]),
                float(r["ci_hi"].iloc[0])) if len(r) else (np.nan, np.nan, np.nan)

    perm = T.circular_shift_permutation(direction, path_val, thr, cost,
                                        B_iter=B_perm, seed=C.SEED)

    # candidate family: 6 models x tau grid
    net_cols, gross_cols = [], []
    for m in MODELS:
        pm = S.model_proba(preds, m, "val")
        p_tbl = get_path_table(thr, pm.index)
        for t in C.TAU_GRID:
            dirn = S.confidence_signals(pm, t)
            led = B.simulate(dirn, p_tbl, thr, cost)
            for store, col in [(net_cols, "net_ret"), (gross_cols, "gross_ret")]:
                r = pd.Series(0.0, index=pm.index)
                if len(led):
                    r.loc[led.index] = led[col].to_numpy()
                store.append(r.reindex(pv.index, fill_value=0.0).to_numpy())
    rc_net = T.whites_reality_check(np.column_stack(net_cols), B_iter=B_rc, seed=C.SEED)
    rc_gross = T.whites_reality_check(np.column_stack(gross_cols), B_iter=B_rc, seed=C.SEED)

    net_pt, net_lo, net_hi = _ci("net", "total_return")
    gr_pt, gr_lo, gr_hi = _ci("gross", "total_return")
    return {
        "n_trades": len(ledger),
        "net_total_return": net_pt, "net_ci_lo": net_lo, "net_ci_hi": net_hi,
        "gross_total_return": gr_pt, "gross_ci_lo": gr_lo, "gross_ci_hi": gr_hi,
        "perm_p_net": perm["p_net"], "perm_p_gross": perm["p_gross"],
        "rc_p_net": rc_net["reality_check_p"], "rc_p_gross": rc_gross["reality_check_p"],
    }


def run_threshold(thr: float, do_stats: bool = True) -> dict:
    """Full isomorphic chain for one barrier. Returns a summary dict."""
    split = load_split(thr)
    preds, val_metrics = train_all_models(split)
    preds.to_parquet(C.OUT_MODELS / f"predictions_{thr:.2f}.parquet")

    main_val = val_metrics[(val_metrics.model == MAIN_MODEL) &
                           (val_metrics.split == "val")].iloc[0]
    tau, sweep = select_tau(preds, MAIN_MODEL, thr, C.BASE_COST, split)
    bt = backtest_scenarios(preds, MAIN_MODEL, tau, thr, split, "val")
    base = bt[bt.scenario == "base"].iloc[0]

    # class distribution (val)
    yv = split.y("val")
    summary = {
        "threshold": thr,
        "val_extreme_events": int((yv != 0).sum()),
        "val_pr_auc_logistic": float(main_val["extreme_pr_auc"]),
        "val_extreme_recall_logistic": float(main_val["extreme_recall"]),
        "selected_tau": tau,
        "n_trades_val": int(base["n_trades"]),
        "net_return_base": float(base["total_net_return"]),
        "gross_return_base": float(base["total_gross_return"]),
        "sharpe_base": float(base["sharpe"]),
        "max_drawdown_base": float(base["max_drawdown"]),
    }
    if do_stats:
        summary.update(run_stats(preds, MAIN_MODEL, tau, thr, C.BASE_COST))
    return summary
