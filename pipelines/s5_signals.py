"""Stage 5 pipeline — trading-signal construction + confidence-threshold (tau)
calibration on validation.

Run:  python pipelines/s5_signals.py

Uses the Stage-4 selected model's stored probabilities. Sweeps tau over the grid,
scoring each by validation trading performance (base cost) and signal quality,
then FREEZES the tau that maximizes validation net return subject to a minimum
trade count. Writes the sweep table, a figure, the frozen config, and a report.
Test is never touched.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from aps import config as C
from aps import signals as S
from aps import backtest as B
from aps.pathdata import get_path_table
from aps.plotting import apply_style, save_fig

MIN_TRADES = 20  # floor to avoid picking a degenerate high-tau operating point


def _signal_precision_recall(direction: pd.Series, y: pd.Series) -> dict:
    """Directional precision/recall of extreme signals vs the true label."""
    d = direction.to_numpy()
    yt = y.to_numpy()
    sig = d != 0
    correct = (d == yt) & sig                 # signalled the right extreme side
    n_sig = int(sig.sum())
    n_extreme = int((yt != 0).sum())
    return {
        "precision": correct.sum() / n_sig if n_sig else 0.0,
        "recall": correct.sum() / n_extreme if n_extreme else 0.0,
    }


def main() -> None:
    C.ensure_dirs()
    apply_style()

    model = C.SELECTED_MODEL
    preds = pd.read_parquet(C.OUT_MODELS / f"predictions_{C.MAIN_THRESHOLD:.2f}.parquet")
    pv = S.model_proba(preds, model, "val")
    path_val = get_path_table(C.MAIN_THRESHOLD, pv.index)

    rows = []
    # reference: argmax signal (no confidence filter)
    for label, direction in [("argmax", S.argmax_signals(pv))] + \
            [(f"tau={t:.2f}", S.confidence_signals(pv, t)) for t in C.TAU_GRID]:
        cov = S.signal_coverage(direction)
        pr = _signal_precision_recall(direction, pv["y"])
        led = B.simulate(direction, path_val, C.MAIN_THRESHOLD, C.BASE_COST)
        met = B.trade_metrics(led, pv.index)
        rows.append({
            "signal": label,
            "n_signals": cov["n_signals"],
            "coverage": cov["coverage"],
            "n_long": cov["n_long"], "n_short": cov["n_short"],
            "sig_precision": pr["precision"], "sig_recall": pr["recall"],
            "net_return": met["total_net_return"],
            "gross_return": met["total_gross_return"],
            "sharpe": met["sharpe"],
            "win_rate": met.get("win_rate", np.nan),
            "profit_factor": met.get("profit_factor", np.nan),
            "n_trades": met["n_trades"],
        })
    sweep = pd.DataFrame(rows)
    sweep.to_csv(C.OUT_BACKTEST / "s5_tau_sweep_val.csv", index=False)

    # -- select tau: max validation net return with enough trades -------------
    cand = sweep[sweep["signal"].str.startswith("tau")].copy()
    eligible = cand[cand["n_trades"] >= MIN_TRADES]
    pick_from = eligible if len(eligible) else cand
    best = pick_from.loc[pick_from["net_return"].idxmax()]
    selected_tau = float(best["signal"].split("=")[1])

    config_out = {
        "selected_model": model,
        "threshold": C.MAIN_THRESHOLD,
        "selected_tau": selected_tau,
        "base_cost_per_side": C.BASE_COST,
        "min_trades_floor": MIN_TRADES,
        "entry": "next_minute_open",
        "tp_sl": C.MAIN_THRESHOLD,
        "max_hold_hours": 1,
    }
    (C.OUT_BACKTEST / "selected_config.json").write_text(
        json.dumps(config_out, indent=2), encoding="utf-8")

    # -- report --------------------------------------------------------------
    md = ["# Stage 5 — Trading Signals & Confidence-Threshold Calibration\n"]
    md.append(f"Selected model: **{model}** · base cost {C.BASE_COST:.2%}/side · "
              "entry = next-minute open · TP=SL=barrier · calibrated on "
              "validation.\n")
    md.append("## Validation tau sweep\n")
    md.append(sweep.round(4).to_markdown(index=False) + "\n")
    md.append(f"\n**Frozen operating point: tau = {selected_tau:.2f}** "
              f"(max validation net return with >= {MIN_TRADES} trades).\n")
    md.append("\nThe probabilities are over-confident (Stage 4), so tau is an "
              "empirical operating point, not a literal probability. It is frozen "
              "here and reused unchanged on the test set in Stage 8.\n")
    (C.OUT_BACKTEST / "STAGE5_SIGNALS.md").write_text("\n".join(md), encoding="utf-8")

    # -- figure: coverage vs net return / precision across tau ----------------
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    taus = [float(s.split("=")[1]) for s in cand["signal"]]
    ax1.plot(taus, cand["net_return"], "o-", color="#2980b9", label="net return")
    ax1.axvline(selected_tau, color="green", ls="--", lw=1,
                label=f"selected tau={selected_tau:.2f}")
    ax1.set_xlabel("confidence threshold tau")
    ax1.set_ylabel("validation net return", color="#2980b9")
    ax1.axhline(0, color="gray", lw=0.6)
    ax2 = ax1.twinx()
    ax2.plot(taus, cand["sig_precision"], "s--", color="#c0392b",
             label="signal precision")
    ax2.set_ylabel("signal precision", color="#c0392b")
    ax1.set_title(f"Validation tau calibration ({model}, ±{C.MAIN_THRESHOLD:.0%})")
    ax1.legend(loc="upper left", frameon=False, fontsize=8)
    ax1.grid(alpha=0.3)
    save_fig(fig, "s5_tau_calibration")

    # -- console -------------------------------------------------------------
    print("=" * 70)
    print("STAGE 5 SIGNALS — VALIDATION TAU SWEEP")
    print("=" * 70)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(sweep.round(4).to_string(index=False))
    print(f"\nFROZEN tau = {selected_tau:.2f}  (model={model}, base cost {C.BASE_COST:.2%}/side)")
    print("Config: outputs/backtest/selected_config.json")
    print("Report: outputs/backtest/STAGE5_SIGNALS.md")


if __name__ == "__main__":
    main()
