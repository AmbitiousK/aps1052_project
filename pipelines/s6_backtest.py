"""Stage 6 pipeline — barrier-aligned backtest of the frozen strategy.

Run:  python pipelines/s6_backtest.py   (requires Stage 5 selected_config.json)

Applies the frozen (model, tau, TP=SL=threshold, next-minute-open) strategy on
train + validation across the zero/base/high cost scenarios. Produces financial
metrics, equity curves vs benchmarks, and drawdown. TEST IS FROZEN until Stage 8.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from aps import config as C
from aps import signals as S
from aps import backtest as B
from aps.data import load_split
from aps.pathdata import get_path_table
from aps.plotting import apply_style, save_fig

REPORT_SPLITS = ["train", "val"]  # test frozen until Stage 8


def main() -> None:
    C.ensure_dirs()
    apply_style()

    cfg = json.loads((C.OUT_BACKTEST / "selected_config.json").read_text())
    model, tau, thr = cfg["selected_model"], cfg["selected_tau"], cfg["threshold"]

    split = load_split(thr)
    preds = pd.read_parquet(C.OUT_MODELS / f"predictions_{thr:.2f}.parquet")

    metrics_rows = []
    ledgers = {}
    equities = {}   # (split, scenario) -> equity series (net)
    for part in REPORT_SPLITS:
        pv = S.model_proba(preds, model, part)
        direction = S.confidence_signals(pv, tau)
        path_tbl = get_path_table(thr, pv.index)
        close = split.parts[part].loc[pv.index, C.COL_CLOSE]
        for scen, cost in C.COST_SCENARIOS.items():
            led = B.simulate(direction, path_tbl, thr, cost)
            met = B.trade_metrics(led, pv.index)
            met.update({"split": part, "scenario": scen, "cost_per_side": cost})
            metrics_rows.append(met)
            equities[(part, scen)] = B.equity_curve(led, pv.index, "net_ret")
            if scen == "base":
                ledgers[part] = led
        # benchmark buy&hold on this split
        equities[(part, "buyhold")] = B.buy_and_hold(close)

    metrics = pd.DataFrame(metrics_rows)
    lead = ["split", "scenario", "n_trades", "n_long", "n_short",
            "total_net_return", "total_gross_return", "sharpe", "sortino",
            "max_drawdown", "win_rate", "profit_factor", "avg_hold_min"]
    metrics = metrics[[c for c in lead if c in metrics.columns]]
    metrics.to_csv(C.OUT_BACKTEST / "s6_backtest_metrics.csv", index=False)

    # persist the base-cost trade ledgers
    for part, led in ledgers.items():
        led.to_csv(C.OUT_BACKTEST / f"s6_ledger_{part}_base.csv")

    # -- report --------------------------------------------------------------
    md = ["# Stage 6 — Barrier-Aligned Backtest (frozen strategy)\n"]
    md.append(f"Model **{model}** · tau **{tau:.2f}** · TP=SL=±{thr:.0%} · "
              "entry next-minute open · max hold 1h · train + validation "
              "(test frozen).\n")
    md.append("## Financial metrics by split and cost scenario\n")
    md.append(metrics.round(4).to_markdown(index=False) + "\n")
    base_val = metrics[(metrics.split == "val") & (metrics.scenario == "base")].iloc[0]
    md.append(f"\n**Validation (base cost): net return {base_val['total_net_return']:.2%}, "
              f"gross {base_val['total_gross_return']:.2%}, "
              f"Sharpe {base_val['sharpe']:.2f}, {int(base_val['n_trades'])} trades.**\n")
    md.append("\nAt ±3% the strategy is not profitable net of costs: extreme events "
              "are extremely rare in the 2024-25 validation regime and signal "
              "precision is low, so trading costs dominate. Note that gross returns "
              "are less negative (and positive at the highest tau), i.e. the edge is "
              "real but too small to survive costs at this barrier. Barrier "
              "sensitivity (±1%/±2%) is examined in Stage 7+.\n")
    (C.OUT_BACKTEST / "STAGE6_BACKTEST.md").write_text("\n".join(md), encoding="utf-8")

    # -- figures: equity curves (net base vs buy&hold) + drawdown ------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 8),
                             gridspec_kw={"height_ratios": [2, 1]})
    for j, part in enumerate(REPORT_SPLITS):
        ax, axd = axes[0, j], axes[1, j]
        for scen in ["zero", "base", "high"]:
            eq = equities[(part, scen)]
            ax.plot(eq.index, eq.values, lw=1.2, label=f"strategy ({scen})")
        bh = equities[(part, "buyhold")]
        ax.plot(bh.index, bh.values, lw=1.2, color="black", ls=":",
                label="buy & hold")
        ax.axhline(1.0, color="gray", lw=0.6)
        ax.set_title(f"{part} equity (±{thr:.0%}, tau={tau:.2f})")
        ax.set_ylabel("equity (start=1)")
        ax.legend(frameon=False, fontsize=8)
        # drawdown of base
        eqb = equities[(part, "base")]
        dd = (eqb.cummax() - eqb) / eqb.cummax()
        axd.fill_between(dd.index, -dd.values, 0, color="#c0392b", alpha=0.6)
        axd.set_ylabel("drawdown")
        axd.set_title(f"{part} drawdown (base cost)")
    save_fig(fig, "s6_equity_curves")

    # -- figure: cost scenario net returns bar -------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(REPORT_SPLITS)); width = 0.25
    for i, scen in enumerate(["zero", "base", "high"]):
        vals = [metrics[(metrics.split == p) & (metrics.scenario == scen)]
                ["total_net_return"].values[0] for p in REPORT_SPLITS]
        ax.bar(x + (i - 1) * width, vals, width=width, label=scen)
    ax.set_xticks(x); ax.set_xticklabels(REPORT_SPLITS)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_ylabel("net return")
    ax.set_title(f"Net return by cost scenario (±{thr:.0%}, tau={tau:.2f})")
    ax.legend(frameon=False)
    save_fig(fig, "s6_cost_scenarios")

    # -- console -------------------------------------------------------------
    print("=" * 70)
    print("STAGE 6 BACKTEST — TRAIN + VALIDATION")
    print("=" * 70)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(metrics.round(4).to_string(index=False))
    print("\nReport: outputs/backtest/STAGE6_BACKTEST.md")
    print("Figures: s6_equity_curves, s6_cost_scenarios")


if __name__ == "__main__":
    main()
