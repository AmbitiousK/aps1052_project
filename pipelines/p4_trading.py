"""Pipeline 4 — quantile trading, test equity curve, and statistical diagnostics.

Run:  python pipelines/p4_trading.py   (requires p2 outputs)

Strategy: long the top predicted-return quantile, short the bottom, flat
otherwise; quantile edges fit on TRAIN predictions. Evaluates the selected model
on the sealed test set across cost scenarios, plots equity vs buy-and-hold, and
runs the required diagnostics:
  * White's Reality Check over the 5-model candidate family (test net returns);
  * Monte-Carlo permutation p-value using the Profit Factor as the statistic;
  * bootstrap CIs; and Sharpe / Profit Factor / CAGR.
"""
from __future__ import annotations

import ast
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from aps import config as C
from aps import backtest as BT
from aps import stats_tests as ST
from aps.data import load_dataset
from aps.runner import fit_predict
from aps.plotting import apply_style, save_fig

ALL_MODELS = ["linear", "svr", "random_forest", "lightgbm", "keras_mlp"]


def main() -> None:
    C.ensure_dirs()
    apply_style()

    sel = json.loads((C.OUT_MODELS / "selected_model.json").read_text())
    selected = sel["selected_model"]
    vm = pd.read_csv(C.OUT_MODELS / "p2_val_metrics.csv")
    best_params = {r["model"]: ast.literal_eval(r["best_params"]) for _, r in vm.iterrows()}

    ds = load_dataset()

    # trade every model on test (candidate family), keep selected for the headline
    cand_net = {}      # model -> test net-return series (base cost)
    selected_fp = None
    for name in ALL_MODELS:
        fp = fit_predict(name, best_params[name], ds, parts=("train", "test"))
        edges = BT.quantile_thresholds(fp["pred"]["train"])
        pos = BT.quantile_signal(fp["pred"]["test"], edges)
        actual = fp["y"]["test"]
        bt = BT.run_backtest(pos, actual.to_numpy(), C.BASE_COST, index=actual.index)
        cand_net[name] = bt["net_ret"]
        if name == selected:
            selected_fp, selected_edges = fp, edges
        print(f"  traded {name}")

    # -- selected model: cost scenarios + equity ----------------------------
    actual = selected_fp["y"]["test"]
    close = selected_fp["split_ml"].test[C.COL_CLOSE]
    pos = BT.quantile_signal(selected_fp["pred"]["test"], selected_edges)
    scen_metrics, equities = [], {}
    for scen, cost in C.COST_SCENARIOS.items():
        bt = BT.run_backtest(pos, actual.to_numpy(), cost, index=actual.index)
        m = BT.equity_metrics(bt)
        m.update({"scenario": scen, "cost_per_side": cost})
        scen_metrics.append(m)
        equities[scen] = bt["equity_net"]
    bh = BT.buy_and_hold(actual.to_numpy(), index=actual.index)
    metrics = pd.DataFrame(scen_metrics)
    metrics.to_csv(C.OUT_BACKTEST / "p4_test_trade_metrics.csv", index=False)
    base_bt = BT.run_backtest(pos, actual.to_numpy(), C.BASE_COST, index=actual.index)
    base_bt.to_csv(C.OUT_BACKTEST / "p4_test_ledger_base.csv")

    # -- diagnostics ---------------------------------------------------------
    net_base = cand_net[selected].to_numpy()
    boot = ST.bootstrap_ci(net_base, avg_block=10, B_iter=5000, seed=C.SEED)
    perm = ST.mc_permutation_pf(pos, actual.to_numpy(), C.BASE_COST,
                                B_iter=2000, seed=C.SEED)
    rc_matrix = np.column_stack([cand_net[m].to_numpy() for m in ALL_MODELS])
    rc = ST.whites_reality_check(rc_matrix, avg_block=10, B_iter=2000, seed=C.SEED)
    diag = {"bootstrap": boot, "mc_permutation_pf": perm, "reality_check": rc}
    (C.OUT_STATS / "p4_diagnostics.json").write_text(
        json.dumps(diag, indent=2, default=float), encoding="utf-8")

    base = metrics[metrics.scenario == "base"].iloc[0]

    # -- report --------------------------------------------------------------
    md = ["# Pipeline 4 — Quantile Trading, Test Equity & Diagnostics\n"]
    md.append(f"Selected model **{selected}**. Long top quantile (Q{C.LONG_QUANTILE}), "
              f"short bottom (Q{C.SHORT_QUANTILE}); edges from train predictions. "
              "Test set, base cost 0.05%/side.\n")
    md.append("## Test trade metrics by cost scenario\n")
    show = ["scenario", "n_trades", "n_long", "n_short", "total_return", "cagr",
            "sharpe", "profit_factor", "max_drawdown", "win_rate"]
    md.append(metrics[show].round(4).to_markdown(index=False) + "\n")
    md.append("\n## Statistical diagnostics (test, base cost)\n")
    md.append(f"- Sharpe **{base['sharpe']:.2f}**, Profit Factor "
              f"**{base['profit_factor']:.2f}**, CAGR **{base['cagr']:.2%}**.\n")
    md.append(f"- Bootstrap 95% CI total return "
              f"[{boot['total_return']['ci_lo']:.2%}, {boot['total_return']['ci_hi']:.2%}] "
              f"(point {boot['total_return']['point']:.2%}); "
              f"P(total<=0)={boot['p_total_return_le_0']:.3f}.\n")
    md.append(f"- Bootstrap 95% CI profit factor "
              f"[{boot['profit_factor']['ci_lo']:.2f}, {boot['profit_factor']['ci_hi']:.2f}].\n")
    md.append(f"- **Monte-Carlo permutation p (Profit Factor) = {perm['p_value']:.3f}** "
              f"(PF_obs {perm['pf_obs']:.2f}).\n")
    md.append(f"- **White's Reality Check p = {rc['reality_check_p']:.3f}** "
              f"({rc['n_candidates']} candidate models).\n")
    (C.OUT_BACKTEST / "P4_TRADING.md").write_text("\n".join(md), encoding="utf-8")

    # -- figures -------------------------------------------------------------
    fig, (ax, axd) = plt.subplots(2, 1, figsize=(11, 7),
                                  gridspec_kw={"height_ratios": [2, 1]}, sharex=True)
    for scen in ["zero", "base", "high"]:
        ax.plot(equities[scen].index, equities[scen].values, lw=1.3,
                label=f"strategy ({scen})")
    ax.plot(bh.index, bh.values, color="black", ls=":", lw=1.3, label="buy & hold")
    ax.axhline(1.0, color="gray", lw=0.6)
    ax.set_ylabel("equity (start=1)")
    ax.set_title(f"Test equity — {selected} quantile strategy vs buy & hold")
    ax.legend(frameon=False, fontsize=8)
    eqb = equities["base"]
    dd = (eqb.cummax() - eqb) / eqb.cummax()
    axd.fill_between(dd.index, -dd.values, 0, color="#c0392b", alpha=0.6)
    axd.set_ylabel("drawdown"); axd.set_title("Strategy drawdown (base cost)")
    axd.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    axd.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(axd.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    save_fig(fig, "p4_test_equity")

    print("=" * 68)
    print(f"TEST trading ({selected}, base cost):")
    print(metrics[show].round(4).to_string(index=False))
    print(f"\nMC permutation p (PF) = {perm['p_value']:.3f}  |  "
          f"White RC p = {rc['reality_check_p']:.3f}")
    print("Report: outputs/backtest/P4_TRADING.md")


if __name__ == "__main__":
    main()
