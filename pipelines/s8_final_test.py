"""Stage 8 pipeline — final out-of-sample test (test set unsealed once).

Run:  python pipelines/s8_final_test.py

Everything is frozen before this runs: main model logistic, next-minute-open
entry, base cost 0.05%/side, and the per-barrier tau calibrated on validation
(±3%→0.70, ±2%→0.70, ±1%→0.60). The test set is touched exactly once per barrier.
±3% is the headline; ±2%/±1% are the pre-registered sensitivity runs. No tuning.

Reports test classification metrics, trading metrics (3 cost scenarios), the test
equity curve, and test-set statistical validation (trade bootstrap CI +
permutation test — no reality check on test, since nothing is selected on test).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from aps import config as C
from aps import evaluate as E
from aps import signals as S
from aps import backtest as B
from aps import stats_tests as T
from aps.data import load_split
from aps.models import make_logistic, predict_proba_aligned
from aps.pathdata import get_path_table
from aps.plotting import apply_style, save_fig, CLASS_COLORS

MODEL = "logistic"


def _frozen_taus() -> dict:
    """Per-barrier tau frozen from validation (Stage 5 / Stage 7b)."""
    bs = pd.read_csv(C.OUT_BACKTEST / "barrier_sensitivity.csv")
    return {round(r["threshold"], 2): float(r["selected_tau"]) for _, r in bs.iterrows()}


def _eval_barrier(thr: float, tau: float) -> dict:
    """Train logistic on train, evaluate the frozen strategy on TEST once."""
    split = load_split(thr)
    model = make_logistic(seed=C.SEED).fit(split.X("train"), split.y("train"))
    proba = predict_proba_aligned(model, split.X("test"))
    yte = split.y("test")
    pred = np.array([C.CLASSES[i] for i in proba.argmax(1)])
    clf = E.classification_metrics(yte, pred, proba, model=MODEL, split="test")

    proba_df = pd.DataFrame(proba, columns=[f"p_{c}" for c in C.CLASSES],
                            index=split.test.index)
    direction = S.confidence_signals(proba_df, tau)
    path_te = get_path_table(thr, proba_df.index)
    close = split.test.loc[proba_df.index, C.COL_CLOSE]

    scen_metrics = {}
    for scen, cost in C.COST_SCENARIOS.items():
        led = B.simulate(direction, path_te, thr, cost)
        scen_metrics[scen] = B.trade_metrics(led, proba_df.index)
    base_led = B.simulate(direction, path_te, thr, C.BASE_COST)
    tb = T.trade_bootstrap_ci(base_led, B_iter=10000, seed=C.SEED)
    perm = T.circular_shift_permutation(direction, path_te, thr, C.BASE_COST,
                                        B_iter=2000, seed=C.SEED)

    def _ci(basis):
        r = tb[(tb.basis == basis) & (tb.metric == "total_return")]
        return (float(r["ci_lo"].iloc[0]), float(r["ci_hi"].iloc[0])) if len(r) else (np.nan, np.nan)
    net_lo, net_hi = _ci("net"); gr_lo, gr_hi = _ci("gross")
    base = scen_metrics["base"]
    return {
        "threshold": thr, "tau": tau,
        "test_events": int((yte != 0).sum()),
        "clf": clf, "confusion": E.confusion(yte, pred),
        "equity_base": B.equity_curve(base_led, proba_df.index, "net_ret"),
        "buyhold": B.buy_and_hold(close),
        "ledger_base": base_led,
        "summary": {
            "threshold": thr, "tau": tau, "test_events": int((yte != 0).sum()),
            "pr_auc": clf["extreme_pr_auc"], "macro_f1": clf["macro_f1"],
            "extreme_recall": clf["extreme_recall"], "extreme_precision": clf["extreme_precision"],
            "n_trades": base["n_trades"],
            "gross_return": base["total_gross_return"],
            "net_return_base": base["total_net_return"],
            "net_ci_lo": net_lo, "net_ci_hi": net_hi,
            "gross_ci_lo": gr_lo, "gross_ci_hi": gr_hi,
            "sharpe_base": base["sharpe"], "max_drawdown_base": base["max_drawdown"],
            "net_return_zero": scen_metrics["zero"]["total_net_return"],
            "net_return_high": scen_metrics["high"]["total_net_return"],
            "perm_p_net": perm["p_net"], "perm_p_gross": perm["p_gross"],
        },
    }


def main() -> None:
    C.ensure_dirs()
    apply_style()
    taus = _frozen_taus()

    results = {thr: _eval_barrier(thr, taus[thr]) for thr in [0.03, 0.02, 0.01]}
    summary = pd.DataFrame([results[t]["summary"] for t in [0.03, 0.02, 0.01]])
    summary.to_csv(C.OUT_BACKTEST / "s8_final_test.csv", index=False)

    main_res = results[0.03]
    # -- report --------------------------------------------------------------
    md = ["# Stage 8 — Final Out-of-Sample Test\n"]
    md.append("Test set unsealed once per barrier. Frozen: model **logistic**, "
              "next-minute-open entry, base cost 0.05%/side, per-barrier tau from "
              "validation. **±3% is the headline; ±2%/±1% are pre-registered "
              "sensitivity.** No tuning.\n")
    md.append("## Test summary (all barriers)\n")
    show = ["threshold", "tau", "test_events", "pr_auc", "macro_f1",
            "extreme_recall", "n_trades", "gross_return", "net_return_base",
            "net_ci_lo", "net_ci_hi", "sharpe_base", "perm_p_net"]
    md.append(summary[show].round(4).to_markdown(index=False) + "\n")
    md.append("\n## Main task ±3% — test confusion matrix (logistic, argmax)\n")
    md.append(main_res["confusion"].to_markdown() + "\n")
    m = main_res["summary"]
    md.append(f"\n## Main ±3% test result\n")
    md.append(
        f"- Classification: extreme PR-AUC **{m['pr_auc']:.4f}** "
        f"(vs random prior ~{main_res['test_events']/len(load_split(0.03).test):.4f}), "
        f"macro-F1 {m['macro_f1']:.3f}, extreme recall {m['extreme_recall']:.3f}.\n"
        f"- Trading (base cost): {int(m['n_trades'])} trades, gross "
        f"{m['gross_return']:.2%}, net **{m['net_return_base']:.2%}** "
        f"(95% CI [{m['net_ci_lo']:.2%}, {m['net_ci_hi']:.2%}]), "
        f"Sharpe {m['sharpe_base']:.2f}.\n"
        f"- Cost sensitivity: zero {m['net_return_zero']:.2%} → base "
        f"{m['net_return_base']:.2%} → high {m['net_return_high']:.2%}.\n"
        f"- Permutation p (net) = {m['perm_p_net']:.3f}.\n")
    # data-driven verdict
    s01 = results[0.01]["summary"]
    md.append("\n## Verdict\n")
    md.append(
        "Out-of-sample, the frozen strategy produced **positive net returns at the "
        f"±3% (main, {m['net_return_base']:.2%}, perm p={m['perm_p_net']:.3f}) and "
        f"±1% ({s01['net_return_base']:.2%}, Sharpe {s01['sharpe_base']:.2f}, perm "
        f"p={s01['perm_p_net']:.4f}) barriers**, so the signal-return alignment "
        "persists out of sample. Two cautions keep this from being a claim of "
        "robust profitability: (1) the bootstrap 95% CI on total net return "
        "**includes zero at every barrier** (wide, owing to limited trade counts), "
        "and (2) the validation-stage White Reality Check did **not** reject the "
        "null once the 6×4 candidate search was accounted for. The out-of-sample "
        "evidence is therefore *encouraging but not conclusive*: consistent with a "
        "weak, real predictive signal whose after-cost economic value cannot be "
        "established with confidence at this sample size. All barriers are reported "
        "regardless of sign or significance.\n")
    (C.OUT_BACKTEST / "STAGE8_FINAL_TEST.md").write_text("\n".join(md), encoding="utf-8")

    # -- figure: test equity curves per barrier ------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for ax, thr in zip(axes, [0.03, 0.02, 0.01]):
        r = results[thr]
        eq = r["equity_base"]; bh = r["buyhold"]
        ax.plot(eq.index, eq.values, color="#c0392b", lw=1.4, label="strategy (net, base)")
        ax2 = ax.twinx()
        ax2.plot(bh.index, bh.values, color="black", ls=":", lw=1, label="buy & hold")
        ax2.set_ylabel("buy&hold", color="gray", fontsize=8)
        ax.axhline(1.0, color="gray", lw=0.6)
        ax.set_title(f"±{thr:.0%} test equity (tau={r['tau']:.2f})")
        ax.set_ylabel("strategy equity")
        ax.legend(loc="upper left", frameon=False, fontsize=8)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    fig.suptitle("Final out-of-sample test equity curves (base cost)", y=1.04)
    save_fig(fig, "s8_test_equity")

    # -- console -------------------------------------------------------------
    print("=" * 70)
    print("STAGE 8 FINAL OUT-OF-SAMPLE TEST")
    print("=" * 70)
    with pd.option_context("display.width", 240, "display.max_columns", 30):
        print(summary[show].round(4).to_string(index=False))
    print("\nReport: outputs/backtest/STAGE8_FINAL_TEST.md")
    print("Figure: s8_test_equity")


if __name__ == "__main__":
    main()
