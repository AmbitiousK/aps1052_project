"""Stage 7b pipeline — pre-registered barrier sensitivity (±1% / ±2% vs ±3%).

Run:  python pipelines/s7b_barrier_sensitivity.py

±1% and ±2% are run through the SAME isomorphic chain as the ±3% main task
(aps.experiment.run_threshold): re-labelled, re-trained, tau re-calibrated on
validation, backtested, and statistically tested. They are declared sensitivity
experiments — NOT used to re-select the main model or main barrier. The ±3% row
is read from its already-frozen artifacts so the canonical outputs are untouched.
TEST STAYS SEALED for every barrier.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from aps import config as C
from aps.data import load_split
from aps.experiment import run_threshold
from aps.plotting import apply_style, save_fig

SENS_THRESHOLDS = [0.02, 0.01]   # ±3% already done in Stages 3-7


def _summary_003_from_stored() -> dict:
    """Assemble the ±3% summary from the frozen Stage 3-7 artifacts."""
    thr = 0.03
    yv = load_split(thr).y("val")
    m3 = pd.read_csv(C.OUT_MODELS / "s3_model_metrics.csv")
    lv = m3[(m3.model == "logistic") & (m3.split == "val")].iloc[0]
    cfg = json.loads((C.OUT_BACKTEST / "selected_config.json").read_text())
    m6 = pd.read_csv(C.OUT_BACKTEST / "s6_backtest_metrics.csv")
    b = m6[(m6.split == "val") & (m6.scenario == "base")].iloc[0]
    tb = pd.read_csv(C.OUT_STATS / "s7_trade_bootstrap_ci.csv")
    def _ci(basis):
        r = tb[(tb.basis == basis) & (tb.metric == "total_return")].iloc[0]
        return float(r["ci_lo"]), float(r["ci_hi"])
    net_lo, net_hi = _ci("net"); gr_lo, gr_hi = _ci("gross")
    perm = pd.read_csv(C.OUT_STATS / "s7_permutation_test.csv").iloc[0]
    rc = pd.read_csv(C.OUT_STATS / "s7_reality_check.csv")
    rc_net = rc[rc.basis == "net"]["reality_check_p"].iloc[0]
    rc_gross = rc[rc.basis == "gross"]["reality_check_p"].iloc[0]
    return {
        "threshold": thr,
        "val_extreme_events": int((yv != 0).sum()),
        "val_pr_auc_logistic": float(lv["extreme_pr_auc"]),
        "val_extreme_recall_logistic": float(lv["extreme_recall"]),
        "selected_tau": float(cfg["selected_tau"]),
        "n_trades_val": int(b["n_trades"]),
        "net_return_base": float(b["total_net_return"]),
        "gross_return_base": float(b["total_gross_return"]),
        "sharpe_base": float(b["sharpe"]),
        "max_drawdown_base": float(b["max_drawdown"]),
        "net_total_return": float(b["total_net_return"]),
        "net_ci_lo": net_lo, "net_ci_hi": net_hi,
        "gross_total_return": float(b["total_gross_return"]),
        "gross_ci_lo": gr_lo, "gross_ci_hi": gr_hi,
        "perm_p_net": float(perm["p_net"]), "perm_p_gross": float(perm["p_gross"]),
        "rc_p_net": float(rc_net), "rc_p_gross": float(rc_gross),
    }


def main() -> None:
    C.ensure_dirs()
    apply_style()

    summaries = [_summary_003_from_stored()]
    for thr in SENS_THRESHOLDS:
        print(f"running isomorphic chain for ±{thr:.0%} ...")
        summaries.append(run_threshold(thr, do_stats=True))

    df = pd.DataFrame(summaries).sort_values("threshold", ascending=False)
    df.to_csv(C.OUT_BACKTEST / "barrier_sensitivity.csv", index=False)

    show = ["threshold", "val_extreme_events", "val_pr_auc_logistic",
            "selected_tau", "n_trades_val", "gross_return_base",
            "net_return_base", "net_ci_lo", "net_ci_hi", "sharpe_base",
            "perm_p_net", "rc_p_net"]
    tbl = df[show].copy()

    # -- report --------------------------------------------------------------
    md = ["# Stage 7b — Pre-Registered Barrier Sensitivity\n"]
    md.append("Same isomorphic chain per barrier (re-label, re-train, re-calibrate "
              "tau on validation, backtest, test). Main model **logistic** frozen; "
              "sensitivity barriers NOT used to re-select main model/barrier. "
              "Validation only; test sealed.\n")
    md.append("## Barrier comparison (validation, base cost)\n")
    md.append(tbl.round(4).to_markdown(index=False) + "\n")
    md.append("\n## Full detail\n")
    md.append(df.round(4).to_markdown(index=False) + "\n")
    md.append("\n## Reading\n")
    md.append(
        "- Lower barriers have far more extreme events, so more trades and tighter "
        "intervals.\n"
        "- Gross edge is present at every barrier; whether it survives costs and, "
        "crucially, the data-snooping correction differs by barrier.\n"
        "- A single permutation test may look significant while White's Reality "
        "Check (correcting for the 6×4 candidate search) does not reject — that gap "
        "is the whole point of the reality check.\n"
        "- All results are retained regardless of significance.\n")
    (C.OUT_BACKTEST / "STAGE7B_BARRIER_SENSITIVITY.md").write_text(
        "\n".join(md), encoding="utf-8")

    # -- figure --------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    thrs = df["threshold"].tolist()
    labels = [f"±{t:.0%}" for t in thrs]
    # (a) gross vs net return
    ax = axes[0]
    x = np.arange(len(thrs)); w = 0.35
    ax.bar(x - w/2, df["gross_return_base"], w, label="gross", color="#27ae60")
    ax.bar(x + w/2, df["net_return_base"], w, label="net", color="#c0392b")
    ax.errorbar(x + w/2, df["net_return_base"],
                yerr=[df["net_return_base"] - df["net_ci_lo"],
                      df["net_ci_hi"] - df["net_return_base"]],
                fmt="none", ecolor="black", capsize=4, lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.axhline(0, color="gray", lw=0.6)
    ax.set_ylabel("total return (val)"); ax.set_title("Gross vs net (95% CI on net)")
    ax.legend(frameon=False)
    # (b) p-values
    ax = axes[1]
    ax.bar(x - w/2, df["perm_p_net"], w, label="permutation p", color="#2980b9")
    ax.bar(x + w/2, df["rc_p_net"], w, label="reality-check p", color="#8e44ad")
    ax.axhline(0.05, color="red", ls="--", lw=1, label="0.05")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("p-value (net)"); ax.set_title("Significance by barrier")
    ax.legend(frameon=False, fontsize=8)
    # (c) events & trades
    ax = axes[2]
    ax.bar(x - w/2, df["val_extreme_events"], w, label="val extreme events", color="#e67e22")
    ax.bar(x + w/2, df["n_trades_val"], w, label="val trades", color="#16a085")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("count"); ax.set_title("Sample size by barrier")
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Barrier sensitivity (validation, base cost)", y=1.03)
    save_fig(fig, "s7b_barrier_sensitivity")

    # -- console -------------------------------------------------------------
    print("=" * 70)
    print("STAGE 7b BARRIER SENSITIVITY — VALIDATION")
    print("=" * 70)
    with pd.option_context("display.width", 240, "display.max_columns", 30):
        print(tbl.round(4).to_string(index=False))
    print("\nReport: outputs/backtest/STAGE7B_BARRIER_SENSITIVITY.md")


if __name__ == "__main__":
    main()
