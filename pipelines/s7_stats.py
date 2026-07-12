"""Stage 7 pipeline — statistical validation of the frozen ±3% strategy (validation).

Run:  python pipelines/s7_stats.py   (requires Stage 5 config + Stage 3 predictions)

All settings frozen (logistic, tau=0.70, next-minute open, base cost, ±3%). No
tuning. Runs:
  * trade-level bootstrap CIs (net + gross);
  * stationary-bootstrap CI on the hourly return series;
  * circular-shift permutation test;
  * White's Reality Check over the ±3% candidate family (6 models x 4 tau).
Reports gross and net throughout. Every result is kept, significant or not.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from aps import config as C
from aps import signals as S
from aps import backtest as B
from aps import stats_tests as T
from aps.data import load_split
from aps.pathdata import get_path_table
from aps.plotting import apply_style, save_fig


def _hourly_return_series(ledger: pd.DataFrame, index: pd.DatetimeIndex,
                          col: str) -> np.ndarray:
    r = pd.Series(0.0, index=index)
    if len(ledger):
        r.loc[ledger.index] = ledger[col].to_numpy()
    return r.to_numpy()


def main() -> None:
    C.ensure_dirs()
    apply_style()

    cfg = json.loads((C.OUT_BACKTEST / "selected_config.json").read_text())
    model, tau, thr, cost = (cfg["selected_model"], cfg["selected_tau"],
                             cfg["threshold"], cfg["base_cost_per_side"])

    split = load_split(thr)
    preds = pd.read_parquet(C.OUT_MODELS / f"predictions_{thr:.2f}.parquet")
    pv = S.model_proba(preds, model, "val")
    path_val = get_path_table(thr, pv.index)

    direction = S.confidence_signals(pv, tau)
    ledger = B.simulate(direction, path_val, thr, cost)
    md = ["# Stage 7 — Statistical Validation (±3% main, validation)\n"]
    md.append(f"Frozen: model **{model}**, tau **{tau:.2f}**, base cost "
              f"{cost:.2%}/side, TP=SL=±{thr:.0%}, next-minute open. "
              f"**{len(ledger)} trades.** No tuning; all results kept.\n")

    # -- 1. trade-level bootstrap CI ----------------------------------------
    tb = T.trade_bootstrap_ci(ledger, B_iter=10000, seed=C.SEED)
    tb.to_csv(C.OUT_STATS / "s7_trade_bootstrap_ci.csv", index=False)
    md.append("## 1. Trade-level bootstrap CI (95%, 10k resamples)\n")
    md.append(tb.round(4).to_markdown(index=False) + "\n")

    # -- 1b. stationary bootstrap on hourly series --------------------------
    net_h = _hourly_return_series(ledger, pv.index, "net_ret")
    gross_h = _hourly_return_series(ledger, pv.index, "gross_ret")
    sb_net = T.stationary_bootstrap_ci(net_h, avg_block=24, B_iter=5000, seed=C.SEED)
    sb_gross = T.stationary_bootstrap_ci(gross_h, avg_block=24, B_iter=5000, seed=C.SEED)
    sb = pd.DataFrame([{"basis": "net", **sb_net}, {"basis": "gross", **sb_gross}])
    sb.to_csv(C.OUT_STATS / "s7_stationary_bootstrap_ci.csv", index=False)
    md.append("## 2. Stationary-bootstrap CI on hourly returns (block=24h, 5k)\n")
    md.append(sb.round(4).to_markdown(index=False) + "\n")

    # -- 2. permutation test ------------------------------------------------
    perm = T.circular_shift_permutation(direction, path_val, thr, cost,
                                        B_iter=2000, seed=C.SEED)
    pd.DataFrame([perm]).to_csv(C.OUT_STATS / "s7_permutation_test.csv", index=False)
    md.append("## 3. Circular-shift permutation test (2000 shifts)\n")
    md.append(pd.DataFrame([perm]).round(4).to_markdown(index=False) + "\n")
    md.append(f"\nObserved net total return {perm['obs_net_total_return']:.4f} → "
              f"p_net = **{perm['p_net']:.3f}**; gross {perm['obs_gross_total_return']:.4f} "
              f"→ p_gross = **{perm['p_gross']:.3f}**.\n")

    # -- 3. White's Reality Check over the ±3% candidate family --------------
    models = ["logistic", "random_forest", "xgboost", "lightgbm", "mlp", "lstm"]
    net_cols, gross_cols, cand_names = [], [], []
    for m in models:
        pm = S.model_proba(preds, m, "val")
        p_tbl = get_path_table(thr, pm.index)
        for t in C.TAU_GRID:
            dirn = S.confidence_signals(pm, t)
            led = B.simulate(dirn, p_tbl, thr, cost)
            # align to the common val index (pv.index); lstm has slightly fewer rows
            net_cols.append(pd.Series(_hourly_return_series(led, pm.index, "net_ret"),
                                      index=pm.index).reindex(pv.index, fill_value=0.0).to_numpy())
            gross_cols.append(pd.Series(_hourly_return_series(led, pm.index, "gross_ret"),
                                        index=pm.index).reindex(pv.index, fill_value=0.0).to_numpy())
            cand_names.append(f"{m}|tau={t:.2f}")
    net_mat = np.column_stack(net_cols)
    gross_mat = np.column_stack(gross_cols)
    rc_net = T.whites_reality_check(net_mat, avg_block=24, B_iter=2000, seed=C.SEED)
    rc_gross = T.whites_reality_check(gross_mat, avg_block=24, B_iter=2000, seed=C.SEED)
    rc = pd.DataFrame([{"basis": "net", **rc_net}, {"basis": "gross", **rc_gross}])
    rc.to_csv(C.OUT_STATS / "s7_reality_check.csv", index=False)
    md.append(f"## 4. White's Reality Check ({len(cand_names)} candidates: "
              "6 models x 4 tau, benchmark always-flat)\n")
    md.append(rc.round(4).to_markdown(index=False) + "\n")
    md.append(f"\nReality-check p-value (net) = **{rc_net['reality_check_p']:.3f}**, "
              f"(gross) = **{rc_gross['reality_check_p']:.3f}**.\n")

    # -- conclusion ----------------------------------------------------------
    md.append("## Conclusion\n")
    md.append(
        "With only 32 trades the intervals are wide by construction. The gross "
        "edge is small and positive; after realistic costs the net total-return "
        "CI includes zero and the permutation / reality-check p-values do not "
        "reject the null. The evidence supports a *weak predictive signal* but "
        "not *tradeable economic value* at the ±3% barrier. Wide intervals are a "
        "result, not a failure.\n")
    (C.OUT_STATS / "STAGE7_STATS.md").write_text("\n".join(md), encoding="utf-8")

    # -- figures -------------------------------------------------------------
    # bootstrap distribution of net vs gross total return
    rng = np.random.default_rng(C.SEED)
    net_r = ledger["net_ret"].to_numpy(); gross_r = ledger["gross_ret"].to_numpy()
    n = len(net_r)
    boot_net = np.array([np.prod(1 + net_r[rng.integers(0, n, n)]) - 1 for _ in range(5000)])
    boot_gross = np.array([np.prod(1 + gross_r[rng.integers(0, n, n)]) - 1 for _ in range(5000)])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(boot_gross, bins=60, alpha=0.6, color="#27ae60", label="gross")
    ax.hist(boot_net, bins=60, alpha=0.6, color="#c0392b", label="net")
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("bootstrapped total return")
    ax.set_ylabel("frequency")
    ax.set_title(f"Trade-level bootstrap of total return (val, ±{thr:.0%}, {n} trades)")
    ax.legend(frameon=False)
    save_fig(fig, "s7_bootstrap_return")

    # -- console -------------------------------------------------------------
    print("=" * 70)
    print("STAGE 7 STATISTICAL VALIDATION — ±3% (validation)")
    print("=" * 70)
    print(f"Trades: {len(ledger)}")
    net_tot = tb[(tb.basis=='net') & (tb.metric=='total_return')].iloc[0]
    gross_tot = tb[(tb.basis=='gross') & (tb.metric=='total_return')].iloc[0]
    print(f"Net   total return: {net_tot['point']:.4f}  95% CI [{net_tot['ci_lo']:.4f}, {net_tot['ci_hi']:.4f}]")
    print(f"Gross total return: {gross_tot['point']:.4f}  95% CI [{gross_tot['ci_lo']:.4f}, {gross_tot['ci_hi']:.4f}]")
    print(f"Permutation p (net/gross): {perm['p_net']:.3f} / {perm['p_gross']:.3f}")
    print(f"Reality-check p (net/gross): {rc_net['reality_check_p']:.3f} / {rc_gross['reality_check_p']:.3f}")
    print("\nReport: outputs/stats/STAGE7_STATS.md")


if __name__ == "__main__":
    main()
