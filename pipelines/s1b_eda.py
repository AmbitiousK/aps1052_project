"""Stage 1b pipeline — exploratory data analysis (main ±3% dataset).

Run:  python pipelines/s1b_eda.py

Decision-relevant statistics (summary, correlation, VIF, MI, feature-by-label)
are computed on the TRAIN split only, to avoid peeking at validation/test. The
extreme-event timeline uses the full sample purely for description.

Writes tables to outputs/eda, figures to outputs/figures, and a consolidated
outputs/eda/STAGE1B_EDA.md report.
"""
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from aps import config as C
from aps import eda
from aps.data import load_split
from aps.plotting import apply_style, save_fig, CLASS_COLORS


def main() -> None:
    C.ensure_dirs()
    apply_style()

    split = load_split(C.MAIN_THRESHOLD)
    train = split.train           # decisions on train only
    full = pd.concat([split.train, split.val, split.test])  # timeline only

    md: list[str] = ["# Stage 1b — Exploratory Data Analysis\n"]
    md.append(f"Dataset: main ±{C.MAIN_THRESHOLD:.0%} · statistics on **train** "
              f"({len(train)} rows) unless noted.\n")

    # -- 1. Summary statistics (train) ---------------------------------------
    summ = eda.summary_stats(train)
    summ.to_csv(C.OUT_EDA / "summary_stats_train.csv", index=False)
    md.append("## 1. Summary statistics (train)\n")
    md.append(summ.round(4).to_markdown(index=False) + "\n")

    # -- 2. Correlation (Pearson + Spearman) ---------------------------------
    pear = eda.correlation(train, "pearson")
    spear = eda.correlation(train, "spearman")
    pear.to_csv(C.OUT_EDA / "corr_pearson_train.csv")
    spear.to_csv(C.OUT_EDA / "corr_spearman_train.csv")
    pairs = eda.top_correlated_pairs(pear, k=10)
    pairs.to_csv(C.OUT_EDA / "top_correlated_pairs.csv", index=False)
    md.append("## 2. Feature correlation (train)\n")
    md.append("Top-10 most correlated feature pairs (Pearson):\n")
    md.append(pairs.round(3).to_markdown(index=False) + "\n")
    max_abs = pairs["abs_corr"].max()
    strong = pairs[pairs["abs_corr"] >= 0.8]
    md.append(
        f"\nHighest absolute pairwise correlation: **{max_abs:.3f}**. "
        f"{len(strong)} pair(s) exceed 0.8, all within a single economic dimension "
        "— a *volatility* cluster (`range_24h`↔`rvol_24h`) and a "
        "*trend/stretch* cluster (`close_vs_sma24`↔`rsi_14`↔`ret_24h`). These are "
        "the same information seen through different lenses, not meaningless "
        "duplication; each feature still has a distinct one-sentence meaning, so "
        "no feature is dropped. Interpretation caveat noted for the linear model.\n")

    # -- 3. VIF + Mutual information -----------------------------------------
    vif = eda.variance_inflation(train)
    mi = eda.mutual_information(train)
    vif.to_csv(C.OUT_EDA / "vif_train.csv", index=False)
    mi.to_csv(C.OUT_EDA / "mutual_info_train.csv", index=False)
    md.append("## 3. Redundancy & informativeness (train)\n")
    md.append("VIF (variance inflation factor):\n")
    md.append(vif.round(3).to_markdown(index=False) + "\n")
    md.append("\nMutual information with 3-class label:\n")
    md.append(mi.round(5).to_markdown(index=False) + "\n")
    md.append(f"\nMax VIF: **{vif['VIF'].max():.2f}** "
              f"({'all < 5, low multicollinearity' if vif['VIF'].max() < 5 else 'inspect'}).\n")

    # -- 4. Feature separation by label --------------------------------------
    fbl = eda.feature_by_label(train)
    fbl.to_csv(C.OUT_EDA / "feature_by_label_train.csv", index=False)
    md.append("## 4. Feature means by label class (train)\n")
    md.append(fbl.round(5).to_markdown(index=False) + "\n")

    # -- 5. Distribution shift across splits ---------------------------------
    shift = eda.distribution_shift(split)
    shift.to_csv(C.OUT_EDA / "distribution_shift.csv", index=False)
    md.append("## 5. Distribution shift (KS statistic vs train)\n")
    md.append(shift.round(4).to_markdown(index=False) + "\n")
    worst = shift.iloc[0]
    md.append(f"\nLargest train→test shift: **{worst['feature']}** "
              f"(KS={worst['ks_train_test']:.3f}).\n")

    # -- 6. Extreme events over time -----------------------------------------
    ext = eda.extreme_events_by_period(full, freq="MS")
    ext.to_csv(C.OUT_EDA / "extreme_events_monthly.csv")
    md.append("## 6. Extreme-event time clustering\n")
    md.append(f"Monthly extreme-event ratio: min={ext['extreme_ratio'].min():.4f}, "
              f"max={ext['extreme_ratio'].max():.4f}, "
              f"mean={ext['extreme_ratio'].mean():.4f}. "
              f"Non-stationarity is visible in `s1b_extreme_timeline.png`.\n")

    # ========================= FIGURES ======================================

    # Fig A: feature histograms grid
    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    for ax, feat in zip(axes.ravel(), C.FEATURES):
        ax.hist(train[feat], bins=60, color="#34495e", alpha=0.85)
        ax.set_title(feat, fontsize=9)
        ax.tick_params(labelsize=7)
    fig.suptitle("Feature distributions (train)", y=1.02)
    save_fig(fig, "s1b_feature_hist")

    # Fig B: Pearson correlation heatmap
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(pear.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(C.FEATURES)))
    ax.set_yticks(range(len(C.FEATURES)))
    ax.set_xticklabels(C.FEATURES, rotation=90, fontsize=8)
    ax.set_yticklabels(C.FEATURES, fontsize=8)
    for i in range(len(C.FEATURES)):
        for j in range(len(C.FEATURES)):
            ax.text(j, i, f"{pear.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Pearson correlation (train)")
    ax.grid(False)
    save_fig(fig, "s1b_corr_heatmap")

    # Fig C: mutual information bar
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(mi["feature"][::-1], mi["mutual_info"][::-1], color="#2980b9")
    ax.set_xlabel("mutual information with label")
    ax.set_title("Feature informativeness (train)")
    save_fig(fig, "s1b_mutual_info")

    # Fig D: feature-by-label boxplots (standardized for comparability)
    z = (train[C.FEATURES] - train[C.FEATURES].mean()) / train[C.FEATURES].std()
    z[C.COL_LABEL] = train[C.COL_LABEL].values
    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    for ax, feat in zip(axes.ravel(), C.FEATURES):
        data = [z.loc[z[C.COL_LABEL] == c, feat].values for c in C.CLASSES]
        bp = ax.boxplot(data, labels=[C.CLASS_NAMES[c] for c in C.CLASSES],
                        showfliers=False, patch_artist=True)
        for patch, c in zip(bp["boxes"], C.CLASSES):
            patch.set_facecolor(CLASS_COLORS[c])
            patch.set_alpha(0.6)
        ax.set_title(feat, fontsize=9)
        ax.tick_params(labelsize=7)
    fig.suptitle("Standardized feature distribution by label (train)", y=1.02)
    save_fig(fig, "s1b_feature_by_label")

    # Fig E: extreme-event timeline (monthly)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax1.bar(ext.index, ext["up"], width=25, color=CLASS_COLORS[1], label="up (+1)")
    ax1.bar(ext.index, -ext["down"], width=25, color=CLASS_COLORS[-1], label="down (-1)")
    ax1.axhline(0, color="black", lw=0.6)
    ax1.set_ylabel("events / month")
    ax1.legend(frameon=False, ncol=2)
    ax1.set_title(f"Extreme events over time (±{C.MAIN_THRESHOLD:.0%})")
    # split boundary markers
    for b in [split.train.index[-1], split.val.index[-1]]:
        for ax in (ax1, ax2):
            ax.axvline(b, color="gray", ls="--", lw=0.8)
    ax2.plot(ext.index, ext["extreme_ratio"], color="#8e44ad")
    ax2.set_ylabel("extreme ratio")
    ax2.set_xlabel("month")
    save_fig(fig, "s1b_extreme_timeline")

    # -- Write report --------------------------------------------------------
    report = C.OUT_EDA / "STAGE1B_EDA.md"
    report.write_text("\n".join(md), encoding="utf-8")

    # -- Console summary -----------------------------------------------------
    print("=" * 70)
    print("STAGE 1b EDA — SUMMARY")
    print("=" * 70)
    print(f"Max |pairwise corr|: {max_abs:.3f}  |  Max VIF: {vif['VIF'].max():.2f}")
    print("Top-3 features by mutual information:")
    for _, r in mi.head(3).iterrows():
        print(f"  {r['feature']:20s} MI={r['mutual_info']:.5f}")
    print(f"Largest train->test KS shift: {worst['feature']} ({worst['ks_train_test']:.3f})")
    print(f"Monthly extreme ratio range: [{ext['extreme_ratio'].min():.4f}, "
          f"{ext['extreme_ratio'].max():.4f}]")
    print(f"\nReport: {report}")
    print("Figures: s1b_feature_hist, s1b_corr_heatmap, s1b_mutual_info, "
          "s1b_feature_by_label, s1b_extreme_timeline")


if __name__ == "__main__":
    main()
