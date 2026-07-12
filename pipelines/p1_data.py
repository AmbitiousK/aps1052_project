"""Pipeline 1 — dataset audit + exploratory data analysis (daily regression).

Run:  python pipelines/p1_data.py

Writes a data dictionary, a timestamp/missing audit, descriptive statistics, a
feature correlation heatmap, feature and target distributions, and a price/target
timeline. Analysis only — the dataset is fixed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from aps import config as C
from aps.data import load_dataset, chronological_split
from aps.plotting import apply_style, save_fig


def main() -> None:
    C.ensure_dirs()
    apply_style()
    ds = load_dataset()
    split = chronological_split(ds)

    # -- data dictionary -----------------------------------------------------
    rows = []
    for f in C.FEATURES:
        s = ds[f]
        rows.append({"feature": f, "from_target_ohlcv": C.FEATURE_SPEC[f]["bar"],
                     "group": C.FEATURE_SPEC[f]["group"],
                     "min": s.min(), "max": s.max(), "mean": s.mean(), "std": s.std(),
                     "description": C.FEATURE_DESCRIPTIONS.get(f, "")})
    ddict = pd.DataFrame(rows)
    ddict.to_csv(C.OUT_TABLES / "data_dictionary.csv", index=False)

    # -- timestamp / missing audit ------------------------------------------
    idx = ds.index
    full_days = (idx[-1] - idx[0]).days + 1
    audit = {
        "n_rows": len(ds), "start": idx[0], "end": idx[-1],
        "calendar_days": full_days, "coverage_pct": round(100 * len(ds) / full_days, 2),
        "monotonic": bool(idx.is_monotonic_increasing),
        "duplicates": int(idx.duplicated().sum()),
        "missing_values": int(ds.isna().sum().sum()),
        "n_features": len(C.FEATURES),
        "n_non_ohlcv": len(C.NON_OHLCV_FEATURES),
    }
    pd.DataFrame([audit]).to_csv(C.OUT_AUDIT / "timestamp_audit.csv", index=False)

    # -- descriptive stats + correlation ------------------------------------
    desc = ds[C.FEATURES + [C.TARGET]].describe().T
    desc.to_csv(C.OUT_EDA / "summary_stats.csv")
    corr = ds[C.FEATURES].corr()
    corr.to_csv(C.OUT_EDA / "corr_pearson.csv")

    # -- report --------------------------------------------------------------
    md = ["# Pipeline 1 — Data Audit & EDA (daily)\n"]
    md.append(f"**{audit['n_rows']} days**, {audit['start'].date()} → "
              f"{audit['end'].date()}, coverage {audit['coverage_pct']}%. "
              f"{audit['n_features']} features ({audit['n_non_ohlcv']} not from the "
              f"target OHLCV bar), {audit['missing_values']} missing, "
              f"{audit['duplicates']} duplicate dates.\n")
    md.append("## Data dictionary\n")
    md.append(ddict.round(4).to_markdown(index=False) + "\n")
    md.append("\n## Split boundaries\n")
    for p, (a, b) in split.boundaries().items():
        md.append(f"- {p}: {len(split.parts[p])} rows, {a.date()} → {b.date()}")
    md.append(f"\n\n## Target ({C.TARGET})\n")
    t = ds[C.TARGET]
    md.append(f"mean {t.mean():.5f}, std {t.std():.5f}, skew {t.skew():.3f}, "
              f"min {t.min():.4f}, max {t.max():.4f}.\n")
    (C.OUT_EDA / "P1_DATA.md").write_text("\n".join(md), encoding="utf-8")

    # -- figures -------------------------------------------------------------
    # correlation heatmap
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(C.FEATURES))); ax.set_yticks(range(len(C.FEATURES)))
    ax.set_xticklabels(C.FEATURES, rotation=90, fontsize=7)
    ax.set_yticklabels(C.FEATURES, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Feature correlation (Pearson)"); ax.grid(False)
    save_fig(fig, "p1_corr_heatmap")

    # feature distributions
    fig, axes = plt.subplots(4, 5, figsize=(16, 11))
    for ax, f in zip(axes.ravel(), C.FEATURES):
        ax.hist(ds[f], bins=40, color="#34495e", alpha=0.85)
        ax.set_title(f, fontsize=8); ax.tick_params(labelsize=6)
    for ax in axes.ravel()[len(C.FEATURES):]:
        ax.axis("off")
    fig.suptitle("Feature distributions", y=1.01)
    save_fig(fig, "p1_feature_hist")

    # target distribution + price/target timeline
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4))
    a1.hist(ds[C.TARGET], bins=60, color="#2980b9", alpha=0.85)
    a1.axvline(0, color="black", lw=1); a1.set_title("Next-day log return (target)")
    a2.plot(ds.index, ds["close"], color="#e67e22"); a2.set_yscale("log")
    a2.set_title("BTC close (log scale)")
    for b in (split.train.index[-1], split.val.index[-1]):
        a2.axvline(b, color="gray", ls="--", lw=0.8)
    save_fig(fig, "p1_target_price")

    print("=" * 68)
    print(f"AUDIT: {audit['n_rows']} days, coverage {audit['coverage_pct']}%, "
          f"missing {audit['missing_values']}, "
          f"{audit['n_non_ohlcv']}/{audit['n_features']} non-OHLCV features")
    print("Report: outputs/eda/P1_DATA.md")


if __name__ == "__main__":
    main()
