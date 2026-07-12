"""Stage 1 pipeline — data integrity audit.

Run:  python pipelines/s1_audit.py

Writes:
  outputs/tables/data_dictionary.csv
  outputs/audit/class_distribution.csv
  outputs/audit/missing_by_split.csv
  outputs/audit/timestamp_audit.csv
  outputs/audit/label_integrity.csv
  outputs/audit/split_boundaries.csv
  outputs/figures/s1_class_distribution.png
  outputs/figures/s1_timestamp_gaps.png
  outputs/audit/STAGE1_AUDIT.md   (consolidated human-readable report)
"""
from __future__ import annotations

import pandas as pd

from aps import config as C
from aps import audit
from aps.data import load_dataset, chronological_split, load_split
from aps.plotting import apply_style, save_fig, CLASS_COLORS
import matplotlib.pyplot as plt


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def main() -> None:
    C.ensure_dirs()
    apply_style()

    md: list[str] = ["# Stage 1 — Data Integrity Audit\n"]
    md.append(f"Main threshold: **±{C.MAIN_THRESHOLD:.0%}** · "
              f"features: **{len(C.FEATURES)}** · seed: **{C.SEED}**\n")

    # -- Data dictionary (main dataset) --------------------------------------
    main_df = load_dataset(C.MAIN_THRESHOLD)
    ddict = audit.data_dictionary(main_df)
    ddict.to_csv(C.OUT_TABLES / "data_dictionary.csv", index=False)
    md.append("## 1. Data dictionary\n")
    md.append(ddict.to_markdown(index=False,
                                floatfmt=".4g") + "\n")

    # -- Timestamp audit (per threshold) -------------------------------------
    ts_rows, top_gaps_main = [], None
    for thr in C.THRESHOLDS:
        df = load_dataset(thr)
        info = audit.timestamp_audit(df)
        if thr == C.MAIN_THRESHOLD:
            top_gaps_main = info["top_gaps"]
        row = {k: v for k, v in info.items() if k != "top_gaps"}
        row["threshold"] = thr
        ts_rows.append(row)
    ts_df = pd.DataFrame(ts_rows).set_index("threshold").reset_index()
    ts_df.to_csv(C.OUT_AUDIT / "timestamp_audit.csv", index=False)
    md.append("## 2. Timestamp audit\n")
    md.append(ts_df.to_markdown(index=False, floatfmt=".4g") + "\n")
    md.append(f"\nTop gaps (main ±{C.MAIN_THRESHOLD:.0%}, hours): "
              + ", ".join(f"{g:.0f}" for g in top_gaps_main.values) + "\n")

    # -- Missing-value audit (main split) ------------------------------------
    split_main = chronological_split(main_df, C.MAIN_THRESHOLD)
    miss = audit.missing_audit(split_main)
    miss.to_csv(C.OUT_AUDIT / "missing_by_split.csv", index=False)
    md.append("## 3. Missing-value audit (main split)\n")
    md.append(miss.to_markdown(index=False) + "\n")
    md.append(f"\nTotal missing across all features/splits: "
              f"**{int(miss['total_missing'].sum())}**\n")

    # -- Class-distribution audit (all thresholds) ---------------------------
    cd_all = pd.concat(
        [audit.class_distribution(load_split(thr)) for thr in C.THRESHOLDS],
        ignore_index=True,
    )
    cd_all.to_csv(C.OUT_AUDIT / "class_distribution.csv", index=False)
    md.append("## 4. Class-distribution audit\n")
    md.append(cd_all.to_markdown(index=False, floatfmt=".4f") + "\n")

    # -- Label-integrity audit (all thresholds) ------------------------------
    li = pd.DataFrame([audit.label_integrity(load_dataset(thr), thr)
                       for thr in C.THRESHOLDS])
    li.to_csv(C.OUT_AUDIT / "label_integrity.csv", index=False)
    md.append("## 5. Label-integrity audit\n")
    md.append(li.to_markdown(index=False) + "\n")
    ok = int(li["total_violations"].sum()) == 0
    md.append(f"\nLabel integrity: **{'PASS — 0 violations' if ok else 'FAIL'}**\n")

    # -- Split boundaries ----------------------------------------------------
    sb_rows = []
    for part, (a, b) in split_main.boundaries().items():
        sb_rows.append({"split": part, "n": len(split_main.parts[part]),
                        "start": a, "end": b})
    sb = pd.DataFrame(sb_rows)
    sb.to_csv(C.OUT_AUDIT / "split_boundaries.csv", index=False)
    md.append("## 6. Chronological split boundaries (main dataset)\n")
    md.append(sb.to_markdown(index=False) + "\n")

    # -- Figure: class distribution (share) across splits, all thresholds ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=False)
    for ax, thr in zip(axes, C.THRESHOLDS):
        sub = cd_all[cd_all["threshold"] == thr]
        parts = sub["split"].tolist()
        bottoms = [0.0] * len(parts)
        for cls, col in [(-1, "share_-1"), (0, "share_0"), (1, "share_+1")]:
            vals = sub[col].values
            ax.bar(parts, vals, bottom=bottoms, color=CLASS_COLORS[cls],
                   label=C.CLASS_NAMES[cls])
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        ax.set_title(f"±{thr:.0%}")
        ax.set_ylim(0, 1)
        if thr == C.THRESHOLDS[0]:
            ax.set_ylabel("class share")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle("Class distribution by split and barrier threshold", y=1.05)
    save_fig(fig, "s1_class_distribution")

    # -- Figure: extreme-event share only (readable at 99:1 imbalance) -------
    fig, ax = plt.subplots(figsize=(7, 3.6))
    width = 0.25
    xs = range(3)
    for i, part in enumerate(["train", "val", "test"]):
        sub = cd_all[cd_all["split"] == part].sort_values("threshold")
        ax.bar([x + (i - 1) * width for x in xs], sub["extreme_ratio"].values,
               width=width, label=part)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"±{t:.0%}" for t in sorted(C.THRESHOLDS)])
    ax.set_ylabel("extreme-event ratio")
    ax.set_title("Extreme-event ratio by threshold and split")
    ax.legend(frameon=False)
    save_fig(fig, "s1_extreme_ratio")

    # -- Figure: timestamp gaps (main) ---------------------------------------
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(range(len(top_gaps_main)), top_gaps_main.values, color="#2c3e50")
    ax.set_xlabel("gap rank")
    ax.set_ylabel("gap length (hours)")
    ax.set_title(f"Top-10 timestamp gaps (main ±{C.MAIN_THRESHOLD:.0%})")
    save_fig(fig, "s1_timestamp_gaps")

    # -- Write consolidated report -------------------------------------------
    report = C.OUT_AUDIT / "STAGE1_AUDIT.md"
    report.write_text("\n".join(md), encoding="utf-8")

    # -- Console summary -----------------------------------------------------
    print("=" * 70)
    print("STAGE 1 AUDIT — SUMMARY")
    print("=" * 70)
    print(f"Rows (main ±{C.MAIN_THRESHOLD:.0%}): {len(main_df)}")
    print(f"Timestamp: monotone={ts_df.loc[0,'is_monotonic_increasing']}  "
          f"dupes={ts_df.loc[0,'n_duplicates']}  "
          f"on_the_hour={ts_df.loc[0,'all_on_the_hour']}  "
          f"coverage={ts_df.loc[0,'coverage_pct']}%")
    print(f"Missing values total: {int(miss['total_missing'].sum())}")
    print(f"Label integrity violations: {int(li['total_violations'].sum())} "
          f"({'PASS' if ok else 'FAIL'})")
    print("Split boundaries:")
    for _, r in sb.iterrows():
        print(f"  {r['split']:5s} n={r['n']:6d}  {r['start']}  ->  {r['end']}")
    print(f"\nReport: {report}")
    print("Figures: s1_class_distribution, s1_extreme_ratio, s1_timestamp_gaps")


if __name__ == "__main__":
    main()
