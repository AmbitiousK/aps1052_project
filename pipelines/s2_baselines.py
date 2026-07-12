"""Stage 2 pipeline — classification baselines + logistic regression.

Run:  python pipelines/s2_baselines.py

Models: majority / prior-random / momentum-rule / multinomial logistic regression.
Evaluated on train (diagnostic) and validation. TEST IS NOT TOUCHED (frozen until
Stage 8). Writes a metrics comparison table, per-model confusion matrices, and a
consolidated report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from aps import config as C
from aps import evaluate as E
from aps.data import load_split
from aps.models import (
    MajorityClassifier, PriorRandomClassifier, MomentumRule,
    make_logistic, predict_proba_aligned,
)
from aps.plotting import apply_style, save_fig

# Which splits Stage 2 is allowed to look at.
EVAL_SPLITS = ["train", "val"]

# Headline columns for the console/report comparison (val).
HEADLINE = ["model", "split", "balanced_accuracy", "macro_f1", "mcc",
            "extreme_precision", "extreme_recall", "extreme_f1",
            "recall_-1", "recall_1", "extreme_pr_auc"]


def _proba_or_none(model, X):
    if hasattr(model, "predict_proba"):
        try:
            return predict_proba_aligned(model, X)
        except Exception:
            return None
    return None


def main() -> None:
    C.ensure_dirs()
    apply_style()

    split = load_split(C.MAIN_THRESHOLD)

    models = {
        "majority": MajorityClassifier(),
        "prior_random": PriorRandomClassifier(seed=C.SEED),
        "momentum": MomentumRule(),
        "logistic": make_logistic(seed=C.SEED),
    }

    X_tr, y_tr = split.X("train"), split.y("train")

    results: list[dict] = []
    confusions: dict[str, pd.DataFrame] = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        fitted[name] = model
        for part in EVAL_SPLITS:
            X, y = split.X(part), split.y(part)
            y_pred = model.predict(X)
            y_proba = _proba_or_none(model, X)
            results.append(E.classification_metrics(
                y, y_pred, y_proba, model=name, split=part))
            if part == "val":
                confusions[name] = E.confusion(y, y_pred)

    metrics = E.metrics_frame(results)
    metrics.to_csv(C.OUT_MODELS / "s2_baseline_metrics.csv", index=False)

    val_tbl = metrics[metrics["split"] == "val"][HEADLINE].round(4)

    # -- Report --------------------------------------------------------------
    md = ["# Stage 2 — Classification Baselines\n"]
    md.append(f"Main threshold ±{C.MAIN_THRESHOLD:.0%} · evaluated on train + "
              "validation (test frozen). Accuracy is omitted from the headline "
              "because predicting all-flat already scores ~99.7%.\n")
    md.append("## Validation comparison\n")
    md.append(val_tbl.to_markdown(index=False) + "\n")
    md.append(f"\nMomentum rule selected θ = {fitted['momentum'].theta_:.5f} "
              f"on ret_1h (train macro-F1 {fitted['momentum'].train_macro_f1_:.4f}).\n")
    md.append("\n## Confusion matrices (validation)\n")
    for name, cm in confusions.items():
        md.append(f"\n**{name}**\n")
        md.append(cm.to_markdown() + "\n")
    md.append("\n## Full metrics (train + val)\n")
    md.append(metrics.round(4).to_markdown(index=False) + "\n")
    (C.OUT_MODELS / "STAGE2_BASELINES.md").write_text("\n".join(md), encoding="utf-8")

    # -- Figure: confusion matrices grid (val) -------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (name, cm) in zip(axes, confusions.items()):
        im = ax.imshow(cm.values, cmap="Blues")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels([C.CLASS_NAMES[c] for c in C.CLASSES])
        ax.set_yticklabels([C.CLASS_NAMES[c] for c in C.CLASSES])
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(name)
        ax.grid(False)
        vmax = cm.values.max()
        for i in range(3):
            for j in range(3):
                v = cm.values[i, j]
                ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                        color="white" if v > vmax * 0.5 else "black")
    fig.suptitle(f"Validation confusion matrices (±{C.MAIN_THRESHOLD:.0%})", y=1.04)
    save_fig(fig, "s2_confusion_val")

    # -- Figure: headline metric bars (val) ----------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.2))
    metrics_to_plot = ["balanced_accuracy", "macro_f1", "extreme_recall",
                       "extreme_precision"]
    names = list(models.keys())
    width = 0.2
    x = np.arange(len(names))
    for i, m in enumerate(metrics_to_plot):
        vals = [val_tbl[val_tbl["model"] == n][m].values[0] for n in names]
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=m)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("score")
    ax.set_title(f"Validation metrics by model (±{C.MAIN_THRESHOLD:.0%})")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    save_fig(fig, "s2_metrics_val")

    # -- Console -------------------------------------------------------------
    print("=" * 70)
    print("STAGE 2 BASELINES — VALIDATION")
    print("=" * 70)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(val_tbl.to_string(index=False))
    print(f"\nMomentum θ={fitted['momentum'].theta_:.5f}")
    print("Report: outputs/models/STAGE2_BASELINES.md")
    print("Figures: s2_confusion_val, s2_metrics_val")


if __name__ == "__main__":
    main()
