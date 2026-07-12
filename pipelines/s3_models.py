"""Stage 3 pipeline — primary models (RF / XGBoost / LightGBM / MLP / LSTM).

Run:  python pipelines/s3_models.py

Trains the five primary models plus the logistic baseline (carried forward so the
Stage-4 selection sees every learning model in one table). Evaluates on train +
validation; TEST STAYS FROZEN until Stage 8. Persists per-model validation/train
probabilities to outputs/models/predictions_<thr>.parquet for reuse by later
stages, and writes a comparison table, figures, and a report.

The LSTM predicts only on rows with a full contiguous 24h lookback window, so its
reported n is slightly smaller; this is stated in the table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from aps import config as C
from aps import evaluate as E
from aps.data import load_split
from aps.models import (make_logistic, make_random_forest, make_lightgbm,
                        make_xgboost, predict_proba_aligned)
from aps.nn_models import MLPClassifier, LSTMClassifier
from aps.plotting import apply_style, save_fig

EVAL_SPLITS = ["train", "val"]
PROBA_COLS = [f"p_{c}" for c in C.CLASSES]
HEADLINE = ["model", "split", "n", "balanced_accuracy", "macro_f1", "mcc",
            "extreme_precision", "extreme_recall", "extreme_f1",
            "extreme_pr_auc", "log_loss"]

# LSTM is handled specially (sequence subset); the rest share the tabular flow.
TABULAR_BUILDERS = {
    "logistic": make_logistic,
    "random_forest": make_random_forest,
    "xgboost": make_xgboost,
    "lightgbm": make_lightgbm,
    "mlp": lambda seed=C.SEED: MLPClassifier(seed=seed),
}


def _pred_frame(model_name, split_name, ts, proba, y_true) -> pd.DataFrame:
    df = pd.DataFrame(proba, columns=PROBA_COLS, index=ts)
    df["pred"] = [C.CLASSES[i] for i in proba.argmax(1)]
    df["y"] = np.asarray(y_true)
    df.insert(0, "model", model_name)
    df.insert(1, "split", split_name)
    df.index.name = "ts"
    return df.reset_index()


def main() -> None:
    C.ensure_dirs()
    apply_style()
    split = load_split(C.MAIN_THRESHOLD)
    X_tr, y_tr = split.X("train"), split.y("train")

    results: list[dict] = []
    pred_frames: list[pd.DataFrame] = []
    notes: dict[str, str] = {}

    # -- tabular models ------------------------------------------------------
    for name, builder in TABULAR_BUILDERS.items():
        model = builder(seed=C.SEED)
        model.fit(X_tr, y_tr)
        if hasattr(model, "best_val_f1_"):
            notes[name] = f"early-stop val macro-F1 {model.best_val_f1_:.3f}"
        for part in EVAL_SPLITS:
            X, y = split.X(part), split.y(part)
            proba = predict_proba_aligned(model, X)
            pred = np.array([C.CLASSES[i] for i in proba.argmax(1)])
            results.append(E.classification_metrics(y, pred, proba,
                                                    model=name, split=part))
            pred_frames.append(_pred_frame(name, part, X.index, proba, y))
        print(f"  trained {name}")

    # -- LSTM (sequence subset) ---------------------------------------------
    lstm = LSTMClassifier(seed=C.SEED)
    lstm.fit(split.train, y_tr)
    notes["lstm"] = f"early-stop val macro-F1 {lstm.best_val_f1_:.3f}"
    for part in EVAL_SPLITS:
        part_df = split.parts[part]
        proba, end_idx = lstm.predict_proba_df(part_df)
        y = part_df.loc[end_idx, C.COL_LABEL]
        pred = np.array([C.CLASSES[i] for i in proba.argmax(1)])
        results.append(E.classification_metrics(y, pred, proba,
                                                model="lstm", split=part))
        pred_frames.append(_pred_frame("lstm", part, end_idx, proba, y))
    print("  trained lstm")

    # -- persist predictions + metrics --------------------------------------
    preds = pd.concat(pred_frames, ignore_index=True)
    preds.to_parquet(C.OUT_MODELS / f"predictions_{C.MAIN_THRESHOLD:.2f}.parquet")
    metrics = E.metrics_frame(results)
    metrics.to_csv(C.OUT_MODELS / "s3_model_metrics.csv", index=False)

    val_tbl = metrics[metrics["split"] == "val"][HEADLINE].round(4)
    val_tbl = val_tbl.sort_values("extreme_pr_auc", ascending=False)

    # -- report --------------------------------------------------------------
    md = ["# Stage 3 — Primary Models\n"]
    md.append(f"Main threshold ±{C.MAIN_THRESHOLD:.0%}. Six learning models "
              "evaluated on train + validation (test frozen). Ranked by "
              "extreme-event PR-AUC (the metric that matters for this task).\n")
    md.append("## Validation comparison (ranked by extreme PR-AUC)\n")
    md.append(val_tbl.to_markdown(index=False) + "\n")
    md.append("\nEarly-stopping notes: " +
              "; ".join(f"{k}: {v}" for k, v in notes.items()) + "\n")
    md.append("\n## Full metrics (train + val)\n")
    md.append(metrics.round(4).to_markdown(index=False) + "\n")
    (C.OUT_MODELS / "STAGE3_MODELS.md").write_text("\n".join(md), encoding="utf-8")

    # -- figures -------------------------------------------------------------
    order = val_tbl["model"].tolist()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(order, val_tbl.set_index("model").loc[order, "extreme_pr_auc"],
           color="#2980b9")
    # random baseline reference (extreme prior on val)
    base = (split.val[C.COL_LABEL] != 0).mean()
    ax.axhline(base, color="red", ls="--", lw=1, label=f"random prior ({base:.4f})")
    ax.set_ylabel("extreme PR-AUC")
    ax.set_title(f"Validation extreme-event PR-AUC (±{C.MAIN_THRESHOLD:.0%})")
    ax.legend(frameon=False)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    save_fig(fig, "s3_pr_auc_val")

    fig, ax = plt.subplots(figsize=(9, 4.2))
    mplot = ["balanced_accuracy", "macro_f1", "extreme_recall", "extreme_precision"]
    x = np.arange(len(order)); width = 0.2
    for i, m in enumerate(mplot):
        vals = val_tbl.set_index("model").loc[order, m].values
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=m)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("score"); ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.set_title(f"Validation metrics by model (±{C.MAIN_THRESHOLD:.0%})")
    save_fig(fig, "s3_metrics_val")

    # -- console -------------------------------------------------------------
    print("=" * 70)
    print("STAGE 3 PRIMARY MODELS — VALIDATION (ranked by extreme PR-AUC)")
    print("=" * 70)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(val_tbl.to_string(index=False))
    print(f"\nPredictions saved: outputs/models/predictions_{C.MAIN_THRESHOLD:.2f}.parquet")
    print("Report: outputs/models/STAGE3_MODELS.md")


if __name__ == "__main__":
    main()
