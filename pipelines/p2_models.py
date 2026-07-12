"""Pipeline 2 — train the 5 models (grid-search CV) and select on validation.

Run:  python pipelines/p2_models.py

Models (>=5, two require scaling): LinearRegression, SVR, RandomForest, LightGBM
(tabular, StandardScaler+SelectKBest inside a manual TimeSeriesSplit grid search)
and a Keras MLP (rolling-scaled features). Both pipelines share ONE common index
and split, so validation metrics are comparable and the test set is identical.

Writes: feature ranking, per-model CV/val metrics, validation predictions, and a
report. Selection is by validation Spearman RHO. TEST IS NOT TOUCHED.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from aps import config as C
from aps import evaluate as E
from aps import features as F
from aps import models as M
from aps.nn_keras import KerasMLPRegressor, manual_grid_search_nn
from aps.data import load_dataset, chronological_split
from aps.plotting import apply_style, save_fig

TABULAR = ["linear", "svr", "random_forest", "lightgbm"]


def main() -> None:
    C.ensure_dirs()
    apply_style()

    ds = load_dataset()
    prep = F.prepare(ds)
    # common chronological split (same rows for every model)
    split_ml = chronological_split(prep.X_ml.assign(**{C.TARGET: prep.y}))
    split_nn = chronological_split(prep.X_nn.assign(**{C.TARGET: prep.y}))
    y = {p: split_ml.parts[p][C.TARGET] for p in ["train", "val", "test"]}

    print(f"common index: {len(prep.index)} rows  "
          f"train/val/test = {len(split_ml.train)}/{len(split_ml.val)}/{len(split_ml.test)}")

    # -- R3: feature ranking (train) ----------------------------------------
    rank = F.feature_ranking(split_ml.train[C.FEATURES], y["train"])
    rank.to_csv(C.OUT_TABLES / "feature_ranking.csv", index=False)
    worst = rank.tail(2)["feature"].tolist()

    # -- R4: tabular ML grid search -----------------------------------------
    results, val_pred = [], {}
    cv_summary = []
    Xtr_ml = split_ml.train[C.FEATURES]
    Xval_ml = split_ml.val[C.FEATURES]
    for name in TABULAR:
        gs = M.manual_grid_search(Xtr_ml, y["train"], name, n_splits=5)
        pipe = M.fit_best(Xtr_ml, y["train"], name, gs["best_params"])
        pred = pipe.predict(Xval_ml)
        val_pred[name] = pd.Series(pred, index=Xval_ml.index)
        results.append(E.regression_metrics(y["val"], pred, model=name, split="val"))
        cv_summary.append({"model": name, "cv_spearman": gs["best_cv_spearman"],
                           "best_params": str(gs["best_params"])})
        print(f"  {name:14s} cv_rho={gs['best_cv_spearman']:+.3f} "
              f"val_rho={results[-1]['spearman_rho']:+.3f}")

    # -- R5: Keras MLP grid search ------------------------------------------
    Xtr_nn = split_nn.train[C.FEATURES]
    Xval_nn = split_nn.val[C.FEATURES]
    gs_nn = manual_grid_search_nn(Xtr_nn, y["train"], n_splits=3)
    mlp = KerasMLPRegressor(**gs_nn["best_params"]).fit(Xtr_nn, y["train"])
    pred_nn = mlp.predict(Xval_nn)
    val_pred["keras_mlp"] = pd.Series(pred_nn, index=Xval_nn.index)
    results.append(E.regression_metrics(y["val"], pred_nn, model="keras_mlp", split="val"))
    cv_summary.append({"model": "keras_mlp", "cv_spearman": gs_nn["best_cv_spearman"],
                       "best_params": str(gs_nn["best_params"])})
    print(f"  keras_mlp      cv_rho={gs_nn['best_cv_spearman']:+.3f} "
          f"val_rho={results[-1]['spearman_rho']:+.3f}")

    # -- R6: assemble metrics + select --------------------------------------
    metrics = E.metrics_frame(results) if hasattr(E, "metrics_frame") else pd.DataFrame(results)
    metrics = pd.DataFrame(results)
    metrics = metrics.merge(pd.DataFrame(cv_summary), on="model")
    metrics = metrics.sort_values("spearman_rho", ascending=False).reset_index(drop=True)
    metrics.to_csv(C.OUT_MODELS / "p2_val_metrics.csv", index=False)

    selected = metrics.iloc[0]["model"]
    # persist val predictions + selected config
    pred_df = pd.DataFrame(val_pred)
    pred_df[C.TARGET] = y["val"]
    pred_df.to_parquet(C.OUT_MODELS / "val_predictions.parquet")
    import json
    (C.OUT_MODELS / "selected_model.json").write_text(json.dumps({
        "selected_model": selected,
        "best_params": metrics.iloc[0]["best_params"],
        "common_index_rows": len(prep.index),
    }, indent=2), encoding="utf-8")

    # -- report --------------------------------------------------------------
    show = ["model", "cv_spearman", "spearman_rho", "mae", "dir_acc",
            "top_q_dir_acc", "top_q_edge", "bot_q_dir_acc", "bot_q_edge"]
    md = ["# Pipeline 2 — Models & Validation Selection\n"]
    md.append(f"Common index {len(prep.index)} rows; split "
              f"{len(split_ml.train)}/{len(split_ml.val)}/{len(split_ml.test)}. "
              "Tabular models: StandardScaler+SelectKBest inside TimeSeriesSplit "
              "grid search. Keras MLP: rolling-scaled features. Selection by "
              "validation Spearman RHO.\n")
    md.append("## Validation metrics (ranked by Spearman RHO)\n")
    md.append(metrics[show].round(4).to_markdown(index=False) + "\n")
    md.append(f"\n**Selected model: `{selected}`.**\n")
    md.append("\n## Feature ranking (train; f_regression + mutual information)\n")
    md.append(rank.round(4).to_markdown(index=False) + "\n")
    md.append(f"\nWeakest features by average rank: **{', '.join(worst)}** — "
              "SelectKBest inside the CV already drops weak features per model "
              "(tuned `k`); these two are the first candidates to discard.\n")
    (C.OUT_MODELS / "P2_MODELS.md").write_text("\n".join(md), encoding="utf-8")

    # -- figures -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.2))
    order = metrics["model"].tolist()
    ax.bar(order, metrics["spearman_rho"], color="#2980b9")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_ylabel("validation Spearman RHO")
    ax.set_title("Validation rank correlation by model")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    save_fig(fig, "p2_val_spearman")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(rank["feature"][::-1], rank["mutual_info"][::-1], color="#16a085")
    ax.set_xlabel("mutual information with target")
    ax.set_title("Feature ranking (train)")
    save_fig(fig, "p2_feature_ranking")

    print("=" * 68)
    print(f"SELECTED: {selected}  (val Spearman {metrics.iloc[0]['spearman_rho']:+.3f})")
    print("Report: outputs/models/P2_MODELS.md")


if __name__ == "__main__":
    main()
