"""Pipeline 3 — final out-of-sample test metrics + SHAP feature importance.

Run:  python pipelines/p3_test_shap.py   (requires p2 selected_model.json)

Retrains the selected model on train (deterministic), evaluates the regression
metrics on validation AND the sealed test set, and computes SHAP feature
importance on the test set. Everything is frozen from p2. No tuning.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from aps import config as C
from aps import evaluate as E
from aps.data import load_dataset
from aps.runner import fit_predict
from aps.plotting import apply_style, save_fig


def _shap_importance(name, model, split_ml, split_nn):
    """Return (feature_names, mean_abs_shap, shap_values, X_test_used)."""
    is_nn = name == "keras_mlp"
    split = split_nn if is_nn else split_ml
    X_tr = split.train[C.FEATURES]
    X_te = split.test[C.FEATURES]
    if name == "linear" or name == "svr":
        # linear pipeline: explain the final estimator on transformed features
        pipe = model
        scaler, select, est = (pipe.named_steps["scaler"], pipe.named_steps["select"],
                               pipe.named_steps["model"])
        feat = np.array(C.FEATURES)[select.get_support()]
        Xtr_t = select.transform(scaler.transform(X_tr))
        Xte_t = select.transform(scaler.transform(X_te))
        if name == "linear":
            expl = shap.LinearExplainer(est, Xtr_t)
            sv = expl.shap_values(Xte_t)
        else:
            expl = shap.KernelExplainer(est.predict, shap.sample(Xtr_t, 50, random_state=C.SEED))
            sv = expl.shap_values(Xte_t, nsamples=100)
        return feat, np.abs(sv).mean(0), sv, Xte_t
    elif name in ("random_forest", "lightgbm"):
        pipe = model
        scaler, select, est = (pipe.named_steps["scaler"], pipe.named_steps["select"],
                               pipe.named_steps["model"])
        feat = np.array(C.FEATURES)[select.get_support()]
        Xtr_t = select.transform(scaler.transform(X_tr))
        Xte_t = select.transform(scaler.transform(X_te))
        expl = shap.TreeExplainer(est)
        sv = expl.shap_values(Xte_t)
        return feat, np.abs(sv).mean(0), sv, Xte_t
    else:  # keras_mlp
        feat = np.array(C.FEATURES)
        bg = shap.sample(X_tr.to_numpy(), 50, random_state=C.SEED)
        expl = shap.KernelExplainer(model.predict, bg)
        sv = expl.shap_values(X_te.to_numpy()[:100], nsamples=100)
        sv = np.array(sv)
        return feat, np.abs(sv).mean(0), sv, X_te.to_numpy()[:100]


def main() -> None:
    C.ensure_dirs()
    apply_style()

    sel = json.loads((C.OUT_MODELS / "selected_model.json").read_text())
    name, params = sel["selected_model"], sel["best_params"]

    ds = load_dataset()
    fp = fit_predict(name, params, ds, parts=("val", "test"))
    y = fp["y"]

    rows = []
    for part in ("val", "test"):
        rows.append(E.regression_metrics(y[part], fp["pred"][part].to_numpy(),
                                         model=name, split=part))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(C.OUT_MODELS / "p3_test_metrics.csv", index=False)

    # per-quantile directional accuracy on test
    dq = E.directional_accuracy_by_quantile(y["test"], fp["pred"]["test"].to_numpy())
    dq.to_csv(C.OUT_MODELS / "p3_test_quantile_diracc.csv", index=False)

    # persist predictions for the trading pipeline
    pd.DataFrame({"pred": fp["pred"]["test"], C.TARGET: y["test"]}
                 ).to_parquet(C.OUT_MODELS / "test_predictions.parquet")
    pd.DataFrame({"pred": fp["pred"]["val"], C.TARGET: y["val"]}
                 ).to_parquet(C.OUT_MODELS / "val_predictions_selected.parquet")

    # -- SHAP ---------------------------------------------------------------
    feat, mean_abs, sv, X_used = _shap_importance(name, fp["model"],
                                                  fp["split_ml"], fp["split_nn"])
    shap_df = pd.DataFrame({"feature": feat, "mean_abs_shap": mean_abs}
                           ).sort_values("mean_abs_shap", ascending=False)
    shap_df.to_csv(C.OUT_MODELS / "p3_shap_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(shap_df["feature"][::-1], shap_df["mean_abs_shap"][::-1], color="#8e44ad")
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title(f"SHAP feature importance on test ({name})")
    save_fig(fig, "p3_shap_importance")

    # -- report --------------------------------------------------------------
    show = ["split", "mae", "spearman_rho", "dir_acc", "top_q_dir_acc",
            "top_q_edge", "bot_q_dir_acc", "bot_q_edge"]
    md = ["# Pipeline 3 — Test Metrics & SHAP\n"]
    md.append(f"Selected model **{name}** retrained on train; evaluated on "
              "validation and the sealed test set.\n")
    md.append("## Regression metrics\n")
    md.append(metrics[show].round(4).to_markdown(index=False) + "\n")
    md.append("\n## Test directional accuracy by predicted-return quantile\n")
    md.append(dq.round(4).to_markdown(index=False) + "\n")
    md.append("\n## SHAP feature importance (test)\n")
    md.append(shap_df.round(5).to_markdown(index=False) + "\n")
    (C.OUT_MODELS / "P3_TEST_SHAP.md").write_text("\n".join(md), encoding="utf-8")

    print("=" * 68)
    print(f"TEST metrics ({name}):")
    print(metrics[show].round(4).to_string(index=False))
    print("\nTop SHAP features:", ", ".join(shap_df["feature"].head(5)))
    print("Report: outputs/models/P3_TEST_SHAP.md")


if __name__ == "__main__":
    main()
