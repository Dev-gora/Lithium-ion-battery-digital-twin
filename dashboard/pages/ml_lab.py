"""Page 7 — AI / ML lab."""

from __future__ import annotations

import io

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from dashboard import data_access as da
from dashboard import diagrams
from dashboard.guide import guide, tab_note
from dashboard.theme import MEASURED, PREDICTED, style
from dashboard.ui import sidebar_profile


def render() -> None:
    sidebar_profile()
    st.title("AI / ML lab")
    st.caption("Compare algorithms, inspect feature attribution and export trained models.")
    guide(
        "FIG. 08",
        "What the ML lab is for",
        "The working end of the project: compare the four algorithms per task, retrain a model on "
        "one dataset and test it on a cell it has never seen, inspect which features carry the "
        "prediction, and download the trained artifacts. Use it when you want to check a model "
        "rather than consume its output.",
        [
            "Compare algorithms: pick a task and read the metric table, then the per-fold spread.",
            "Train a model: choose a dataset, an algorithm and a held-out cell; the hold-out error "
            "is computed on that cell only.",
            "Feature attribution: SHAP importance for the saved SOH model, ranked by mean absolute "
            "contribution.",
        ],
        read="Each point in the box plot is one fold, i.e. one whole battery removed from training. "
             "A tight box means the model generalises consistently; a wide one means it depends on "
             "a dataset it happened to see.",
        svg=diagrams.validation(),
    )

    tab_compare, tab_train, tab_shap, tab_export = st.tabs(
        ["Compare algorithms", "Train a model", "Feature attribution", "Export"]
    )

    with tab_compare:
        comparison()
    with tab_train:
        trainer()
    with tab_shap:
        shap_view()
    with tab_export:
        exporter()


def comparison() -> None:
    tab_note("Every algorithm trained for one task, scored on batteries excluded from training. "
             "The table gives the mean over folds; the plot gives the spread, which is the part "
             "that tells you whether to trust the mean.")
    metrics = da.metrics()
    if not metrics:
        st.warning("No metrics found. Run `python models/training/train_all.py`.")
        return
    task = st.selectbox("Task", ["SOH", "RUL", "SOC"])
    if task == "SOH":
        kind = st.radio("Feature set", ["capacity_aware", "sensor_only"], horizontal=True)
        summary = metrics["soh"][kind]["summary"]
        folds = da.fold_table(f"soh_folds_{kind}.csv")
    elif task == "RUL":
        summary = metrics["rul"]["summary"]
        folds = da.fold_table("rul_folds.csv")
    else:
        summary = metrics["soc"]["summary"]
        folds = da.fold_table("soc_folds.csv")

    table = pd.DataFrame(summary).T.reset_index().rename(columns={"index": "Model"})
    st.dataframe(table.round(4), use_container_width=True, hide_index=True)

    if not folds.empty:
        value_col = "mae"
        fig = px.box(folds, x="model", y=value_col, points="all",
                     color_discrete_sequence=[PREDICTED])
        fig.update_layout(template="plotly_white", height=380, showlegend=False,
                          margin=dict(l=50, r=20, t=30, b=60),
                          xaxis_title="", yaxis_title="MAE per held-out battery")
        st.plotly_chart(fig, use_container_width=True, theme=None)
        st.caption("Each point is one leave-one-battery-out fold: the spread is the honest measure of generalisation.")

    if task == "SOH":
        preds = da.fold_table(f"soh_predictions_{kind}.csv")
        if not preds.empty:
            model = st.selectbox("Parity plot for", sorted(preds.model.unique()))
            sub = preds[preds.model == model].sample(min(4000, len(preds)), random_state=0)
            fig = px.scatter(sub, x="y_true", y="y_pred", color="dataset", opacity=0.4,
                             labels={"y_true": "Measured SOH", "y_pred": "Predicted SOH"})
            fig.add_trace(go.Scatter(x=[0.1, 1], y=[0.1, 1], mode="lines",
                                     line=dict(color="#8a97a5", dash="dash"), name="parity"))
            fig.update_layout(template="plotly_white", height=420, margin=dict(l=50, r=20, t=30, b=40))
            st.plotly_chart(fig, use_container_width=True, theme=None)


def trainer() -> None:
    tab_note("A live training run you control: one dataset, one algorithm, one battery held out "
             "entirely. Nothing is saved over the shipped artifacts — this is a sandbox.")
    st.markdown("Train a SOH regressor on one dataset, validating on a held-out cell from the same dataset.")
    table = da.cycle_table()
    if table.empty:
        st.warning("Cycle features missing.")
        return
    cols = st.columns(4)
    with cols[0]:
        dataset = st.selectbox("Training dataset", sorted(table.dataset.unique()))
    with cols[1]:
        algorithm = st.selectbox("Algorithm", ["Random Forest", "Gradient Boosting"])
    with cols[2]:
        holdout = st.selectbox("Hold-out cell", sorted(table[table.dataset == dataset].cell_id.unique()))
    with cols[3]:
        test_size = st.slider("Max training rows", 200, 4000, 2000, 100)

    features = ["cycle_index", "voltage_mean_V", "voltage_min_V", "voltage_max_V",
                "current_absmean_A", "temperature_mean_C", "temperature_max_C", "duration_s"]
    subset = table[table.dataset == dataset].dropna(subset=["soh"])
    train = subset[subset.cell_id != holdout].sort_values("cycle_index").tail(test_size)
    test = subset[subset.cell_id == holdout]
    if train.empty or test.empty:
        st.warning("Not enough rows.")
        return
    med = train[features].median(numeric_only=True)
    model = (RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
             if algorithm == "Random Forest"
             else GradientBoostingRegressor(random_state=42))
    model.fit(train[features].fillna(med).to_numpy(), train.soh.to_numpy())
    pred = model.predict(test[features].fillna(med).to_numpy())

    c1, c2, c3 = st.columns(3)
    c1.metric("Hold-out MAE", f"{mean_absolute_error(test.soh, pred)*100:.2f} pp")
    c2.metric("Hold-out R²", f"{r2_score(test.soh, pred):.3f}")
    c3.metric("Rows", f"{len(train):,} train / {len(test):,} test")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test.cycle_index, y=test.soh * 100, name="Measured", line=dict(color=MEASURED, width=2)))
    fig.add_trace(go.Scatter(x=test.cycle_index, y=pred * 100, name="Predicted", line=dict(color=PREDICTED, width=2, dash="dot")))
    style(fig, f"{dataset}: {holdout} held out entirely", y_title="SOH / %", x_title="Cycle")
    st.plotly_chart(fig, use_container_width=True, theme=None)
    st.session_state["lab_model"] = {"model": model, "features": features, "impute": med.to_dict()}


def shap_view() -> None:
    tab_note("Why the saved SOH model says what it says. Mean absolute SHAP value per feature, "
             "measured on a sample of the audited cycle table. These are associations in this "
             "dataset, not causal effects.")
    table = da.cycle_table()
    bundle = da.model_bundle("soh_model_capacity_aware.joblib")
    if bundle is None or table.empty:
        st.warning("Train the SOH model first.")
        return
    try:
        import shap
    except ImportError:
        st.info("Install `shap` to enable explanations.")
        return

    rows = table.dropna(subset=["soh"]).sample(min(300, len(table)), random_state=1)
    features = bundle["features"]
    med = pd.Series(bundle["impute"])
    x = rows.reindex(columns=features).fillna(med.reindex(features))
    explainer = shap.TreeExplainer(bundle["model"])
    values = explainer.shap_values(x.to_numpy())
    importance = pd.DataFrame({
        "Feature": features,
        "Mean |SHAP|": np.abs(values).mean(axis=0),
    }).sort_values("Mean |SHAP|", ascending=False)
    fig = px.bar(importance, x="Mean |SHAP|", y="Feature", orientation="h", color_discrete_sequence=[MEASURED])
    fig.update_layout(template="plotly_white", height=380, margin=dict(l=150, r=20, t=30, b=40),
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True, theme=None)
    st.caption("SHAP values describe the saved SOH model's behaviour on the audited cycle table; "
               "they are associations, not causal effects.")


def exporter() -> None:
    tab_note("Download the trained estimator. Each bundle carries the model, the ordered feature "
             "list and the imputation medians, so a prediction can be reproduced elsewhere "
             "without re-running training.")
    bundles = da.bundles()
    if not bundles:
        st.warning("No saved models.")
        return
    name = st.selectbox("Model artifact", list(bundles))
    path = da.SAVED / name
    st.caption(f"{path.stat().st_size/1024:.0f} kB")
    with open(path, "rb") as handle:
        st.download_button("Download .joblib", handle.read(), file_name=name,
                           mime="application/octet-stream")
    st.caption("Models are scikit-learn / XGBoost estimators with the feature list and imputation values bundled.")
