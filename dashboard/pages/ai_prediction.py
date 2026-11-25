"""Page 4 — AI prediction (SOC / SOH / RUL)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import data_access as da
from dashboard import diagrams
from dashboard.guide import guide, tab_note
from dashboard.theme import ALERT, MEASURED, MUTED, NEUTRAL, PREDICTED, SIMULATED, style
from dashboard.ui import method_label, sidebar_profile


def render() -> None:
    sidebar_profile()
    st.title("AI prediction")
    st.caption("Learned estimators evaluated on batteries that were never used for training.")
    method_label()
    guide(
        "FIG. 04",
        "What AI prediction covers",
        "Three learned quantities, one per tab: how full the cell is (SOC), how much capacity it "
        "has lost (SOH), and how many cycles it has left before a chosen end-of-life threshold "
        "(RUL). Each tab trains on some batteries and reports error on a battery the model never "
        "saw, so the numbers are forecast errors rather than curve fits.",
        [
            "Pick a tab, then choose the algorithm and the held-out battery to inspect its "
            "trajectory.",
            "Read the metric row first — MAE, RMSE, R² — then the chart, which shows where the model "
            "drifts rather than only how wrong it is on average.",
            "Check the caption: it names the reference used for the truth values and any assumption "
            "behind them.",
        ],
        read="Predicted values are green, measured values are graphite, and dashed lines are model "
             "output. Where a cell has no measurable ground truth — for example it never reaches "
             "80% SOH — the page declines to print a validated RUL instead of extrapolating.",
        svg=diagrams.pipeline(),
    )

    tab_soc, tab_soh, tab_rul = st.tabs(["SOC estimation", "SOH estimation", "RUL prediction"])

    with tab_soc:
        soc_tab()
    with tab_soh:
        soh_tab()
    with tab_rul:
        rul_tab()


# ---------------------------------------------------------------- SOC
def soc_tab() -> None:
    st.subheader("State of charge")
    tab_note("Estimating the charge fraction from voltage, current, temperature and their short "
             "rates of change. The reference is coulomb integration of measured current, so this "
             "tab compares estimators against the best truth available — not against a fuel gauge "
             "that never existed in the files.")
    preds = da.fold_table("soc_predictions.csv")
    folds = da.fold_table("soc_folds.csv")
    if preds.empty or folds.empty:
        st.warning("No SOC predictions found. Run `python models/training/train_all.py`.")
        return

    col_a, col_b = st.columns([1, 2])
    with col_a:
        model = st.selectbox("Algorithm", sorted(preds.model.unique()), key="soc_model")
        cell = st.selectbox("Held-out cell", sorted(preds.cell_id.unique()), key="soc_cell")
    sub = preds[(preds.model == model) & (preds.cell_id == cell)]
    metrics = folds[(folds.model == model) & (folds.held_out == cell)]
    if not metrics.empty:
        row = metrics.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{row.mae*100:.2f} %")
        c2.metric("RMSE", f"{row.rmse*100:.2f} %")
        c3.metric("R²", f"{row.r2:.4f}")

    # A sampled time-series of concatenated cycles reads as a comb; a parity view
    # shows the same error structure legibly.
    sample = sub.sample(min(len(sub), 4000), random_state=0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Parity",
                             line=dict(color=NEUTRAL, width=1, dash="dash")))
    fig.add_trace(go.Scatter(x=sample.y_true * 100, y=sample.y_pred * 100, mode="markers",
                             name=model, marker=dict(size=3.5, color=PREDICTED, opacity=0.35)))
    style(fig, f"Held-out cell {cell}: predicted against reference SOC", height=420,
          y_title="Predicted SOC / %", x_title="Reference SOC / %")
    fig.update_yaxes(range=[-2, 102])
    fig.update_xaxes(range=[-2, 102])
    st.plotly_chart(fig, use_container_width=True, theme=None)

    st.markdown("**Algorithm comparison (mean over the four held-out cells)**")
    table = folds.groupby("model")[["mae", "rmse", "r2"]].mean().sort_values("mae")
    table.index.name = "Model"
    st.dataframe((table * [100, 100, 1]).round({"mae": 2, "rmse": 2, "r2": 4}), use_container_width=True)
    st.caption(
        "Reference SOC is coulomb integration of the measured current over each discharge, normalised by that "
        "cycle's measured capacity. The EKF and the naive coulomb counter use a fixed nominal 2.0 Ah capacity."
    )


# ---------------------------------------------------------------- SOH
def soh_tab() -> None:
    st.subheader("State of health")
    tab_note("Capacity relative to rated capacity, per cycle. The capacity-aware feature set may "
             "use the previous measured discharge capacity; sensor-only may not, and its larger "
             "error is the price of running on BMS signals alone.")
    table = da.cycle_table()
    metrics = da.metrics()
    if table.empty:
        st.warning("No cycle features found.")
        return

    kind = st.radio("Feature set", ["capacity_aware", "sensor_only"], horizontal=True,
                    help="capacity_aware uses the previous measured discharge capacity; sensor_only uses voltage, "
                         "current, temperature and timing only.")
    bundle = da.model_bundle(f"soh_model_{kind}.joblib")
    if bundle is None:
        st.warning("SOH model not found.")
        return

    col_a, col_b = st.columns([1, 2])
    with col_a:
        dataset = st.selectbox("Dataset", sorted(table.dataset.unique()), key="soh_ds")
        cells = sorted(table[table.dataset == dataset].cell_id.unique())
        cell = st.selectbox("Cell", cells, key="soh_cell")
    rows = table[(table.dataset == dataset) & (table.cell_id == cell)].sort_values("cycle_index")
    pred = da.predict_soh(bundle, rows)
    band = da.uncertainty(bundle, rows)

    summary = metrics.get("soh", {}).get(kind, {}).get("summary", {})
    if summary:
        st.markdown("**Leave-one-battery-out performance**")
        cols = st.columns(3)
        best = min(summary.items(), key=lambda kv: kv[1]["mae_mean"])
        cols[0].metric("Best model", best[0])
        cols[1].metric("MAE (mean over folds)", f"{best[1]['mae_mean']*100:.2f} pp")
        cols[2].metric("R² (mean over folds)", f"{best[1]['r2_mean']:.3f}")

    fig = go.Figure()
    if band is not None:
        lo, hi = band
        fig.add_trace(go.Scatter(x=rows.cycle_index, y=hi * 100, line=dict(width=0), showlegend=False,
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=rows.cycle_index, y=lo * 100, fill="tonexty", line=dict(width=0),
                                 fillcolor="rgba(42,157,143,0.15)", name="10–90% tree interval",
                                 hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=rows.cycle_index, y=rows.soh * 100, name="Measured SOH",
                             line=dict(color=MEASURED, width=2)))
    fig.add_trace(go.Scatter(x=rows.cycle_index, y=pred * 100, name="Predicted SOH",
                             line=dict(color=PREDICTED, width=2, dash="dot")))
    fig.add_hline(y=80, line=dict(color=ALERT, dash="dash", width=1))
    style(fig, f"{dataset} / {cell}: measured vs predicted SOH", y_title="SOH / %", x_title="Cycle")
    st.plotly_chart(fig, use_container_width=True, theme=None)
    st.caption("Prediction interval is the 10–90th percentile across the forest's trees, not a calibrated CI.")


# ---------------------------------------------------------------- RUL
def rul_tab() -> None:
    st.subheader("Remaining useful life")
    tab_note("Cycles remaining before SOH crosses the threshold you set. Each threshold has its "
             "own trained artifact because the label definition changes with it. Cells that never "
             "reach the threshold are excluded, not extrapolated.")
    table = da.cycle_table()
    metrics = da.metrics()
    eol = st.slider("End-of-life threshold / % SOH", 70, 90, 80, 1) / 100
    bundle_name = f"rul_model_eol{int(round(eol*100))}.joblib"
    bundle = da.model_bundle(bundle_name)
    if bundle is None:
        st.info(f"No model trained for a {eol*100:.0f}% threshold. Available: {', '.join(da.bundles())}")
        return

    candidates = sorted(table.dataset.unique())
    dataset = st.selectbox("Dataset", candidates, key="rul_ds")
    cells = sorted(table[table.dataset == dataset].cell_id.unique())
    cell = st.selectbox("Cell", cells, key="rul_cell")
    rows = table[(table.dataset == dataset) & (table.cell_id == cell)].sort_values("cycle_index")

    measured = rows.soh.to_numpy()
    below = np.where(measured <= eol)[0]
    if below.size == 0:
        st.warning(
            f"{dataset} / {cell} never reaches {eol*100:.0f}% SOH in the measured data, so a remaining-life "
            "prediction would be an extrapolation with no ground truth. The model's output is not shown as validated."
        )
        return

    actual_eol_cycle = float(rows.cycle_index.iloc[int(below[0])])
    pred = da.predict_soh(bundle, rows)
    rul_pred = np.maximum(pred, 0)
    rul_actual = np.maximum(np.arange(len(rows), 0, -1) - 1, 0).astype(float)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rows.cycle_index, y=rul_actual, name="True remaining cycles",
                             line=dict(color=MEASURED, width=2)))
    fig.add_trace(go.Scatter(x=rows.cycle_index, y=rul_pred, name="Predicted remaining cycles",
                             line=dict(color=PREDICTED, width=2, dash="dot")))
    style(fig, f"{dataset} / {cell}: remaining cycles before {eol*100:.0f}% SOH",
          y_title="Remaining cycles", x_title="Cycle")
    st.plotly_chart(fig, use_container_width=True, theme=None)

    tail = len(rows) - 1
    cols = st.columns(3)
    cols[0].metric("Measured EOL cycle", f"{actual_eol_cycle:.0f}")
    cols[1].metric("Model RUL at last cycle", f"{rul_pred[tail]:.0f} cycles")
    cols[2].metric("Threshold", f"{eol*100:.0f}% SOH")

    summary = metrics.get("rul", {}).get("summary", {})
    if summary:
        st.markdown("**RUL models (leave-one-battery-out)**")
        st.dataframe(pd.DataFrame(summary).T.rename(columns={"mae_mean": "MAE / cycles", "rmse_mean": "RMSE / cycles"}).round(1),
                     use_container_width=True)
    curve = da.fold_table("rul_degradation_curve.csv")
    if not curve.empty:
        st.markdown("**Degradation-curve regression baseline** (linear SOH extrapolation per cell)")
        st.dataframe(curve.round(1), use_container_width=True, hide_index=True)
    st.caption(
        "Cells whose measured trajectory never crosses the threshold are excluded rather than given an optimistic "
        "label. Predictions are only reported where measured degradation history exists."
    )
