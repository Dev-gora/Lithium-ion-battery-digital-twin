"""Page 6 — dataset explorer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard import data_access as da
from dashboard import diagrams
from dashboard.guide import guide
from dashboard.theme import ALERT, MEASURED, MUTED, PREDICTED, style
from dashboard.ui import sidebar_profile
from preprocessing import loaders


def render() -> None:
    sidebar_profile()
    st.title("Dataset explorer")
    st.caption("Original laboratory measurements, read directly from the raw files.")
    guide(
        "FIG. 07",
        "What this page plots",
        "One cycle of one battery, drawn straight from the provider's file. It is the fastest way "
        "to see what a discharge actually looks like before any feature engineering: the voltage "
        "sag, the current profile the cycler applied, and the temperature rise it caused.",
        [
            "Choose a dataset and a cell, then a cycle. The selector is limited to cycles the audit "
            "confirmed exist, so it cannot silently fail.",
            "Read the three stacked channels on a shared time axis — that alignment is what lets "
            "you see voltage respond to current.",
            "Open the data table or export the cycle as CSV when you need the numbers themselves.",
        ],
        read="Oxford entries are characterisation check-ups rather than single cycles: they contain "
             "more than one segment, and the recorded time restarts between them. A missing "
             "temperature trace means the dataset has no temperature channel, not that the sensor "
             "failed.",
        svg=diagrams.signal_strip(),
    )

    dataset = st.selectbox("Dataset", ["NASA", "CALCE", "Oxford", "Samsung"])
    try:
        cells = loaders.dataset_cells(dataset)
    except Exception as error:  # missing audit output
        st.error(f"Audit output unavailable: {error}")
        return

    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        cell = st.selectbox("Battery / cell", cells)
    with col_b:
        cycles = available_cycles(dataset, cell)
        cycle = st.selectbox("Cycle", cycles[:500] if len(cycles) > 500 else cycles,
                             format_func=lambda v: f"cycle {v}")
    with col_c:
        st.caption({
            "NASA": "Discharge or charge segment from the .mat structure.",
            "CALCE": "Full Arbin spreadsheet row block for the selected cycle.",
            "Oxford": "Characterisation check-up (1C and pseudo-OCV segments).",
            "Samsung": "One raw cycle block (10 s sampling).",
        }[dataset])

    frame = load(dataset, cell, cycle)
    if frame.empty:
        st.warning("No samples could be read for this selection.")
        return

    st.markdown(f"**{len(frame):,} samples** · columns: {', '.join(frame.columns)}")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=("Voltage", "Current", "Temperature"), vertical_spacing=0.09)
    fig.add_trace(go.Scatter(x=frame.time_s / 60, y=frame.voltage_V, line=dict(color=MEASURED, width=1.4), name="Voltage"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=frame.time_s / 60, y=frame.current_A, line=dict(color=PREDICTED, width=1.2), name="Current"),
                  row=2, col=1)
    if frame.temperature_C.notna().any():
        fig.add_trace(go.Scatter(x=frame.time_s / 60, y=frame.temperature_C, line=dict(color=ALERT, width=1.2),
                                 name="Temperature"), row=3, col=1)
    fig.update_yaxes(title_text="V", row=1, col=1, gridcolor="#e4e9ef")
    fig.update_yaxes(title_text="A", row=2, col=1, gridcolor="#e4e9ef")
    fig.update_yaxes(title_text="°C", row=3, col=1, gridcolor="#e4e9ef")
    fig.update_xaxes(title_text="Time / min", row=3, col=1, gridcolor="#e4e9ef")
    fig.update_layout(template="plotly_white", height=600, showlegend=False,
                      font=dict(family="Inter, Segoe UI, sans-serif", size=11),
                      margin=dict(l=50, r=20, t=50, b=40))
    st.plotly_chart(fig, use_container_width=True, theme=None)

    with st.expander("Data table"):
        st.dataframe(frame.head(500), use_container_width=True, hide_index=True)
    st.download_button("Export selected cycle (CSV)", frame.to_csv(index=False).encode(),
                       file_name=f"{dataset}_{cell}_{cycle}.csv", mime="text/csv")

    st.divider()
    st.subheader("Dataset audit summary")
    key = {"Samsung": "samsung", "NASA": "nasa", "Oxford": "oxford", "CALCE": "calce"}[dataset]
    info = da.summary(key)
    if info:
        st.json({k: v for k, v in info.items() if k not in {"cells"}} if dataset == "CALCE" else info,
                expanded=False)
    if dataset == "CALCE":
        summary = info.get("cells", {})
        if summary:
            st.dataframe(pd.DataFrame(summary).T, use_container_width=True)


def available_cycles(dataset: str, cell: str) -> list[int]:
    if dataset == "NASA":
        df = pd.read_csv(da.AUDIT / "nasa_discharge_trajectory.csv")
        return sorted(df[df.cell_id == cell].discharge_number.dropna().astype(int).unique().tolist())
    if dataset == "CALCE":
        df = pd.read_csv(da.AUDIT / "calce_cycles.csv")
        return sorted(df[df.cell_id == cell].global_cycle.dropna().astype(int).unique().tolist())
    if dataset == "Oxford":
        df = pd.read_csv(da.AUDIT / "oxford_capacity_trajectory.csv")
        return sorted(df[df.cell_id == cell].characterisation_cycle.dropna().astype(int).unique().tolist())
    df = pd.read_csv(da.AUDIT / "samsung_cycle_summary.csv")
    return sorted(df[(df.battery == cell) & df.valid_discharge].global_cycle.dropna().astype(int).unique().tolist())


def load(dataset: str, cell: str, cycle: int) -> pd.DataFrame:
    if dataset == "NASA":
        full = loaders.nasa_cycles(cell)
        return full[full.discharge_number == cycle].reset_index(drop=True)
    if dataset == "CALCE":
        return loaders.calce_cycle(cell, cycle)
    if dataset == "Oxford":
        return loaders.oxford_checkup(cell, cycle)
    return loaders.samsung_cycle(cell, cycle)
