"""Page 5 — battery ageing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import data_access as da
from dashboard import diagrams
from dashboard.guide import guide
from dashboard.theme import ALERT, MEASURED, MUTED, NEUTRAL, SIMULATED, style
from dashboard.ui import sidebar_profile


def render() -> None:
    sidebar_profile()
    st.title("Battery ageing")
    st.caption("Measured capacity fade, impedance growth and end-of-life projection from the audited datasets.")
    guide(
        "FIG. 05",
        "What this page measures",
        "Degradation as it actually happened in the laboratory files: every cell's capacity-based "
        "SOH against cycle number, its discharge capacity in amp-hours, NASA's impedance "
        "measurements, and a linear projection to end of life. Nothing here is simulated.",
        [
            "Restrict the dataset selection to compare chemistries, then move the end-of-life "
            "threshold to see which cells cross it and when.",
            "Use the table for numbers — initial and final SOH, measured EOL cycle, fade rate in "
            "percentage points per cycle.",
            "Pick a cell under Projected end of life to extrapolate its observed fade to the "
            "threshold.",
        ],
        read="The early, near-linear part of a fade curve is usually SEI growth; the sharp bend "
             "afterwards is the knee, where plating or dry-out takes over. A linear projection "
             "through a knee is optimistic, which is why the measured EOL cycle is shown beside it "
             "and the model-based RUL lives on the AI prediction page.",
        svg=diagrams.ageing_curve(),
    )

    table = da.cycle_table()
    if table.empty:
        st.warning("Cycle features are missing. Run the audit and preprocessing stages first.")
        return

    dataset = st.multiselect("Datasets", sorted(table.dataset.unique()), default=sorted(table.dataset.unique()))
    threshold = st.slider("End-of-life threshold / % SOH", 70, 90, 80, 1) / 100

    subset = table[table.dataset.isin(dataset)]
    fig = go.Figure()
    for (ds, cell), group in subset.groupby(["dataset", "cell_id"]):
        group = group.sort_values("cycle_index")
        fig.add_trace(go.Scatter(x=group.cycle_index, y=group.soh * 100, name=f"{ds}/{cell}",
                                 mode="lines", line=dict(width=1.4)))
    fig.add_hline(y=threshold * 100, line=dict(color=ALERT, dash="dash", width=1),
                  annotation_text=f"EOL {threshold*100:.0f}%", annotation_position="top left")
    style(fig, "Capacity-based SOH trajectories", y_title="SOH / %", x_title="Cycle", height=430)
    st.plotly_chart(fig, use_container_width=True, theme=None)

    rows = []
    for (ds, cell), group in subset.groupby(["dataset", "cell_id"]):
        g = group.sort_values("cycle_index")
        below = g[g.soh <= threshold]
        eol_cycle = float(below.cycle_index.iloc[0]) if not below.empty else np.nan
        n = len(g)
        # Linear fade rate from a least-squares fit over the observed history.
        slope = np.polyfit(g.cycle_index.to_numpy(float), g.soh.to_numpy(float), 1)[0] if n > 3 else np.nan
        rows.append({
            "Dataset": ds, "Cell": cell, "Observations": n,
            "Initial SOH / %": round(float(g.soh.iloc[0]) * 100, 2),
            "Final SOH / %": round(float(g.soh.iloc[-1]) * 100, 2),
            "Measured EOL cycle": None if np.isnan(eol_cycle) else int(eol_cycle),
            "Fade rate / pp per cycle": round(float(slope) * 100, 4) if np.isfinite(slope) else None,
        })
    ageing = pd.DataFrame(rows)
    st.dataframe(ageing, use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Capacity retention")
        fig = go.Figure()
        for (ds, cell), group in subset.groupby(["dataset", "cell_id"]):
            g = group.sort_values("cycle_index")
            if g.capacity_Ah.dropna().empty:
                continue
            fig.add_trace(go.Scatter(x=g.cycle_index, y=g.capacity_Ah, name=f"{ds}/{cell}", mode="lines",
                                     line=dict(width=1.3)))
        style(fig, "Discharge capacity vs cycle", y_title="Capacity / Ah", x_title="Cycle")
        st.plotly_chart(fig, use_container_width=True, theme=None)
    with right:
        st.subheader("Resistance growth (NASA EIS)")
        eis = pd.read_csv(da.AUDIT / "nasa_eis_summary.csv") if (da.AUDIT / "nasa_eis_summary.csv").exists() else pd.DataFrame()
        if eis.empty:
            st.info("No impedance records found.")
        else:
            fig = go.Figure()
            for cell, group in eis.groupby("cell_id"):
                group = group.sort_values("discharge_number")
                fig.add_trace(go.Scatter(x=group.discharge_number, y=group.Re_Ohm * 1000, name=f"{cell} Re",
                                         line=dict(width=1.4)))
                fig.add_trace(go.Scatter(x=group.discharge_number, y=group.Rct_Ohm * 1000, name=f"{cell} Rct",
                                         line=dict(width=1.1, dash="dot"), opacity=0.7))
            style(fig, "Ohmic (solid) and charge-transfer (dotted) resistance", y_title="Resistance / mΩ",
                  x_title="Discharge cycle")
            st.plotly_chart(fig, use_container_width=True, theme=None)

    st.divider()
    st.subheader("Projected end of life")
    candidates = ageing[ageing["Fade rate / pp per cycle"].notna() & (ageing["Fade rate / pp per cycle"] < 0)]
    if candidates.empty:
        st.info("No cell shows a measurable fade trend.")
        return
    labels = [f"{r.Dataset}/{r.Cell}" for r in candidates.itertuples()]
    pick = st.selectbox("Cell for projection", labels)
    row = candidates.iloc[labels.index(pick)]
    g = subset[(subset.dataset == row.Dataset) & (subset.cell_id == row.Cell)].sort_values("cycle_index")
    slope = row["Fade rate / pp per cycle"] / 100
    last_cycle, last_soh = float(g.cycle_index.iloc[-1]), float(g.soh.iloc[-1])
    if slope < 0:
        projected = last_cycle + (threshold - last_soh) / slope
    else:
        projected = np.nan
    cols = st.columns(3)
    cols[0].metric("Current SOH", f"{last_soh*100:.1f} %")
    cols[1].metric("Current cycle", f"{last_cycle:.0f}")
    cols[2].metric("Projected EOL cycle", "n/a" if not np.isfinite(projected) else f"{projected:.0f}",
                   help="Linear extrapolation of the observed fade; assumes the current fade rate continues.")
    st.caption("A linear projection is a convenience estimate. The AI prediction page gives model-based RUL with "
               "leave-one-battery-out error.")
