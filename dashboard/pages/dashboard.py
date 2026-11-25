"""Page 1 — operational dashboard."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from dashboard import data_access as da
from dashboard import diagrams
from dashboard.guide import guide
from dashboard.theme import ACCENT, ALERT, DATASET_COLORS, HAIRLINE, NEUTRAL, SIMULATED, style
from dashboard.ui import kpi_row, method_label, profile_header, sidebar_profile, status_chip
from simulator.engine import pack_scale
from simulator.profiles import run_discharge


def render() -> None:
    profile = sidebar_profile()
    st.title("Battery digital twin")
    st.caption("State estimation, health monitoring and remaining-useful-life prediction for lithium-ion cells.")
    profile_header(profile)
    guide(
        "FIG. 01",
        "What this dashboard is",
        "A single screen for the configured pack. The top row is a live readout: the battery's "
        "identity, its state of charge and health, pack voltage and energy, and what the last "
        "simulated discharge did. Below it are the two views you need to trust those numbers — one "
        "simulated discharge curve, and every measured capacity-fade curve in the project.",
        [
            "Read the identity strip: cells in series × parallel, pack voltage, capacity, energy.",
            "Compare the simulated discharge on the left with the measured fade curves on the right.",
            "Use the badge colours — measured, simulated, predicted — to see what kind of number "
            "you are looking at. Nothing on this page is a customer measurement.",
        ],
        read="The simulated curve is one 0.5C discharge from the current SOC. The right-hand chart "
             "plots every audited cell's SOH against its cycle number, coloured by laboratory, with "
             "the 80% end-of-life line marked. A cell that stops early simply ran out of recorded "
             "cycles; it did not recover.",
        svg=diagrams.cell_cutaway(),
    )
    method_label()

    cell = profile.cell
    soh = cell.initial_soh
    st.markdown(f"**Status** {status_chip(soh)}", unsafe_allow_html=True)

    # One representative discharge to show live terminal quantities.
    sim = run_discharge(profile, "Constant C-rate", c_rate=0.5, duration_s=4 * 3600, dt_s=15)
    frame = sim.frame
    if "pack_voltage_V" not in frame.columns:
        frame = pack_scale(frame, profile)
    last = frame.iloc[-1] if not frame.empty else None

    kpi_row(
        [
            ("State of charge", f"{float(cell.initial_soc) * 100:.0f} %", "Configured initial SOC."),
            ("State of health", f"{soh * 100:.1f} %", "Capacity relative to the rated cell capacity."),
            ("Pack voltage", f"{profile.pack_nominal_voltage_V:.0f} V", "Nominal, from the series count."),
            ("Pack energy", f"{profile.pack_energy_Wh / 1000:.2f} kWh", "Nominal, at beginning of life."),
        ]
    )
    kpi_row(
        [
            ("Load current", f"{float(last['pack_current_A']):.0f} A pack" if last is not None else "—",
             "0.5C discharge across the parallel strings."),
            ("Cell temperature", f"{float(last['temperature_C']):.1f} °C" if last is not None else "—",
             "Lumped thermal model, ambient as configured."),
            ("Discharge time", f"{float(last['time_min']):.0f} min" if last is not None else "—",
             "Simulated until the cut-off voltage is reached."),
            ("Model steps", f"{len(frame):,}", "Fixed-step integration of the equivalent circuit."),
        ]
    )

    st.divider()
    left, right = st.columns([1.5, 1], gap="large")

    with left:
        fig = go.Figure()
        if not frame.empty:
            fig.add_trace(go.Scatter(
                x=frame.time_min, y=frame.pack_voltage_V, name="Pack voltage",
                line=dict(color=SIMULATED, width=2.2)))
            fig.add_trace(go.Scatter(
                x=frame.time_min, y=frame.soc, name="State of charge",
                line=dict(color=ACCENT, width=1.6), yaxis="y2"))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 1.02],
                                      showgrid=False, title="State of charge",
                                      tickformat=".0%", linecolor=HAIRLINE))
        style(fig, "Simulated 0.5C discharge", height=400,
              y_title="Pack voltage / V", x_title="Time / min")
        st.plotly_chart(fig, use_container_width=True, theme=None)
        st.caption("Equivalent-circuit model with lumped thermal and empirical ageing terms. "
                   "Simulated, not measured.")

    with right:
        table = da.cycle_table()
        fig = go.Figure()
        if not table.empty:
            for dataset, group in table.groupby("dataset"):
                color = DATASET_COLORS.get(dataset, NEUTRAL)
                first = True
                for cid, cell_rows in group.groupby("cell_id"):
                    cell_rows = cell_rows.sort_values("cycle_index")
                    fig.add_trace(go.Scatter(
                        x=cell_rows.cycle_index, y=cell_rows.soh * 100,
                        name=dataset, legendgroup=dataset, showlegend=first,
                        mode="lines", line=dict(width=1, color=color), opacity=0.55))
                    first = False
        fig.add_hline(y=80, line=dict(color=ALERT, dash="dash", width=1),
                      annotation_text="80% EOL", annotation_position="bottom right",
                      annotation_font=dict(size=10, color=ALERT))
        style(fig, "Measured capacity fade", height=400, y_title="SOH / %", x_title="Cycle")
        st.plotly_chart(fig, use_container_width=True, theme=None)
        st.caption(f"{table.cell_id.nunique() if not table.empty else 0} audited cells from four laboratories.")

    st.divider()
    st.subheader("Where the numbers come from")
    cols = st.columns(3, gap="large")
    with cols[0]:
        st.markdown("**Measured**")
        st.caption("NASA, CALCE, Oxford and Samsung laboratory files, read unchanged from `data/raw`.")
    with cols[1]:
        st.markdown("**Simulated**")
        st.caption("Thevenin equivalent circuit + lumped thermal model + empirical ageing model.")
    with cols[2]:
        st.markdown("**Predicted**")
        st.caption("Models trained with leave-one-battery-out validation; metrics on the AI prediction page.")

    metrics = da.metrics()
    if metrics:
        soc = metrics.get("soc", {}).get("summary", {})
        soh_summary = metrics.get("soh", {}).get("capacity_aware", {}).get("summary", {})
        rul_summary = metrics.get("rul", {}).get("summary", {})
        st.divider()
        st.subheader("Headline validation results")
        rows = []
        for model, value in soc.items():
            rows.append({"Task": "SOC", "Model": model, "MAE": round(value["mae_mean"], 4), "Unit": "fraction SOC"})
        for model, value in soh_summary.items():
            rows.append({"Task": "SOH", "Model": model, "MAE": round(value["mae_mean"], 4), "Unit": "fraction SOH"})
        for model, value in rul_summary.items():
            rows.append({"Task": "RUL", "Model": model, "MAE": round(value["mae_mean"], 1), "Unit": "cycles"})
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption("Leave-one-battery-out means. See the AI prediction and ML lab pages for per-fold detail.")
