"""Page 3 — battery simulator."""

from __future__ import annotations

import io

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard import diagrams
from dashboard.guide import guide
from dashboard.theme import ACCENT, ALERT, MUTED, PREDICTED, SIMULATED, style
from dashboard.ui import profile_header, sidebar_profile
from simulator.profiles import run_charge, run_cycles, run_discharge


def render() -> None:
    profile = sidebar_profile()
    st.title("Battery simulator")
    st.caption("Physics-based operation of the configured battery. Every curve on this page is simulated.")
    profile_header(profile)
    guide(
        "FIG. 03",
        "What this simulator does",
        "It operates the configured battery forward in time using an equivalent-circuit model with "
        "a thermal and an ageing term. You set the load, the C-rate, the ambient temperature and "
        "the time step; it integrates terminal voltage, current, SOC and cell temperature. This is "
        "the only page whose curves are entirely generated rather than measured.",
        [
            "Pick an operation — discharge, charge or multi-cycle ageing — and a load type, then "
            "run it. Results persist while you navigate, so you can compare profiles.",
            "Watch the four-panel output: SOC falls, voltage sags under load then recovers, current "
            "is set by the load, and temperature lags both because of thermal mass.",
            "Use Cycle ageing to see capacity fade; a single discharge barely moves SOH and the "
            "page says so instead of pretending otherwise.",
        ],
        read="V_terminal = OCV(SOC) − I·R0 − V_RC. R0 is the instantaneous ohmic drop you see the "
             "moment load is applied; the RC pair is the slower recovery after it. Heat is I²R "
             "loss, removed to ambient through the cell surface.",
        svg=diagrams.equivalent_circuit(),
    )

    with st.container(border=True):
        cols = st.columns(4)
        with cols[0]:
            mode = st.selectbox("Operation", ["Discharge", "Charge", "Cycle ageing"])
        with cols[1]:
            load = st.selectbox("Load type", ["Constant C-rate", "Constant power", "Variable load", "Drive cycle"],
                                disabled=mode != "Discharge")
        with cols[2]:
            c_rate = st.slider("C-rate", 0.1, 3.0, 0.5 if mode != "Charge" else 0.5, 0.1)
        with cols[3]:
            ambient = st.slider("Ambient / °C", -10, 50, int(profile.cell.ambient_temperature_C), 1)

        cols = st.columns(4)
        with cols[0]:
            duration_h = st.slider("Duration / h", 0.25, 24.0, 4.0, 0.25, disabled=mode == "Cycle ageing")
        with cols[1]:
            cycles = st.slider("Cycles", 1, 300, 40, 1, disabled=mode != "Cycle ageing")
        with cols[2]:
            dt_s = st.select_slider("Time step / s", [1, 5, 15, 30, 60, 120], value=15)
        with cols[3]:
            ageing = st.toggle("Include ageing", value=mode == "Cycle ageing",
                               help="Empirical capacity-fade and resistance-growth model.")

    profile.cell.ambient_temperature_C = float(ambient)
    run = st.button("Run simulation", type="primary")

    if not run and "sim_result" not in st.session_state:
        st.info("Set the controls and run a simulation. Results stay available while you navigate.")
        return
    if run:
        with st.spinner("Integrating the model…"):
            if mode == "Charge":
                result = run_charge(profile, c_rate=c_rate, duration_s=duration_h * 3600, dt_s=dt_s)
                st.session_state["sim_result"] = ("charge", result.frame)
            elif mode == "Cycle ageing":
                frame = run_cycles(profile, cycles=cycles, c_rate=c_rate, dt_s=max(dt_s, 30))
                st.session_state["sim_result"] = ("cycles", frame)
            else:
                result = run_discharge(profile, load_type(load), c_rate=c_rate,
                                       power_W=profile.pack_energy_Wh / 4,
                                       duration_s=duration_h * 3600, dt_s=dt_s)
                st.session_state["sim_result"] = ("discharge", result.frame)

    kind, frame = st.session_state["sim_result"]
    if frame.empty:
        st.warning("The simulation ended immediately — check the initial SOC and cut-off voltages.")
        return

    if kind != "cycles" and ageing:
        st.caption("Ageing was enabled, but a single operation changes SOH negligibly; use 'Cycle ageing' to see fade.")

    if kind == "cycles":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=frame.cycle, y=frame.soh * 100, name="Simulated SOH",
                                 line=dict(color=SIMULATED, width=2)))
        style(fig, "Simulated capacity fade over cycles", y_title="SOH / %", x_title="Cycle")
        st.plotly_chart(fig, use_container_width=True, theme=None)
        st.dataframe(frame.round(4), use_container_width=True, hide_index=True)
        st.download_button("Download cycle table (CSV)", frame.to_csv(index=False).encode(),
                           file_name="simulated_cycles.csv", mime="text/csv")
        return

    last = frame.iloc[-1]
    cols = st.columns(4)
    cols[0].metric("Final SOC", f"{last.soc*100:.1f} %")
    cols[1].metric("Final cell voltage", f"{last.voltage_V:.3f} V")
    cols[2].metric("Peak cell temperature", f"{frame.temperature_C.max():.1f} °C")
    cols[3].metric("Energy delivered", f"{np.trapezoid(frame.power_W, frame.time_s)/3600:.1f} Wh")

    fig = make_subplots(rows=2, cols=2, shared_xaxes=False,
                        subplot_titles=("SOC", "Cell voltage", "Cell current", "Cell temperature"))
    fig.add_trace(go.Scatter(x=frame.time_min, y=frame.soc * 100, line=dict(color=ACCENT, width=2), name="SOC"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=frame.time_min, y=frame.voltage_V, line=dict(color=SIMULATED, width=2), name="Voltage"),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=frame.time_min, y=frame.current_A, line=dict(color=PREDICTED, width=1.6), name="Current"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=frame.time_min, y=frame.temperature_C, line=dict(color=ALERT, width=2), name="Temperature"),
                  row=2, col=2)
    fig.update_yaxes(title_text="%", row=1, col=1, gridcolor="#e4e9ef")
    fig.update_yaxes(title_text="V", row=1, col=2, gridcolor="#e4e9ef")
    fig.update_yaxes(title_text="A", row=2, col=1, gridcolor="#e4e9ef")
    fig.update_yaxes(title_text="°C", row=2, col=2, gridcolor="#e4e9ef")
    fig.update_xaxes(title_text="Time / min", row=2, col=1, gridcolor="#e4e9ef")
    fig.update_xaxes(title_text="Time / min", row=2, col=2, gridcolor="#e4e9ef")
    fig.update_layout(template="plotly_white", height=560, showlegend=False,
                      margin=dict(l=50, r=20, t=50, b=40),
                      font=dict(family="Inter, Segoe UI, sans-serif", size=11))
    st.plotly_chart(fig, use_container_width=True, theme=None)

    with st.expander("Simulated samples"):
        st.dataframe(frame.round(4).iloc[::max(1, len(frame) // 400)], use_container_width=True, hide_index=True)

    buf = io.StringIO()
    frame.to_csv(buf, index=False)
    st.download_button("Download simulation (CSV)", buf.getvalue().encode(),
                       file_name="simulation.csv", mime="text/csv")
    st.caption(
        "Thevenin ECM + lumped thermal model. Ageing coefficients are literature-scale estimates, "
        "so absolute fade rates are illustrative; measured fade is analysed on the ageing page."
    )


def load_type(label: str) -> str:
    return {"Constant C-rate": "Constant C-rate", "Constant power": "Constant power",
            "Variable load": "Variable load", "Drive cycle": "Drive cycle"}[label]
