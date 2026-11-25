"""Page — data sources: the raw laboratory files, browsable by hand."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard import data_access as da
from dashboard import diagrams
from dashboard.guide import guide
from dashboard.theme import ACCENT, ALERT, MEASURED, PREDICTED
from dashboard.ui import sidebar_profile

RAW = da.PROJECT_ROOT / "data" / "raw"

SETS = {
    "NASA": {
        "provider": "NASA Prognostics Center of Excellence",
        "chemistry": "LCO 18650, 2 Ah",
        "root": RAW / "NASA",
        "cycle_label": "discharge number",
        "note": "One .mat structure per cell. Each discharge and charge is a separate record in the structure.",
    },
    "CALCE": {
        "provider": "University of Maryland CALCE",
        "chemistry": "LCO prismatic, 1.1–1.5 Ah",
        "root": RAW / "CALCE",
        "cycle_label": "global cycle",
        "note": "Arbin workbooks with one sheet per channel. The archive carries no temperature channel.",
    },
    "Oxford": {
        "provider": "University of Oxford",
        "chemistry": "pouch, ~0.74 Ah",
        "root": RAW / "Oxford",
        "cycle_label": "check-up",
        "note": "Characterisation check-ups every ~100 drive cycles: a 1C cycle plus a pseudo-OCV sweep.",
    },
    "Samsung": {
        "provider": "Brno University of Technology / UCT Prague",
        "chemistry": "NMC 18650, 3.4 Ah",
        "root": RAW / "additional" / "Samsung_INR18650_35E",
        "cycle_label": "global cycle",
        "note": "Tab-separated cycling files, C/2 charge and discharge at room temperature, 10 s sampling.",
    },
}

VARIABLES = [("voltage_V", "Voltage", "V"), ("current_A", "Current", "A"),
             ("temperature_C", "Temperature", "°C")]
COLORS = {"voltage_V": MEASURED, "current_A": PREDICTED, "temperature_C": ALERT}


def render() -> None:
    sidebar_profile()
    st.title("Data sources")
    st.caption("The original laboratory files, read directly. Nothing on this page is simulated or model output.")
    guide(
        "FIG. 06",
        "What this page shows",
        "The data behind every model, opened by hand. Choose a laboratory and you get three "
        "levels of detail: the per-cycle ledger the models were trained on, the raw samples of a "
        "single cycle plotted channel by channel, and the file inventory on disk. This is the "
        "page to use when you want to verify a number rather than read a summary.",
        [
            "Pick NASA, CALCE, Oxford or Samsung. The readout names the provider, the chemistry "
            "and the folder the files are read from.",
            "In the cycle ledger choose cells and inspect, filter or download the exact table used "
            "for training.",
            "In raw samples choose a cell and a cycle, then read voltage, current and temperature "
            "as recorded — with sample count, duration, statistics and missing values.",
        ],
        read="A ledger row is one cycle: measured capacity, SOH, mean and extreme voltages, current "
             "and temperature. A raw-sample row is one instant of one cycle. If a channel is "
             "absent — CALCE has no thermocouple — it is shown as missing rather than filled in.",
        svg=diagrams.dataset_map(),
    )

    dataset = st.radio("Dataset", list(SETS), horizontal=True, key="src_dataset")
    spec = SETS[dataset]
    try:
        location = spec["root"].relative_to(da.PROJECT_ROOT).as_posix()
    except ValueError:
        location = str(spec["root"])

    st.markdown(
        f"<div class='readout'><span class='swatch' style='background:{ACCENT}'></span>"
        f"<b>{spec['provider']}</b><span>{spec['chemistry']}</span><i>·</i>"
        f"<span>{location}</span></div>",
        unsafe_allow_html=True,
    )
    st.caption(spec["note"])

    table = da.ledger(dataset)
    if table.empty:
        st.warning(f"No audit table found for {dataset}. Run the stage 1 audit first.")
        return
    cell_col, cycle_col = da.CYCLE_KEYS[dataset]
    cells = sorted(table[cell_col].dropna().unique().tolist())

    ledger_section(dataset, table, cell_col, cycle_col, cells, spec)
    st.divider()
    samples_section(dataset, cells, cycle_col, spec)
    st.divider()
    files_section(dataset, spec)


def ledger_section(dataset, table, cell_col, cycle_col, cells, spec) -> None:
    st.subheader("Cycle ledger")
    st.caption(
        f"{len(table):,} rows from `{da.LEDGERS[dataset]}` — one row per cycle, "
        f"the same table the models were trained on."
    )
    picked = st.multiselect("Cells to show", cells, default=cells[: min(2, len(cells))],
                            key=f"ledger_cells_{dataset}")
    view = table[table[cell_col].isin(picked)] if picked else table.iloc[0:0]
    st.dataframe(view, use_container_width=True, hide_index=True, height=320)
    cols = st.columns([1, 1, 2])
    cols[0].metric("Rows shown", f"{len(view):,}")
    cols[1].metric("Columns", f"{table.shape[1]}")
    with cols[2]:
        st.download_button("Download this slice (CSV)", view.to_csv(index=False).encode(),
                           file_name=f"{dataset.lower()}_ledger.csv", mime="text/csv",
                           key=f"ledger_dl_{dataset}")


def samples_section(dataset, cells, cycle_col, spec) -> None:
    st.subheader("Raw samples")
    cols = st.columns([1.2, 1, 1.4])
    with cols[0]:
        cell = st.selectbox("Battery / cell", cells, key=f"raw_cell_{dataset}")
    cycles = da.cycle_options(dataset, cell)
    with cols[1]:
        cycle = st.selectbox(spec["cycle_label"].capitalize(), cycles, key=f"raw_cycle_{dataset}")
    with cols[2]:
        st.caption(f"Read from the raw file for {dataset} / {cell}, {spec['cycle_label']} {cycle}.")

    frame = da.raw_cycle(dataset, cell, int(cycle))
    if frame.empty:
        st.warning("The loader returned no samples for this cycle. Try another cell or cycle.")
        return

    if "segment" in frame.columns:
        segments = frame.segment.dropna().unique().tolist()
        if len(segments) > 1:
            choice = st.radio("Segment", segments, horizontal=True, key=f"raw_segment_{dataset}")
            frame = frame[frame.segment == choice].reset_index(drop=True)

    present = [(c, label, unit) for c, label, unit in VARIABLES
               if c in frame.columns and frame[c].notna().any()]

    if present:
        fig = make_subplots(rows=len(present), cols=1, shared_xaxes=True,
                            subplot_titles=[label for _, label, _ in present],
                            vertical_spacing=0.08)
        stride = max(1, len(frame) // 4000)
        shown = frame.iloc[::stride]
        for row, (column, label, unit) in enumerate(present, start=1):
            fig.add_trace(
                go.Scatter(x=shown.time_s / 60.0, y=shown[column], name=label,
                           line=dict(color=COLORS[column], width=1.5)),
                row=row, col=1,
            )
            fig.update_yaxes(title_text=unit, row=row, col=1, gridcolor="#DFE6EC")
        fig.update_xaxes(title_text="Time / min", row=len(present), col=1, gridcolor="#DFE6EC")
        fig.update_layout(template="plotly_white", height=170 + 150 * len(present),
                          showlegend=False, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                          margin=dict(l=60, r=24, t=46, b=44),
                          font=dict(family="'IBM Plex Sans', system-ui, sans-serif", size=11,
                                    color="#131C26"))
        st.plotly_chart(fig, use_container_width=True, theme=None)
        if stride > 1:
            st.caption(f"{len(frame):,} samples; the plot shows every {stride}th one.")

    numeric = frame.select_dtypes(include="number")
    cols = st.columns(3)
    cols[0].metric("Samples", f"{len(frame):,}")
    cols[1].metric("Columns", f"{frame.shape[1]}")
    if "time_s" in frame.columns:
        span = float(frame.time_s.max() - frame.time_s.min())
        cols[2].metric("Duration", f"{span / 60:.1f} min")

    with st.expander("Column statistics and missing values"):
        summary = numeric.describe().T[["count", "mean", "std", "min", "max"]].round(4)
        summary["missing"] = len(frame) - numeric.count()
        st.dataframe(summary, use_container_width=True)

    st.markdown("**Samples**")
    st.dataframe(frame.head(500), use_container_width=True, hide_index=True, height=260)
    st.download_button("Download raw samples (CSV)", frame.to_csv(index=False).encode(),
                       file_name=f"{dataset.lower()}_{cell}_{cycle}.csv", mime="text/csv",
                       key=f"raw_dl_{dataset}")


def files_section(dataset, spec) -> None:
    root = spec["root"]
    with st.expander("Files on disk"):
        rows = []
        for path in sorted(root.rglob("*")):
            if path.is_file():
                stat = path.stat()
                rows.append({
                    "File": path.relative_to(root).as_posix(),
                    "Size / MB": round(stat.st_size / 1e6, 3),
                })
        if not rows:
            st.info("No files found.")
            return
        inventory = pd.DataFrame(rows)
        st.dataframe(inventory, use_container_width=True, hide_index=True, height=280)
        st.caption(f"{len(inventory)} files · {inventory['Size / MB'].sum():,.1f} MB total. "
                   "Files are opened read-only; nothing on this page writes to the raw data.")
