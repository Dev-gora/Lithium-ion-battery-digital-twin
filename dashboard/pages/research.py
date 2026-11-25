"""Page 8 — research analytics and report exports."""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages

from dashboard import data_access as da
from dashboard import diagrams
from dashboard.guide import guide, tab_note
from dashboard.ui import sidebar_profile

REPORT_FILES = [
    ("Dataset inventory", "dataset_inventory.md"),
    ("Dataset quality report", "dataset_quality_report.md"),
    ("Preprocessing summary", "preprocessing_summary.md"),
]


def render() -> None:
    sidebar_profile()
    st.title("Research analytics")
    st.caption("Reports, comparison tables and exports generated from the audited data and validated models.")
    guide(
        "FIG. 09",
        "What this page is for",
        "The write-up layer: the audit reports that were produced while the data was being "
        "qualified, one consolidated model-comparison table, the publication figures, and a PDF "
        "assembled on demand. Everything is generated from the artifacts on disk, so it cannot "
        "drift from what the app is showing.",
        [
            "Reports: read the dataset inventory, quality report and preprocessing summary in place, "
            "or download the markdown.",
            "Tables: every task and model's MAE, RMSE and R² in one frame, exportable as CSV.",
            "Export: build a dated PDF containing the validated metrics and every figure.",
        ],
        read="The PDF footer states data provenance and that simulated values are model output, "
             "because a printed page loses the colour coding the app uses for measured, "
             "simulated and predicted.",
        svg=diagrams.pipeline([
            ("AUDIT", "reports\nmarkdown"),
            ("MODELS", "metrics\njson"),
            ("TABLES", "comparison\ncsv"),
            ("FIGURES", "publication\npng"),
            ("EXPORT", "dated\npdf"),
        ]),
    )

    tabs = st.tabs(["Reports", "Tables", "Figures", "Export"])
    with tabs[0]:
        for title, filename in REPORT_FILES:
            path = da.REPORTS / filename
            with st.expander(title, expanded=title.startswith("Dataset inventory")):
                if path.exists():
                    st.markdown(path.read_text(encoding="utf-8"))
                    st.download_button(f"Download {filename}", path.read_bytes(), file_name=filename,
                                       mime="text/markdown", key=f"dl_{filename}")
                else:
                    st.info(f"{filename} has not been generated yet.")
    with tabs[1]:
        tables()
    with tabs[2]:
        figures()
    with tabs[3]:
        exports()


def tables() -> None:
    tab_note("One row per task and model, with the leave-one-battery-out mean of MAE, RMSE and R². "
             "SOH appears twice because the capacity-aware and sensor-only feature sets are "
             "different claims about what an estimator can know.")
    metrics = da.metrics()
    if not metrics:
        st.warning("No metrics yet.")
        return
    st.subheader("Model comparison")
    rows = []
    for task in ["soc", "soh", "rul"]:
        node = metrics.get(task, {})
        if task == "soh":
            for kind, payload in node.items():
                for model, value in payload.get("summary", {}).items():
                    rows.append({"Task": f"SOH ({kind})", "Model": model,
                                 "MAE": value["mae_mean"], "RMSE": value["rmse_mean"],
                                 "R²": value.get("r2_mean")})
        else:
            for model, value in node.get("summary", {}).items():
                rows.append({"Task": task.upper(), "Model": model,
                             "MAE": value["mae_mean"], "RMSE": value["rmse_mean"],
                             "R²": value.get("r2_mean")})
    table = pd.DataFrame(rows)
    st.dataframe(table.round(4), use_container_width=True, hide_index=True)
    st.download_button("Download comparison table (CSV)", table.to_csv(index=False).encode(),
                       file_name="model_comparison.csv", mime="text/csv")

    anomaly = metrics.get("anomaly", {})
    if anomaly:
        st.subheader("Anomaly detection")
        st.write(anomaly)
    st.caption(f"Training seed {metrics.get('seed')} · validation: {metrics.get('validation')} · "
               f"runtime {metrics.get('runtime_s')} s")


def figures() -> None:
    tab_note("The static figures written by the analysis stage into data/reports/figures. They are "
             "the same images embedded in the PDF, kept as PNG so they can go into a paper.")
    figs = sorted(da.FIGS.glob("*.png"))
    if not figs:
        st.info("No publication figures found.")
        return
    for path in figs:
        st.image(str(path), caption=path.stem.replace("_", " "), use_container_width=True)


def exports() -> None:
    tab_note("Build a dated PDF from the current metrics plus the figures, and download the raw "
             "prediction tables. The PDF is regenerated on demand, never cached, so it always "
             "matches metrics.json.")
    st.subheader("Generate a PDF report")
    st.caption("Assembles the current metrics, dataset summary and the publication figures into one PDF.")
    if st.button("Build PDF", type="primary"):
        buffer = io.BytesIO()
        metrics = da.metrics()
        with PdfPages(buffer) as pdf:
            fig = plt.figure(figsize=(8.3, 11.7))
            fig.text(0.08, 0.92, "Lithium-Ion Battery Digital Twin", fontsize=20, weight="bold")
            fig.text(0.08, 0.885, "Automated research report", fontsize=12, color="#5b6b7b")
            fig.text(0.08, 0.855, datetime.now().strftime("%Y-%m-%d"), fontsize=10, color="#5b6b7b")
            y = 0.79
            fig.text(0.08, y, "Validated models (leave-one-battery-out)", fontsize=12, weight="bold")
            y -= 0.03
            for task in ["soc", "soh", "rul"]:
                node = metrics.get(task, {})
                if task == "soh":
                    for kind, payload in node.items():
                        for model, value in payload.get("summary", {}).items():
                            fig.text(0.10, y, f"SOH {kind} · {model}: MAE {value['mae_mean']:.4f}", fontsize=9)
                            y -= 0.021
                else:
                    for model, value in node.get("summary", {}).items():
                        fig.text(0.10, y, f"{task.upper()} · {model}: MAE {value['mae_mean']:.4f}", fontsize=9)
                        y -= 0.021
            fig.text(0.08, 0.06,
                     "Measured data: NASA PCoE, CALCE, Oxford, Samsung INR18650-35E. "
                     "Simulated values are model output, not measurements.",
                     fontsize=8, color="#5b6b7b")
            plt.axis("off")
            pdf.savefig(fig)
            plt.close(fig)
            for path in sorted(da.FIGS.glob("*.png")):
                image = plt.imread(path)
                fig = plt.figure(figsize=(11.7, 8.3))
                plt.imshow(image)
                plt.axis("off")
                plt.title(path.stem.replace("_", " "), fontsize=10)
                pdf.savefig(fig)
                plt.close(fig)
        st.download_button("Download PDF", buffer.getvalue(),
                           file_name="battery_digital_twin_report.pdf", mime="application/pdf")
    st.divider()
    st.subheader("CSV exports")
    for name in ["soh_predictions_capacity_aware.csv", "soc_predictions.csv", "rul_predictions.csv",
                 "rul_degradation_curve.csv", "metrics.json"]:
        path = da.RESULTS / name
        if path.exists():
            st.download_button(name, path.read_bytes(), file_name=name, key=f"export_{name}")
