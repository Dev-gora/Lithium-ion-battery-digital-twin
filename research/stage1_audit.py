"""Stage 1 dataset audit for the lithium-ion battery digital twin.

Reads only raw experimental files, computes measured summaries and creates
figures/report inputs. It does not train models and does not alter raw data.

Run from the project root:
    python research/stage1_audit.py --datasets nasa oxford energy samsung
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw"
AUDIT = PROJECT_ROOT / "data" / "processed" / "audit"
FIGS = PROJECT_ROOT / "data" / "reports" / "figures"
for directory in (AUDIT, FIGS):
    directory.mkdir(parents=True, exist_ok=True)

COLORS = ["#1f5a7a", "#c44e52", "#2a9d8f", "#e9a03b", "#6a4c93", "#4d4d4d", "#8c564b", "#17a2b8"]
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#c9d1d9",
        "axes.grid": True,
        "grid.color": "#e6e9ee",
        "grid.linewidth": 0.8,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def finite(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def robust_stats(values: Iterable[float]) -> dict[str, float | None]:
    arr = finite(values)
    if arr.size == 0:
        return {"n": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
    }


def med_dt(t: np.ndarray) -> float | None:
    t = np.asarray(t, dtype=float)
    if t.size < 2:
        return None
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return float(np.median(dt)) if dt.size else None


def matlab_datenum(value: float) -> datetime | None:
    """Convert MATLAB datenum to a Python datetime."""
    try:
        return datetime(1970, 1, 1) + timedelta(days=float(value) - 719529.0)
    except (TypeError, ValueError, OverflowError):
        return None


def save_fig(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGS / name)
    plt.close(fig)
    print(f"figure: {FIGS / name}")


# ---------------------------------------------------------------------------
# NASA PCoE
# ---------------------------------------------------------------------------

NASA_CELLS = ["B0005", "B0006", "B0007", "B0018"]


@dataclass
class NasaCycle:
    cell_id: str
    cycle_index: int
    operation: str
    ambient_temperature_C: float | None
    start_time: str | None
    duration_s: float | None
    samples: int
    voltage_min_V: float | None
    voltage_mean_V: float | None
    voltage_max_V: float | None
    current_min_A: float | None
    current_mean_A: float | None
    current_max_A: float | None
    temperature_min_C: float | None
    temperature_mean_C: float | None
    temperature_max_C: float | None
    reported_capacity_Ah: float | None
    coulomb_count_Ah: float | None
    n_nan: int
    out_of_range: int
    file: str


def _nasa_operation_data(d) -> dict[str, np.ndarray]:
    out = {}
    for field in ["Voltage_measured", "Current_measured", "Temperature_measured", "Time", "Capacity"]:
        if hasattr(d, field):
            out[field] = np.asarray(getattr(d, field), dtype=float).ravel()
    return out


def load_nasa() -> tuple[pd.DataFrame, pd.DataFrame]:
    cycles = []
    eis = []
    files = sorted(RAW.glob("NASA/extracted/*/B*.mat"))
    # Keep primary four cells; record duplicates separately in inventory.
    files = [p for p in files if p.name in {f"{c}.mat" for c in NASA_CELLS}]
    for path in files:
        cell_id = path.stem
        m = loadmat(path, squeeze_me=True, struct_as_record=False)
        struct = m[cell_id]
        discharge_count = 0
        for idx, c in enumerate(np.atleast_1d(struct.cycle)):
            op = str(c.type)
            d = c.data
            if op == "discharge":
                discharge_count += 1
            start = matlab_datenum(c.time) if np.isscalar(c.time) else None
            if op == "impedance":
                fields = {f: getattr(d, f, None) for f in ["Re", "Rct", "Battery_impedance", "Rectified_Impedance"]}
                z_raw = np.asarray(fields["Battery_impedance"]) if fields["Battery_impedance"] is not None else np.array([])
                z = np.asarray(fields["Rectified_Impedance"]) if fields["Rectified_Impedance"] is not None else z_raw
                eis.append(
                    {
                        "cell_id": cell_id,
                        "cycle_index": idx,
                        "discharge_number": discharge_count,
                        "start_time": start.isoformat() if start else None,
                        "Re_Ohm": float(fields["Re"]) if fields["Re"] is not None else None,
                        "Rct_Ohm": float(fields["Rct"]) if fields["Rct"] is not None else None,
                        "n_frequencies": int(z.size),
                        "z_real_min_Ohm": float(np.nanmin(z.real)) if z.size else None,
                        "z_imag_min_Ohm": float(np.nanmin(z.imag)) if z.size else None,
                        "z_real_last_Ohm": float(z.real[-1]) if z.size else None,
                        "z_imag_last_Ohm": float(z.imag[-1]) if z.size else None,
                        "raw_n_frequencies": int(z_raw.size),
                    }
                )
                continue
            if op not in {"charge", "discharge"}:
                continue
            data = _nasa_operation_data(d)
            t = data.get("Time", np.array([]))
            v = data.get("Voltage_measured", np.array([]))
            i = data.get("Current_measured", np.array([]))
            temp = data.get("Temperature_measured", np.array([]))
            length = min(len(t), len(v), len(i)) if min(len(t), len(v), len(i)) else 0
            n_nan = int(sum(np.isnan(x).sum() for x in (v, i, temp)))
            out_of_range = int(
                ((v < 1.5) | (v > 4.6)).sum()
                + ((temp < -40) | (temp > 80)).sum()
                + ((np.abs(i) > 15)).sum()
            )
            coulomb = None
            if length >= 2:
                dt = np.diff(t[:length])
                dt = np.where(np.isfinite(dt) & (dt > 0), dt, 0)
                coulomb = float(np.sum(np.abs(i[:length - 1]) * dt) / 3600.0)
            reported_capacity = None
            if op == "discharge" and data.get("Capacity") is not None and data["Capacity"].size:
                reported_capacity = float(finite(data["Capacity"])[0]) if finite(data["Capacity"]).size else None
            cycles.append(
                asdict(
                    NasaCycle(
                        cell_id=cell_id,
                        cycle_index=idx,
                        operation=op,
                        ambient_temperature_C=float(c.ambient_temperature) if np.isscalar(c.ambient_temperature) else None,
                        start_time=start.isoformat() if start else None,
                        duration_s=float(t[-1] - t[0]) if len(t) > 1 else None,
                        samples=int(length),
                        voltage_min_V=float(np.nanmin(v)) if len(v) else None,
                        voltage_mean_V=float(np.nanmean(v)) if len(v) else None,
                        voltage_max_V=float(np.nanmax(v)) if len(v) else None,
                        current_min_A=float(np.nanmin(i)) if len(i) else None,
                        current_mean_A=float(np.nanmean(i)) if len(i) else None,
                        current_max_A=float(np.nanmax(i)) if len(i) else None,
                        temperature_min_C=float(np.nanmin(temp)) if len(temp) else None,
                        temperature_mean_C=float(np.nanmean(temp)) if len(temp) else None,
                        temperature_max_C=float(np.nanmax(temp)) if len(temp) else None,
                        reported_capacity_Ah=reported_capacity,
                        coulomb_count_Ah=coulomb,
                        n_nan=n_nan,
                        out_of_range=out_of_range,
                        file=str(path.relative_to(PROJECT_ROOT)),
                    )
                )
            )
    return pd.DataFrame(cycles), pd.DataFrame(eis)


def nasa_analysis() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cycles, eis = load_nasa()
    cycles.to_csv(AUDIT / "nasa_cycle_summary.csv", index=False)
    eis.to_csv(AUDIT / "nasa_eis_summary.csv", index=False)

    # Discharge capacity trajectory and SOH (against first measured discharge).
    dis = cycles[cycles.operation == "discharge"].copy()
    dis["discharge_number"] = dis.groupby("cell_id").cumcount() + 1
    dis["soh"] = np.nan
    for cell, grp in dis.groupby("cell_id"):
        valid = grp.dropna(subset=["reported_capacity_Ah"])
        if valid.empty:
            continue
        initial = float(valid.iloc[0]["reported_capacity_Ah"])
        dis.loc[grp.index, "soh"] = grp["reported_capacity_Ah"] / initial
    dis.to_csv(AUDIT / "nasa_discharge_trajectory.csv", index=False)

    # Figure 1: first / last discharge curves, truncated at the cell's cutoff.
    cutoffs = {"B0005": 2.7, "B0006": 2.5, "B0007": 2.2, "B0018": 2.5}
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharey=True)
    for ax, cell in zip(axes.ravel(), NASA_CELLS):
        grp = dis[dis.cell_id == cell].dropna(subset=["reported_capacity_Ah"])
        if grp.empty:
            ax.set_axis_off()
            continue
        for label, row in [("first", grp.iloc[0]), ("last", grp.iloc[-1])]:
            m = loadmat(RAW / "NASA" / "extracted" / "1_ BatteryAgingARC-FY08Q4" / f"{cell}.mat",
                        squeeze_me=True, struct_as_record=False)
            c = np.atleast_1d(m[cell].cycle)[int(row.cycle_index)]
            d = c.data
            t = np.asarray(d.Time, float)
            i = np.asarray(d.Current_measured, float)
            v = np.asarray(d.Voltage_measured, float)
            q = np.concatenate([[0], np.cumsum(np.abs(i[:-1]) * np.maximum(np.diff(t), 0)) / 3600])
            stop = np.where(v <= cutoffs[cell])[0]
            end = int(stop[0]) + 1 if stop.size else len(q)
            q, v = q[:end], v[:end]
            ax.plot(q, v, color=COLORS[0 if label == "first" else 1],
                    label=f"{label} (Q={row.reported_capacity_Ah:.3f} Ah)")
        ax.set_title(cell)
        ax.set_xlabel("Discharged capacity / Ah")
        ax.set_ylabel("Terminal voltage / V")
        ax.legend(loc="best")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.suptitle("NASA PCoE: first and last measured discharge curves", fontweight="bold")
    save_fig(fig, "fig01_nasa_discharge_curves.png")

    # Figure 2: capacity fade and SOH.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for color, (cell, grp) in zip(COLORS, dis.groupby("cell_id")):
        axes[0].plot(grp.discharge_number, grp.reported_capacity_Ah, color=color, label=cell, lw=1.5)
        axes[1].plot(grp.discharge_number, 100 * grp.soh, color=color, label=cell, lw=1.5)
    axes[0].set(xlabel="Discharge cycle number", ylabel="Discharge capacity / Ah")
    axes[1].axhline(80, color="#4d4d4d", ls="--", lw=1, label="80% EOL")
    axes[1].set(xlabel="Discharge cycle number", ylabel="SOH / %", ylim=(55, 105))
    for ax in axes:
        ax.legend(frameon=False)
    fig.suptitle("NASA PCoE: measured capacity and SOH trajectories", fontweight="bold")
    save_fig(fig, "fig02_nasa_capacity_fade.png")

    # Figure 3: EIS ageing for B0005.
    fig = plt.figure(figsize=(10, 3.9))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    eis_b5 = eis[eis.cell_id == "B0005"].sort_values("discharge_number")
    if not eis_b5.empty:
        picks = sorted(set([0, len(eis_b5) // 2, len(eis_b5) - 1]))
        # Load full impedance spectra for the selected records.
        m = loadmat(RAW / "NASA" / "extracted" / "1_ BatteryAgingARC-FY08Q4" / "B0005.mat",
                    squeeze_me=True, struct_as_record=False)
        for k, pos in zip(COLORS, picks):
            row = eis_b5.iloc[pos]
            c = np.atleast_1d(m["B0005"].cycle)[int(row.cycle_index)]
            z = np.asarray(c.data.Rectified_Impedance)
            ax1.plot(z.real, -z.imag, "o-", ms=2.5, lw=1, color=k,
                     label=f"cycle {int(row.discharge_number)} | Re={row.Re_Ohm:.3f}")
        ax1.set(xlabel="Z' / Ohm", ylabel="-Z'' / Ohm", title="B0005 impedance spectra")
        ax1.legend(frameon=False)
    for color, (cell, grp) in zip(COLORS, eis.groupby("cell_id")):
        ax2.plot(grp.discharge_number, grp.Re_Ohm, color=color, lw=1.2, ls="--")
        ax2.plot(grp.discharge_number, grp.Rct_Ohm, color=color, lw=1.2, label=cell)
    ax2.set(xlabel="Discharge cycle number", ylabel="Resistance / Ohm",
            title="Re (dashed) and Rct (solid)")
    ax2.legend(frameon=False, title="Cell")
    fig.suptitle("NASA PCoE: impedance evolution with ageing", fontweight="bold")
    save_fig(fig, "fig03_nasa_eis_ageing.png")

    # Figure 4: representative time-series channels, B0005 first discharge + next charge.
    fig, axes = plt.subplots(3, 1, figsize=(9, 6.2), sharex=True)
    m = loadmat(RAW / "NASA" / "extracted" / "1_ BatteryAgingARC-FY08Q4" / "B0005.mat",
                squeeze_me=True, struct_as_record=False)
    entries = np.atleast_1d(m["B0005"].cycle)
    charge = next(c for c in entries if str(c.type) == "charge")
    discharge = next(c for c in entries if str(c.type) == "discharge")
    for label, c, color in [("charge", charge, COLORS[2]), ("discharge", discharge, COLORS[1])]:
        t = np.asarray(c.data.Time, float)
        t = t - t[0]
        axes[0].plot(t / 60, c.data.Voltage_measured, color=color, label=label)
        axes[1].plot(t / 60, c.data.Current_measured, color=color, label=label)
        axes[2].plot(t / 60, c.data.Temperature_measured, color=color, label=label)
    axes[0].set_ylabel("Voltage / V")
    axes[1].set_ylabel("Current / A")
    axes[2].set_ylabel("Temperature / C")
    axes[2].set_xlabel("Elapsed time / min")
    for ax in axes:
        ax.legend(frameon=False)
    fig.suptitle("NASA B0005: early charge/discharge measurements", fontweight="bold")
    save_fig(fig, "fig04_nasa_time_series.png")

    # Summary statistics for report.
    archive_files = sorted(RAW.glob("NASA/extracted/*/B*.mat"))
    unique_cells = sorted({p.stem for p in archive_files})
    sampling = {}
    for c in NASA_CELLS:
        m = loadmat(RAW / "NASA" / "extracted" / "1_ BatteryAgingARC-FY08Q4" / f"{c}.mat",
                    squeeze_me=True, struct_as_record=False)
        entry = next(x for x in np.atleast_1d(m[c].cycle) if str(x.type) == "discharge")
        sampling[c] = med_dt(np.asarray(entry.data.Time, float))
    summary = {
        "cells_analyzed": NASA_CELLS,
        "mat_files_in_archive": len(archive_files),
        "unique_cells_in_archive": len(unique_cells),
        "duplicate_cell_files": sorted({p.name for p in archive_files if sum(q.name == p.name for q in archive_files) > 1}),
        "operations": dict(Counter(cycles.operation)),
        "total_measurement_rows": int(cycles.samples.sum()),
        "cycle_counts": {c: int((dis.cell_id == c).sum()) for c in NASA_CELLS},
        "median_sampling_s": sampling,
        "voltage_range_V": robust_stats(cycles.voltage_min_V.tolist() + cycles.voltage_max_V.tolist()),
        "temperature_range_C": robust_stats(cycles.temperature_min_C.tolist() + cycles.temperature_max_C.tolist()),
        "nan_values": int(cycles.n_nan.sum()),
        "out_of_range_values": int(cycles.out_of_range.sum()),
        "capacity_initial_Ah": {c: float(dis[dis.cell_id == c].iloc[0].reported_capacity_Ah) for c in NASA_CELLS},
        "capacity_final_Ah": {c: float(dis[dis.cell_id == c].iloc[-1].reported_capacity_Ah) for c in NASA_CELLS},
        "soh_final_pct": {c: float(100 * dis[dis.cell_id == c].iloc[-1].soh) for c in NASA_CELLS},
        "impedance_records": int(len(eis)),
        "impedance_cells": sorted(eis.cell_id.unique().tolist()),
    }
    (AUDIT / "nasa_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return cycles, eis, summary


# ---------------------------------------------------------------------------
# Oxford Battery Degradation Dataset 1
# ---------------------------------------------------------------------------

OXFORD_CELLS = [f"Cell{i}" for i in range(1, 9)]


def _oxford_block(block) -> dict:
    t = np.asarray(block.t, float).ravel()
    v = np.asarray(block.v, float).ravel()
    q = np.asarray(block.q, float).ravel()
    temp = np.asarray(block.T, float).ravel()
    return {
        "t": t, "v": v, "q": q, "T": temp,
        "capacity_mAh": float(np.nanmax(q) - np.nanmin(q)),
        "duration_s": float((np.nanmax(t) - np.nanmin(t)) * 86400) if t.size else None,
        "dt_s": med_dt(t) * 86400 if t.size > 1 else None,
        "v_min": float(np.nanmin(v)), "v_max": float(np.nanmax(v)),
        "T_min": float(np.nanmin(temp)), "T_max": float(np.nanmax(temp)),
        "n_nan": int(np.isnan(v).sum() + np.isnan(q).sum() + np.isnan(temp).sum()),
        # The provider states 40 C chamber tests, so sub-20 C readings are
        # sensor glitches rather than physical cell temperatures.
        "T_below_20": int(np.nansum(temp < 20)),
        "T_below_5": int(np.nansum(temp < 5)),
        "T_n": int(np.isfinite(temp).sum()),
    }


def load_oxford() -> pd.DataFrame:
    path = RAW / "Oxford" / "Oxford_Battery_Degradation_Dataset_1.mat"
    rows = []
    # Load one cell at a time to limit peak memory.
    for cell in OXFORD_CELLS:
        m = loadmat(path, squeeze_me=True, struct_as_record=False, variable_names=[cell])
        c = m[cell]
        cycle_fields = sorted([f for f in c._fieldnames if re.fullmatch(r"cyc\d{4}", f)])
        for field in cycle_fields:
            cycnum = int(field[3:])
            block = getattr(c, field)
            for kind, name in [("C1ch", "1C charge"), ("C1dc", "1C discharge"),
                               ("OCVch", "pseudo-OCV charge"), ("OCVdc", "pseudo-OCV discharge")]:
                if not hasattr(block, kind):
                    continue
                stats = _oxford_block(getattr(block, kind))
                rows.append(
                    {
                        "cell_id": cell,
                        "characterisation_cycle": cycnum,
                        "segment": name,
                        "capacity_mAh": stats["capacity_mAh"],
                        "duration_s": stats["duration_s"],
                        "median_dt_s": stats["dt_s"],
                        "voltage_min_V": stats["v_min"],
                        "voltage_max_V": stats["v_max"],
                        "temperature_min_C": stats["T_min"],
                        "temperature_max_C": stats["T_max"],
                        "temperature_below_20_count": stats["T_below_20"],
                        "temperature_below_5_count": stats["T_below_5"],
                        "temperature_sample_count": stats["T_n"],
                        "n_nan": stats["n_nan"],
                    }
                )
        del m, c
    return pd.DataFrame(rows)


def oxford_analysis() -> tuple[pd.DataFrame, dict]:
    df = load_oxford()
    df.to_csv(AUDIT / "oxford_characterisation_summary.csv", index=False)

    # One row per cell/check-up: use pseudo-OCV discharge capacity as the
    # least rate-biased measured capacity; retain 1C capacity for comparison.
    cap = df[df.segment.isin(["1C discharge", "pseudo-OCV discharge"])].pivot_table(
        index=["cell_id", "characterisation_cycle"], columns="segment", values="capacity_mAh"
    ).reset_index()
    cap.columns.name = None
    cap = cap.rename(columns={"1C discharge": "capacity_1C_mAh", "pseudo-OCV discharge": "capacity_pOCV_mAh"})
    cap["soh_1C"] = np.nan
    cap["soh_pOCV"] = np.nan
    for cell, grp in cap.groupby("cell_id"):
        base1 = grp.sort_values("characterisation_cycle").iloc[0]["capacity_1C_mAh"]
        basep = grp.sort_values("characterisation_cycle").iloc[0]["capacity_pOCV_mAh"]
        if pd.notna(base1) and base1:
            cap.loc[grp.index, "soh_1C"] = grp["capacity_1C_mAh"] / base1
        if pd.notna(basep) and basep:
            cap.loc[grp.index, "soh_pOCV"] = grp["capacity_pOCV_mAh"] / basep
    cap.to_csv(AUDIT / "oxford_capacity_trajectory.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    for color, (cell, grp) in zip(COLORS, cap.groupby("cell_id")):
        axes[0].plot(grp.characterisation_cycle, grp.capacity_pOCV_mAh, "o-", ms=3, lw=1.3,
                     color=color, label=cell)
        axes[1].plot(grp.characterisation_cycle, 100 * grp.soh_pOCV, "o-", ms=3, lw=1.3,
                     color=color, label=cell)
    axes[0].set(xlabel="Drive cycle number (check-ups every 100 cycles)",
                ylabel="Pseudo-OCV discharge capacity / mAh")
    axes[1].axhline(80, color="#4d4d4d", ls="--", lw=1, label="80% EOL")
    axes[1].set(xlabel="Drive cycle number (check-ups every 100 cycles)",
                ylabel="SOH / %", ylim=(65, 102))
    axes[0].set_ylim(480, 780)
    for ax in axes:
        ax.legend(frameon=False, ncol=2)
    fig.suptitle("Oxford Battery Degradation Dataset 1: capacity and SOH", fontweight="bold")
    save_fig(fig, "fig05_oxford_soh.png")

    # Characterisation curves for Cell 1 at first, middle and last check-ups.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    m = loadmat(RAW / "Oxford" / "Oxford_Battery_Degradation_Dataset_1.mat",
                squeeze_me=True, struct_as_record=False, variable_names=["Cell1"])
    c = m["Cell1"]
    fields = sorted([f for f in c._fieldnames if re.fullmatch(r"cyc\d{4}", f)])
    picks = sorted(set([0, len(fields) // 2, len(fields) - 1]))
    for color, pos in zip(COLORS, picks):
        block = getattr(c, fields[pos])
        for kind, ax, title in [("C1dc", axes[0], "1C discharge"), ("OCVdc", axes[1], "Pseudo-OCV discharge (40 mA)")]:
            b = getattr(block, kind)
            q = np.asarray(b.q, float) - np.min(np.asarray(b.q, float))
            order = np.argsort(q)
            ax.plot(q[order], np.asarray(b.v, float)[order], color=color, lw=1.4,
                    label=f"check-up {fields[pos]}")
    for ax, title in zip(axes, ["1C discharge", "Pseudo-OCV discharge (40 mA)"]):
        ax.set(xlabel="Discharged capacity / mAh", ylabel="Voltage / V", title=title)
        ax.legend(frameon=False)
    fig.suptitle("Oxford Cell 1: voltage-capacity curves at three ages", fontweight="bold")
    save_fig(fig, "fig06_oxford_curves.png")

    # Dynamic current profile.
    m2 = loadmat(RAW / "Oxford" / "ExampleDC_C1.mat", squeeze_me=True, struct_as_record=False)
    d = m2["ExampleDC_C1"].dc
    t = np.asarray(d.t, float)
    t = (t - t[0]) * 86400
    i = np.asarray(d.i, float)
    v = np.asarray(d.v, float)
    temp = np.asarray(d.T, float)
    fig, axes = plt.subplots(3, 1, figsize=(9, 5.6), sharex=True)
    axes[0].plot(t / 60, i, color=COLORS[0], lw=0.9)
    axes[0].set_ylabel("Current / mA")
    axes[1].plot(t / 60, v, color=COLORS[1], lw=0.9)
    axes[1].set_ylabel("Voltage / V")
    axes[2].plot(t / 60, temp, color=COLORS[2], lw=0.9)
    axes[2].set_ylabel("Temperature / C")
    axes[2].set_xlabel("Elapsed time / min")
    fig.suptitle("Oxford: first Urban Artemis drive-cycle discharge (Cell 1)", fontweight="bold")
    save_fig(fig, "fig07_oxford_drive_cycle.png")

    temp_samples = int(df.temperature_sample_count.sum())
    temp_below_20 = int(df.temperature_below_20_count.sum())
    temp_below_5 = int(df.temperature_below_5_count.sum())
    summary = {
        "cells": int(cap.cell_id.nunique()),
        "checkups_per_cell": cap.groupby("cell_id").size().to_dict(),
        "max_drive_cycles": cap.groupby("cell_id").characterisation_cycle.max().to_dict(),
        "capacity_initial_mAh": cap.groupby("cell_id").capacity_pOCV_mAh.first().round(3).to_dict(),
        "capacity_final_mAh": cap.groupby("cell_id").capacity_pOCV_mAh.last().round(3).to_dict(),
        "soh_final_pct": (100 * cap.groupby("cell_id").soh_pOCV.last()).round(2).to_dict(),
        "median_checkup_sampling_s": float(df[df.segment == "1C discharge"].median_dt_s.median()),
        "voltage_range_V": [float(df.voltage_min_V.min()), float(df.voltage_max_V.max())],
        "temperature_range_C": [float(df.temperature_min_C.min()), float(df.temperature_max_C.max())],
        "temperature_samples": temp_samples,
        "temperature_below_20_C": temp_below_20,
        "temperature_below_5_C": temp_below_5,
        "temperature_below_20_fraction": round(temp_below_20 / temp_samples, 6) if temp_samples else None,
        "nan_values": int(df.n_nan.sum()),
        "native_soc_labels": False,
        "soh_label": "pseudo-OCV discharge capacity / first check-up capacity",
        "rul_label": "derived only; coarse 100-cycle resolution",
        "quality_flags": [
            "nominally 40 C tests contain isolated sub-20 C thermocouple readings",
            "characterisation only every 100 drive cycles; RUL resolution is coarse",
        ],
    }
    (AUDIT / "oxford_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return cap, summary


# ---------------------------------------------------------------------------
# Oxford energy-trading (stationary storage)
# ---------------------------------------------------------------------------


def energy_analysis() -> tuple[pd.DataFrame, dict]:
    folder = RAW / "additional" / "Oxford_Energy_Trading"
    caps = []
    for path in sorted(folder.glob("*_capacityData.csv")):
        name = path.stem.replace("_capacityData", "")
        df = pd.read_csv(path)
        df["cell_id"] = name
        caps.append(df)
    cap = pd.concat(caps, ignore_index=True)
    cap["soh_vs_initial"] = np.nan
    for cell, grp in cap.groupby("cell_id"):
        base = grp.sort_values("time_s").iloc[0]["capacity_Ah"]
        cap.loc[grp.index, "soh_vs_initial"] = grp["capacity_Ah"] / base
    cap.to_csv(AUDIT / "oxford_energy_capacity.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    for color, (cell, grp) in zip(COLORS, cap.groupby("cell_id")):
        ax.plot(grp.time_s / 86400, 100 * grp.soh_vs_initial, "o-", color=color, label=cell, ms=3)
    ax.set(xlabel="Experiment time / days", ylabel="Capacity retention / %")
    ax.legend(frameon=False, ncol=2)
    ax.set_title("Oxford energy-trading dataset: monthly capacity retention", fontweight="bold")
    save_fig(fig, "fig08_energy_storage_capacity.png")

    # One representative profile with all channels (downsampled for plotting).
    p = pd.read_csv(folder / "BMR_cell1_profileData.csv")
    step = max(1, len(p) // 12000)
    p = p.iloc[::step]
    fig, axes = plt.subplots(3, 1, figsize=(9, 5.6), sharex=True)
    axes[0].plot(p.time_s / 86400, p.current_A, color=COLORS[0], lw=0.7)
    axes[0].set_ylabel("Current / A")
    axes[1].plot(p.time_s / 86400, p.voltage_V, color=COLORS[1], lw=0.7)
    axes[1].set_ylabel("Voltage / V")
    axes[2].plot(p.time_s / 86400, p.cell_temperature_C, color=COLORS[2], lw=0.7, label="cell")
    axes[2].plot(p.time_s / 86400, p.environment_temperature_C, color=COLORS[3], lw=0.7, label="environment")
    axes[2].set_ylabel("Temperature / C")
    axes[2].set_xlabel("Experiment time / days")
    axes[2].legend(frameon=False)
    fig.suptitle("Oxford energy-trading dataset: BMR cell 1 field-like profile", fontweight="bold")
    save_fig(fig, "fig09_energy_storage_profile.png")

    profiles = {}
    for cell in sorted(cap.cell_id.unique()):
        prof = pd.read_csv(folder / f"{cell}_profileData.csv")
        voltage = prof.voltage_V[prof.voltage_V > 0]
        profiles[cell] = {
            "samples": int(len(prof)),
            "duration_days": float((prof.time_s.max() - prof.time_s.min()) / 86400),
            "median_sampling_s": float(np.median(np.diff(finite(prof.time_s)))),
            "current_range_A": [float(prof.current_A.min()), float(prof.current_A.max())],
            "voltage_range_V": [float(voltage.min()), float(voltage.max())],
            "cell_temperature_range_C": [float(prof.cell_temperature_C[prof.cell_temperature_C != 0].min()),
                                         float(prof.cell_temperature_C.max())],
            "zero_voltage_samples": int((prof.voltage_V == 0).sum()),
            "zero_cell_temperature_samples": int((prof.cell_temperature_C == 0).sum()),
            "zero_environment_temperature_samples": int((prof.environment_temperature_C == 0).sum()),
        }
    summary = {
        "cells": int(cap.cell_id.nunique()),
        "rated_capacity_Ah": 16.0,
        "measured_capacity_initial_Ah": cap.groupby("cell_id").capacity_Ah.first().round(3).to_dict(),
        "capacity_retention_final_pct": (100 * cap.groupby("cell_id").soh_vs_initial.last()).round(2).to_dict(),
        "capacity_checkups": cap.groupby("cell_id").size().to_dict(),
        "profiles": profiles,
        "native_soc_labels": False,
        "soh_label": "measured 1C capacity / first measured capacity",
        "rul_label": "not available (no EOL reached)",
    }
    (AUDIT / "oxford_energy_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return cap, summary


# ---------------------------------------------------------------------------
# Samsung INR18650-35E (additional NMC)
# ---------------------------------------------------------------------------


SAMSUNG_FILES = {
    "Battery1": range(25, 32),  # 0025-0031
    "Battery2": range(32, 39),  # 0032-0038
}


def _read_samsung_cycles(folder: Path) -> pd.DataFrame:
    """Read the 14 raw cycling blocks and build a continuous cycle index.

    Each block restarts its local cycle counter at zero, so offsets are
    accumulated from the actual number of local cycles (blocks contain
    100, 30 or 70/71 cycles, not a fixed length).
    """
    frames = []
    for battery, numbers in SAMSUNG_FILES.items():
        offset = 0
        for number in numbers:
            path = folder / f"VZ5_008_021_VUT_D_{number:04d}_v1.txt"
            if not path.exists():
                continue
            df = pd.read_csv(path, sep="\t", encoding="latin-1", usecols=range(7))
            df.columns = ["cycle", "time_s", "Qdis_mAh", "Qch_mAh",
                          "voltage_V", "current_mA", "temperature_C"]
            df = df.dropna(subset=["cycle", "time_s"])
            df["cycle"] = df["cycle"].astype(int)
            local_max = int(df["cycle"].max())
            df["battery"] = battery
            df["global_cycle"] = offset + df["cycle"] + 1
            offset += local_max + 1
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def samsung_analysis() -> tuple[pd.DataFrame, dict]:
    folder = RAW / "additional" / "Samsung_INR18650_35E"
    cyc = _read_samsung_cycles(folder)
    summary = {}
    if cyc.empty:
        summary = {"status": "cycling files not yet downloaded"}
        (AUDIT / "samsung_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return cyc, summary
    for c in ["time_s", "Qdis_mAh", "Qch_mAh", "voltage_V", "current_mA", "temperature_C"]:
        cyc[c] = pd.to_numeric(cyc[c], errors="coerce")
    cyc = cyc.dropna(subset=["global_cycle", "Qdis_mAh"])
    per_cycle = cyc.groupby(["battery", "global_cycle"]).agg(
        Qdis_mAh=("Qdis_mAh", "max"),
        Qch_mAh=("Qch_mAh", "max"),
        Vmin=("voltage_V", "min"),
        Vmax=("voltage_V", "max"),
        Tmean=("temperature_C", "mean"),
        Tmax=("temperature_C", "max"),
        samples=("time_s", "size"),
        duration_s=("time_s", lambda x: float(np.nanmax(x) - np.nanmin(x)) if len(x) else np.nan),
    ).reset_index()

    # Partial/incomplete cycles are flagged but keep their global slot so the
    # provider's 1..580 numbering can be matched unambiguously.
    per_cycle["valid_discharge"] = per_cycle.Qdis_mAh > 0.5 * 3400
    per_cycle["soh_rated"] = per_cycle.Qdis_mAh / 3400.0
    per_cycle["soh_first"] = np.nan
    for battery, grp in per_cycle.groupby("battery"):
        valid = grp[grp.valid_discharge].sort_values("global_cycle")
        if not valid.empty:
            per_cycle.loc[valid.index, "soh_first"] = valid.Qdis_mAh / valid.Qdis_mAh.iloc[0]
    per_cycle.to_csv(AUDIT / "samsung_cycle_summary.csv", index=False)

    # File 0043 is the authors' measured SOH (Qdischarge / rated 3.4 Ah).
    soh_path = folder / "VZ5_008_021_VUT_D_0043_v1.txt"
    soh = pd.DataFrame()
    if soh_path.exists():
        soh = pd.read_csv(soh_path, sep="\t", encoding="utf-8")
        soh.columns = ["cycle", "Battery1", "Battery2"]
        soh.to_csv(AUDIT / "samsung_reported_soh.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    if not soh.empty:
        for color, battery in zip(COLORS, ["Battery1", "Battery2"]):
            axes[0].plot(soh.cycle, 100 * soh[battery], color=color, lw=1.6,
                         label=f"{battery}: provider")
            measured = per_cycle[(per_cycle.battery == battery) & per_cycle.valid_discharge]
            axes[0].plot(measured.global_cycle, 100 * measured.soh_rated, ".",
                         color=color, ms=1.6, alpha=0.55, label=f"{battery}: measured")
            merged = measured.merge(
                soh[["cycle", battery]].rename(columns={battery: "provider", "cycle": "global_cycle"}),
                on="global_cycle", how="inner",
            )
            if not merged.empty:
                axes[1].scatter(100 * merged.provider, 100 * merged.soh_rated,
                                s=4, color=color, alpha=0.5, label=battery)
        axes[1].plot([70, 100], [70, 100], color="#4d4d4d", lw=1, ls="--")
        axes[1].set(xlabel="Provider SOH / %", ylabel="Measured Qdis/3.4 Ah / %")
        axes[1].legend(frameon=False)
    axes[0].axhline(80, color="#4d4d4d", ls="--", lw=1, label="80% EOL")
    axes[0].set(xlabel="Cycle number", ylabel="SOH / %", ylim=(70, 102))
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("Samsung INR18650-35E: provider SOH and remeasured capacity", fontweight="bold")
    save_fig(fig, "fig10_samsung_soh.png")

    valid_cycles = per_cycle[per_cycle.valid_discharge]
    parity = []
    if not soh.empty:
        for battery in ["Battery1", "Battery2"]:
            merged = valid_cycles[valid_cycles.battery == battery].merge(
                soh[["cycle", battery]].rename(columns={battery: "provider", "cycle": "global_cycle"}),
                on="global_cycle", how="inner",
            )
            if not merged.empty:
                parity.append((100 * merged.provider - 100 * merged.soh_rated).abs())
    parity_errors = pd.concat(parity) if parity else pd.Series(dtype=float)

    first_dt = cyc[(cyc.battery == "Battery1") & (cyc.global_cycle == 1)].time_s
    summary = {
        "cells": 2,
        "chemistry": "NMC (Samsung INR18650-35E; provider abbreviation list states NMC)",
        "rated_capacity_Ah": 3.4,
        "raw_files": int(sum(1 for b in SAMSUNG_FILES.values() for n in b
                             if (folder / f"VZ5_008_021_VUT_D_{n:04d}_v1.txt").exists())),
        "cycles_measured": valid_cycles.groupby("battery").size().to_dict(),
        "max_global_cycle": valid_cycles.groupby("battery").global_cycle.max().to_dict(),
        "cycles_provider_soh": int(soh.cycle.max()) if not soh.empty else None,
        "measurement_rows": int(len(cyc)),
        "median_sampling_s": float(med_dt(first_dt.to_numpy())),
        "voltage_range_V": [float(cyc.voltage_V.min()), float(cyc.voltage_V.max())],
        "temperature_range_C": [float(cyc.temperature_C.min()), float(cyc.temperature_C.max())],
        "current_range_mA": [float(cyc.current_mA.min()), float(cyc.current_mA.max())],
        "soh_provider_initial": {c: float(soh[c].iloc[0]) for c in ["Battery1", "Battery2"]} if not soh.empty else {},
        "soh_provider_final": {c: float(soh[c].iloc[-1]) for c in ["Battery1", "Battery2"]} if not soh.empty else {},
        "soh_measured_rated_final_pct": (100 * valid_cycles.groupby("battery").soh_rated.last()).round(2).to_dict(),
        "soh_measured_first_final_pct": (100 * valid_cycles.groupby("battery").soh_first.last()).round(2).to_dict(),
        "provider_vs_measured_mae_pct": round(float(parity_errors.mean()), 3) if len(parity_errors) else None,
        "provider_vs_measured_max_abs_pct": round(float(parity_errors.max()), 3) if len(parity_errors) else None,
        "native_soh_labels": True,
        "native_soc_labels": False,
        "soh_label": "provider SOH = Qdischarge / 3.4 Ah; harmonised SOH = Qdischarge / first measured Qdischarge",
        "rul_label": "not reached: provider SOH ends at 83.1%/82.1% after 580 cycles",
        "quality_flags": [
            "raw blocks restart Cycle number at zero and have unequal lengths (100/30/70 cycles)",
            "Battery2 block 0033 ends with a 366.5 mAh partial discharge that is excluded from SOH",
        ],
    }
    (AUDIT / "samsung_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return per_cycle, summary


# ---------------------------------------------------------------------------
# CALCE CS2 prismatic cells (LCO)
# ---------------------------------------------------------------------------

CALCE_USECOLS = [
    "Test_Time(s)",
    "Date_Time",
    "Step_Index",
    "Cycle_Index",
    "Current(A)",
    "Voltage(V)",
    "Charge_Capacity(Ah)",
    "Discharge_Capacity(Ah)",
]

# The CS2 cells were not all tested with the same protocol.  These rules keep
# each cell's SOH denominator tied to a physically comparable measurement.
CALCE_RULES = {
    "CS2_35": {
        "mode": "full",
        "basis": "discharge",
        "min_capacity_Ah": 0.15,
        "max_voltage_min_V": 2.8,
    },
    "CS2_36": {
        "mode": "full",
        "basis": "discharge",
        "min_capacity_Ah": 0.15,
        "max_voltage_min_V": 2.8,
    },
    "CS2_24": {
        "mode": "checkup",
        "basis": "discharge",
        "min_capacity_Ah": 0.45,
        "max_voltage_min_V": 3.2,
    },
    "CS2_3": {
        "mode": "block",
        "basis": "low-rate discharge",
        "min_capacity_Ah": 0.05,
        "min_discharge_steps": 5,
    },
}

CALCE_SOH_BASIS = {
    "CS2_35": "full-cycle discharge capacity (Vmin <= 2.8 V); partial cycles excluded",
    "CS2_36": "full-cycle discharge capacity (Vmin <= 2.8 V); partial cycles excluded",
    "CS2_24": "full check-up discharge capacity (Vmin <= 3.2 V, Qd >= 0.45 Ah); partial 3.78-4.2 V cycles excluded",
    "CS2_3": "0.11 A (C/10) discharge capacity of each six-rate block; block charge throughput recorded separately",
}


def _calce_info_start(path: Path) -> datetime | None:
    """Return the test start datetime recorded on the Info sheet."""
    try:
        info = pd.read_excel(path, sheet_name="Info", header=None, nrows=10)
    except Exception:
        return None
    for value in info.to_numpy().ravel():
        if isinstance(value, datetime):
            return pd.Timestamp(value).to_pydatetime()
    return None


def _calce_name_date(name: str) -> datetime | None:
    """Parse the date suffix used by CALCE exports, e.g. CS2_35_2_4_11.xlsx."""
    match = re.search(r"_(\d{1,2})_(\d{1,2})_(\d{2})\.xlsx$", name)
    if not match:
        return None
    month, day, year = (int(part) for part in match.groups())
    try:
        return datetime(2000 + year, month, day)
    except ValueError:
        return None


def _calce_channel_sheets(path: Path) -> list[str]:
    try:
        names = pd.ExcelFile(path).sheet_names
    except Exception:
        return []
    return [str(name) for name in names if str(name).startswith("Channel")]


def _calce_read_sheet(path: Path, sheet: str) -> pd.DataFrame:
    """Read the audit columns, tolerating column-name whitespace."""
    try:
        return pd.read_excel(path, sheet_name=sheet, usecols=CALCE_USECOLS)
    except ValueError:
        frame = pd.read_excel(path, sheet_name=sheet)
        frame.columns = [str(column).strip() for column in frame.columns]
        missing = [column for column in CALCE_USECOLS if column not in frame.columns]
        if missing:
            raise ValueError(f"missing columns on {sheet}: {missing}") from None
        return frame.loc[:, CALCE_USECOLS].copy()


def _calce_signature(row: dict) -> tuple:
    """A hashable measured-trajectory fingerprint for one cycle/block."""

    def number(key: str) -> float | None:
        value = row.get(key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return round(value, 8) if np.isfinite(value) else None

    return (
        row.get("start_datetime"),
        number("capacity_Ah"),
        number("qd_Ah"),
        number("qc_Ah"),
        number("voltage_min_V"),
        number("voltage_max_V"),
        number("duration_s"),
        row.get("samples"),
    )


def _calce_cycle_metrics(
    group: pd.DataFrame,
    cell: str,
    rule: dict,
    capture_curve: bool,
) -> tuple[dict, tuple[np.ndarray, np.ndarray] | None]:
    qd = pd.to_numeric(group["Discharge_Capacity(Ah)"], errors="coerce")
    qc = pd.to_numeric(group["Charge_Capacity(Ah)"], errors="coerce")
    current = pd.to_numeric(group["Current(A)"], errors="coerce")
    voltage = pd.to_numeric(group["Voltage(V)"], errors="coerce")
    test_time = pd.to_numeric(group["Test_Time(s)"], errors="coerce")
    timestamps = group["_timestamp"]

    qd_span = float(qd.max() - qd.min()) if qd.notna().any() else float("nan")
    qc_span = float(qc.max() - qc.min()) if qc.notna().any() else float("nan")
    vmin = float(voltage.min()) if voltage.notna().any() else float("nan")
    vmax = float(voltage.max()) if voltage.notna().any() else float("nan")
    imin = float(current.min()) if current.notna().any() else float("nan")
    imax = float(current.max()) if current.notna().any() else float("nan")

    n_nan = int(
        qd.isna().sum()
        + qc.isna().sum()
        + current.isna().sum()
        + voltage.isna().sum()
    )
    out_of_range = int(
        ((voltage < 2.0) | (voltage > 4.5)).sum()
        + (current.abs() > 3.0).sum()
    )
    if test_time.notna().any():
        duration = float(test_time.max() - test_time.min())
    elif timestamps.notna().any():
        duration = float((timestamps.max() - timestamps.min()).total_seconds())
    else:
        duration = float("nan")

    # Each full discharge step gives a capacity at a known current.  CS2_3 is
    # a six-rate characterisation block; the slowest step is the most direct
    # capacity measure, while the total block throughput is still recorded.
    step_caps: list[tuple[float, float]] = []
    for _, step_group in group.groupby("Step_Index", sort=True):
        step_current = pd.to_numeric(step_group["Current(A)"], errors="coerce")
        step_qd = pd.to_numeric(step_group["Discharge_Capacity(Ah)"], errors="coerce")
        span = float(step_qd.max() - step_qd.min()) if step_qd.notna().any() else 0.0
        mean_current = float(step_current.mean()) if step_current.notna().any() else 0.0
        if mean_current < -0.05 and span > 0.002:
            step_caps.append((abs(mean_current), span))

    n_discharge_steps = len(step_caps)
    low_rate_capacity = (
        min(step_caps, key=lambda item: item[0])[1] if step_caps else float("nan")
    )

    if rule["mode"] == "block":
        capacity = low_rate_capacity
        valid = (
            np.isfinite(capacity)
            and capacity >= rule["min_capacity_Ah"]
            and n_discharge_steps >= rule["min_discharge_steps"]
        )
        cycle_type = "block" if valid else "partial_block"
    else:
        capacity = qd_span
        valid = (
            np.isfinite(capacity)
            and capacity >= rule["min_capacity_Ah"]
            and np.isfinite(vmin)
            and vmin <= rule["max_voltage_min_V"]
        )
        cycle_type = (
            ("full" if rule["mode"] == "full" else "checkup") if valid else "partial"
        )

    start_iso = end_iso = None
    if timestamps.notna().any():
        start = timestamps.min()
        end = timestamps.max()
        start_iso = start.isoformat() if pd.notna(start) else None
        end_iso = end.isoformat() if pd.notna(end) else None

    row = {
        "cell_id": cell,
        "cycle_type": cycle_type,
        "basis": rule["basis"],
        "capacity_Ah": capacity,
        "qd_Ah": qd_span,
        "qc_Ah": qc_span,
        "low_rate_capacity_Ah": low_rate_capacity,
        "n_discharge_steps": n_discharge_steps,
        "voltage_min_V": vmin,
        "voltage_max_V": vmax,
        "current_min_A": imin,
        "current_max_A": imax,
        "duration_s": duration,
        "samples": int(len(group)),
        "n_nan": n_nan,
        "out_of_range": out_of_range,
        "start_datetime": start_iso,
        "end_datetime": end_iso,
    }

    curve = None
    if capture_curve and cycle_type == "full":
        mask = (current < -0.05) & voltage.notna() & qd.notna()
        seg_q = qd[mask].to_numpy(dtype=float)
        seg_v = voltage[mask].to_numpy(dtype=float)
        if seg_q.size >= 20 and np.nanmax(seg_q) - np.nanmin(seg_q) > 0.05:
            seg_q = seg_q - np.nanmin(seg_q)
            order = np.argsort(seg_q)
            seg_q, seg_v = seg_q[order], seg_v[order]
            keep = np.concatenate(([True], np.diff(seg_q) > 0))
            seg_q, seg_v = seg_q[keep], seg_v[keep]
            stride = max(1, seg_q.size // 300)
            curve = (seg_q[::stride], seg_v[::stride])

    return row, curve


def _calce_sustained_eol(
    df: pd.DataFrame,
    eligible_mask: pd.Series,
    window: int,
    run: int,
) -> dict:
    """Find an EOL crossing that persists for several eligible cycles."""
    sub = df.loc[eligible_mask].sort_values("global_cycle")
    if sub.empty:
        return {
            "cycle": None,
            "first_below": None,
            "confirmed": False,
            "final_soh": float("nan"),
            "window": window,
            "run": run,
        }

    soh = sub["soh"].astype(float)
    rolled = soh.rolling(window=window, min_periods=window).median()
    below = rolled <= 0.8

    first_below = None
    for cycle, value in zip(sub["global_cycle"], soh):
        if np.isfinite(value) and value <= 0.8:
            first_below = int(cycle)
            break

    eol_cycle = None
    count = 0
    run_start = None
    for cycle, flag in zip(sub["global_cycle"], below):
        if bool(flag):
            if count == 0:
                run_start = int(cycle)
            count += 1
            if count >= run:
                eol_cycle = run_start
                break
        else:
            count = 0
            run_start = None

    final_soh = float(soh.dropna().iloc[-1]) if soh.notna().any() else float("nan")
    return {
        "cycle": eol_cycle,
        "first_below": first_below,
        "confirmed": bool(np.isfinite(final_soh) and final_soh <= 0.8),
        "final_soh": final_soh,
        "window": window,
        "run": run,
    }


def calce_analysis() -> tuple[pd.DataFrame, dict]:
    root = RAW / "CALCE" / "extracted"
    files = sorted(root.glob("CS2_*/*/*.xlsx"))
    by_cell: dict[str, list[Path]] = {}
    for path in files:
        match = re.match(r"(CS2_\d+)_", path.name)
        if match:
            by_cell.setdefault(match.group(1), []).append(path)

    all_cycles: list[dict] = []
    inventory: list[dict] = []
    cell_meta: dict[str, dict] = {}
    audit_curves: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}

    for cell, paths in sorted(by_cell.items()):
        rule = CALCE_RULES.get(cell)
        if rule is None:
            continue

        # File order is only a tie-breaker: cycle order is assigned later from
        # the measured Date_Time.  Earlier filename dates win duplicate ties.
        paths = sorted(
            paths,
            key=lambda item: (
                _calce_info_start(item)
                or _calce_name_date(item.name)
                or datetime(1900, 1, 1),
                item.name,
            ),
        )

        cell_rows: list[dict] = []
        file_infos: list[dict] = []
        seen_files: dict[tuple, str] = {}
        curves: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
        print(f"CALCE {cell}: {len(paths)} files", flush=True)

        for index, path in enumerate(paths, 1):
            start_info = _calce_info_start(path)
            sheets = _calce_channel_sheets(path)
            if not sheets:
                file_infos.append(
                    {
                        "cell_id": cell,
                        "file": path.name,
                        "start_datetime_info": start_info.isoformat() if start_info else None,
                        "channel_sheets": None,
                        "rows_raw": 0,
                        "rows_used": 0,
                        "duplicate_rows_removed": 0,
                        "cycles": 0,
                        "duplicate_file_of": None,
                        "median_sampling_s": None,
                        "voltage_min_V": None,
                        "voltage_max_V": None,
                        "current_min_A": None,
                        "current_max_A": None,
                        "n_nan": 0,
                        "timestamp_nat": 0,
                        "out_of_range": 0,
                        "error": "no Channel sheet",
                    }
                )
                continue

            frames = []
            try:
                for sheet in sheets:
                    frame = _calce_read_sheet(path, sheet)
                    frame["channel_sheet"] = sheet
                    frames.append(frame)
            except Exception as error:  # noqa: BLE001 - keep auditing other files
                file_infos.append(
                    {
                        "cell_id": cell,
                        "file": path.name,
                        "start_datetime_info": start_info.isoformat() if start_info else None,
                        "channel_sheets": ";".join(sheets),
                        "rows_raw": 0,
                        "rows_used": 0,
                        "duplicate_rows_removed": 0,
                        "cycles": 0,
                        "duplicate_file_of": None,
                        "median_sampling_s": None,
                        "voltage_min_V": None,
                        "voltage_max_V": None,
                        "current_min_A": None,
                        "current_max_A": None,
                        "n_nan": 0,
                        "timestamp_nat": 0,
                        "out_of_range": 0,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                continue

            raw = pd.concat(frames, ignore_index=True)
            raw.columns = [str(column).strip() for column in raw.columns]
            for column in [
                "Test_Time(s)",
                "Step_Index",
                "Cycle_Index",
                "Current(A)",
                "Voltage(V)",
                "Charge_Capacity(Ah)",
                "Discharge_Capacity(Ah)",
            ]:
                raw[column] = pd.to_numeric(raw[column], errors="coerce")
            raw["Date_Time"] = pd.to_datetime(raw["Date_Time"], errors="coerce")

            rows_raw = int(len(raw))

            # Multi-channel CS2_3 exports repeat the boundary row.  De-duplicate
            # only rows with a real timestamp so NaT rows are never collapsed.
            valid_time = raw["Date_Time"].notna()
            deduped = raw.loc[valid_time].drop_duplicates(
                subset=["Date_Time"], keep="first"
            )
            raw = pd.concat([deduped, raw.loc[~valid_time]], ignore_index=True)
            duplicate_rows = rows_raw - int(len(raw))

            raw = raw.dropna(
                subset=["Cycle_Index", "Test_Time(s)", "Voltage(V)"]
            ).copy()
            if raw.empty:
                file_infos.append(
                    {
                        "cell_id": cell,
                        "file": path.name,
                        "start_datetime_info": start_info.isoformat() if start_info else None,
                        "channel_sheets": ";".join(sheets),
                        "rows_raw": rows_raw,
                        "rows_used": 0,
                        "duplicate_rows_removed": duplicate_rows,
                        "cycles": 0,
                        "duplicate_file_of": None,
                        "median_sampling_s": None,
                        "voltage_min_V": None,
                        "voltage_max_V": None,
                        "current_min_A": None,
                        "current_max_A": None,
                        "n_nan": 0,
                        "timestamp_nat": 0,
                        "out_of_range": 0,
                        "error": "no valid rows after parsing",
                    }
                )
                continue

            raw["Cycle_Index"] = raw["Cycle_Index"].astype(int)

            base = pd.Timestamp(start_info) if start_info else pd.Timestamp("1970-01-01")
            if raw["Date_Time"].notna().any():
                timestamps = raw["Date_Time"].copy()
                if timestamps.isna().any():
                    timestamps = timestamps.fillna(
                        base + pd.to_timedelta(raw["Test_Time(s)"], unit="s")
                    )
            else:
                timestamps = base + pd.to_timedelta(raw["Test_Time(s)"], unit="s")
            raw["_timestamp"] = timestamps

            dt = med_dt(raw["Test_Time(s)"].to_numpy())
            file_cycles: list[dict] = []
            file_curves: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}

            for local_cycle, group in raw.groupby("Cycle_Index", sort=True):
                row, curve = _calce_cycle_metrics(
                    group,
                    cell,
                    rule,
                    capture_curve=cell in {"CS2_35", "CS2_36"},
                )
                row["file"] = path.name
                row["local_cycle"] = int(local_cycle)
                file_cycles.append(row)
                if curve is not None:
                    file_curves[(path.name, int(local_cycle))] = curve

            file_signature = tuple(
                sorted((_calce_signature(row) for row in file_cycles), key=repr)
            )
            duplicate_of = seen_files.get(file_signature)
            duplicate_file = duplicate_of is not None
            if not duplicate_file:
                seen_files[file_signature] = path.name
                cell_rows.extend(file_cycles)
                curves.update(file_curves)

            file_infos.append(
                {
                    "cell_id": cell,
                    "file": path.name,
                    "start_datetime_info": start_info.isoformat() if start_info else None,
                    "channel_sheets": ";".join(sheets),
                    "rows_raw": rows_raw,
                    "rows_used": int(len(raw)),
                    "duplicate_rows_removed": int(duplicate_rows),
                    "cycles": int(len(file_cycles)),
                    "duplicate_file_of": duplicate_of,
                    "median_sampling_s": dt,
                    "voltage_min_V": float(raw["Voltage(V)"].min()),
                    "voltage_max_V": float(raw["Voltage(V)"].max()),
                    "current_min_A": float(raw["Current(A)"].min()),
                    "current_max_A": float(raw["Current(A)"].max()),
                    "n_nan": int(
                        raw[
                            [
                                "Current(A)",
                                "Voltage(V)",
                                "Charge_Capacity(Ah)",
                                "Discharge_Capacity(Ah)",
                            ]
                        ]
                        .isna()
                        .sum()
                        .sum()
                    ),
                    "timestamp_nat": int(raw["Date_Time"].isna().sum()),
                    "out_of_range": int(
                        ((raw["Voltage(V)"] < 2.0) | (raw["Voltage(V)"] > 4.5)).sum()
                        + (raw["Current(A)"].abs() > 3.0).sum()
                    ),
                }
            )
            print(
                f"CALCE {cell}: {index}/{len(paths)} {path.name} "
                f"rows={len(raw)} cycles={len(file_cycles)} duplicate={duplicate_file}",
                flush=True,
            )

        if not cell_rows:
            continue

        df = pd.DataFrame(cell_rows)
        df["_sort_time"] = pd.to_datetime(df["start_datetime"], errors="coerce")
        df = (
            df.sort_values(
                ["_sort_time", "file", "local_cycle"], na_position="last"
            )
            .drop(columns=["_sort_time"])
            .reset_index(drop=True)
        )

        # Safety net for overlapping exports that are not whole-file duplicates.
        signatures = [_calce_signature(row) for row in df.to_dict("records")]
        seen_cycles = set()
        keep_cycle: list[bool] = []
        for signature in signatures:
            if signature in seen_cycles:
                keep_cycle.append(False)
            else:
                seen_cycles.add(signature)
                keep_cycle.append(True)
        duplicate_cycles_removed = int(len(keep_cycle) - sum(keep_cycle))
        df = df.loc[keep_cycle].reset_index(drop=True)
        df["global_cycle"] = np.arange(1, len(df) + 1)

        eligible_mask = df["cycle_type"].isin(["full", "checkup", "block"])
        if not eligible_mask.any():
            continue

        first_capacity = float(df.loc[eligible_mask, "capacity_Ah"].iloc[0])
        if not np.isfinite(first_capacity) or first_capacity <= 0:
            continue

        # SOH must exist on ``df`` before ``eligible`` is sliced, otherwise the
        # slice never sees the column.
        df["soh"] = np.nan
        df.loc[eligible_mask, "soh"] = (
            df.loc[eligible_mask, "capacity_Ah"] / first_capacity
        )
        eligible = df.loc[eligible_mask].sort_values("global_cycle")

        final_row = eligible.iloc[-1]
        final_soh = float(final_row["capacity_Ah"]) / first_capacity
        partial_capacity = df.loc[~eligible_mask, "capacity_Ah"].dropna()
        window, run = (3, 2) if rule["mode"] == "checkup" else (5, 3)
        # Guard: keep auditing the remaining cells if one cell fails.

        eol = _calce_sustained_eol(df, eligible_mask, window, run)

        sampling = [
            float(info["median_sampling_s"])
            for info in file_infos
            if info.get("median_sampling_s")
        ]
        cell_meta[cell] = {
            "files": len(paths),
            "files_processed": len(file_infos),
            "files_unique": int(
                sum(1 for info in file_infos if not info.get("duplicate_file_of"))
            ),
            "duplicate_trajectory_files": [
                {"file": info["file"], "duplicate_of": info["duplicate_file_of"]}
                for info in file_infos
                if info.get("duplicate_file_of")
            ],
            "duplicate_cycles_removed": duplicate_cycles_removed,
            "measurement_rows_unique": int(
                sum(
                    int(info.get("rows_used", 0))
                    for info in file_infos
                    if not info.get("duplicate_file_of")
                )
            ),
            "cycles_total": int(len(df)),
            "cycles_eligible": int(eligible_mask.sum()),
            "cycle_type_counts": {
                str(key): int(value)
                for key, value in df["cycle_type"].value_counts().items()
            },
            "soh_basis": CALCE_SOH_BASIS.get(cell, rule["basis"]),
            "first_capacity_Ah": round(first_capacity, 4),
            "final_capacity_Ah": round(float(final_row["capacity_Ah"]), 4),
            "final_soh_pct": round(100 * final_soh, 2),
            "min_soh_pct": round(100 * float(eligible["soh"].min()), 2),
            "max_global_cycle": int(df["global_cycle"].max()),
            "eol_80pct_cycle": eol["cycle"],
            "eol_80pct_first_single_cycle": eol["first_below"],
            "eol_confirmed": eol["confirmed"],
            "eol_method": (
                f"trailing rolling median ({window} eligible cycles), "
                f"{run} consecutive cycles <= 80%"
            ),
            "voltage_range_V": [
                round(float(df["voltage_min_V"].min()), 4),
                round(float(df["voltage_max_V"].max()), 4),
            ],
            "current_range_A": [
                round(float(df["current_min_A"].min()), 4),
                round(float(df["current_max_A"].max()), 4),
            ],
            "median_sampling_s": (
                round(float(np.median(sampling)), 4) if sampling else None
            ),
            "n_nan": int(df["n_nan"].sum()),
            "out_of_range": int(df["out_of_range"].sum()),
            "partial_capacity_range_Ah": (
                [
                    round(float(partial_capacity.min()), 4),
                    round(float(partial_capacity.max()), 4),
                ]
                if len(partial_capacity)
                else None
            ),
            "native_temperature": False,
            "native_soc_labels": False,
        }

        all_cycles.extend(df.to_dict("records"))
        inventory.extend(file_infos)

        for cycle_row in df.loc[eligible_mask].to_dict("records"):
            key = (cycle_row["file"], int(cycle_row["local_cycle"]))
            if key in curves:
                audit_curves[(cell, int(cycle_row["global_cycle"]))] = curves[key]

    cycles_df = pd.DataFrame(all_cycles)
    if not cycles_df.empty:
        cycles_df = cycles_df.sort_values(["cell_id", "global_cycle"]).reset_index(
            drop=True
        )
        cycles_df.to_csv(AUDIT / "calce_cycles.csv", index=False)

    inventory_df = pd.DataFrame(inventory)
    if not inventory_df.empty:
        inventory_df.to_csv(AUDIT / "calce_file_inventory.csv", index=False)

    if not cycles_df.empty:
        eligible_df = cycles_df[cycles_df["soh"].notna()].copy()
        fig, (ax_a, ax_b) = plt.subplots(
            1,
            2,
            figsize=(10.8, 4.2),
            gridspec_kw={"width_ratios": [2.1, 1.0]},
            sharey=True,
        )
        for color, (cell, group) in zip(COLORS, eligible_df.groupby("cell_id")):
            group = group.sort_values("global_cycle")
            first_type = str(group["cycle_type"].iloc[0])
            panel = ax_b if first_type == "checkup" else ax_a
            basis_label = {
                "checkup": "check-up",
                "block": "low-rate",
                "full": "full-cycle",
            }.get(first_type, first_type)
            label = f"{cell} ({basis_label})"
            panel.plot(
                group["global_cycle"],
                100 * group["soh"],
                color=color,
                lw=1.4,
                label=label,
            )
            eol_cycle = cell_meta.get(cell, {}).get("eol_80pct_cycle")
            if eol_cycle is not None:
                point = group[group["global_cycle"] == eol_cycle]
                if not point.empty:
                    panel.scatter(
                        point["global_cycle"],
                        100 * point["soh"],
                        color=color,
                        s=24,
                        zorder=5,
                        edgecolor="white",
                        linewidth=0.4,
                    )

        for panel in (ax_a, ax_b):
            panel.axhline(80, color="#4d4d4d", ls="--", lw=1, label="80% EOL")
            panel.set_ylim(0, 105)
            panel.set_xlabel("Global cycle number")
        ax_a.set_ylabel("SOH / %")
        ax_a.set_title("Full-cycle cells and six-rate blocks")
        ax_b.set_title("CS2_24 partial cycling")
        ax_a.legend(frameon=False, fontsize=7, loc="lower left")
        ax_b.legend(frameon=False, fontsize=7, loc="lower left")
        fig.suptitle(
            "CALCE CS2 prismatic cells: measured capacity fade (eligible cycles only)",
            fontweight="bold",
        )
        fig.tight_layout()
        save_fig(fig, "fig11_calce_soh.png")

        if audit_curves:
            curve_cell = (
                "CS2_35"
                if any(cell == "CS2_35" for cell, _ in audit_curves)
                else sorted({cell for cell, _ in audit_curves})[0]
            )
            available = sorted(
                cycle for cell, cycle in audit_curves if cell == curve_cell
            )
            if available:
                picks = sorted(
                    {available[0], available[len(available) // 2], available[-1]}
                )
                fig, ax = plt.subplots(figsize=(6.8, 4.2))
                for color, cycle in zip(COLORS, picks):
                    q, v = audit_curves[(curve_cell, cycle)]
                    row = cycles_df[
                        (cycles_df["cell_id"] == curve_cell)
                        & (cycles_df["global_cycle"] == cycle)
                    ]
                    soh_text = (
                        f"{100 * float(row['soh'].iloc[0]):.0f}% SOH"
                        if not row.empty
                        else ""
                    )
                    ax.plot(
                        1000 * q,
                        v,
                        color=color,
                        lw=1.4,
                        label=f"cycle {cycle} ({soh_text})",
                    )
                ax.set(xlabel="Discharged capacity / mAh", ylabel="Voltage / V")
                ax.legend(frameon=False)
                ax.set_title(
                    f"CALCE {curve_cell}: discharge curves at three ages",
                    fontweight="bold",
                )
                save_fig(fig, "fig12_calce_curves.png")

    total_rows = int(sum(int(info.get("rows_used", 0)) for info in inventory))
    unique_rows = int(
        sum(
            int(info.get("rows_used", 0))
            for info in inventory
            if not info.get("duplicate_file_of")
        )
    )
    summary = {
        "cells": cell_meta,
        "chemistry": "LiCoO2 cathode with trace Mn (provider statement)",
        "files": len(inventory),
        "files_unique": int(
            sum(1 for info in inventory if not info.get("duplicate_file_of"))
        ),
        "measurement_rows": total_rows,
        "measurement_rows_unique": unique_rows,
        "native_soc_labels": False,
        "native_temperature": False,
        "soh_label": (
            "measured capacity normalised to the first eligible cycle of each "
            "cell; basis differs by test design"
        ),
        "rul_label": "derived from a sustained rolling-median measured SOH crossing 80%",
        "notes": [
            "CALCE CS2 Arbin exports contain no temperature channel and no native SOC label.",
            "Cycle_Index, capacity counters and Data_Point restart or split across spreadsheets; files were ordered by measured Date_Time, not filename.",
            "CS2_3 each Cycle_Index is a six-rate characterisation block; the second Channel sheet continues the same counters and shares one boundary row, which was de-duplicated by Date_Time.",
            "CS2_3 SOH uses the 0.11 A discharge capacity of each block. Block charge/discharge throughput varies with where the six-rate block boundary falls and is recorded only as a diagnostic.",
            "CS2_24 is partial cycling between 3.78 V and 4.2 V with exactly one full check-up at cycle 1 of each file.",
            "CS2_35_2_10_11.xlsx is an exact measured-trajectory duplicate of CS2_35_2_4_11.xlsx and was excluded.",
            "Partial cycles are retained in calce_cycles.csv with blank SOH so they can be audited but are excluded from SOH/RUL labels.",
            "EOL requires a sustained run of rolling-median SOH <= 80%, not a single low cycle.",
        ],
    }
    (AUDIT / "calce_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return cycles_df, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["nasa", "oxford", "energy", "samsung", "calce"])
    args = parser.parse_args()
    results = {}
    if "nasa" in args.datasets:
        print("Auditing NASA PCoE ...")
        _, _, results["nasa"] = nasa_analysis()
    if "oxford" in args.datasets:
        print("Auditing Oxford degradation dataset ...")
        _, results["oxford"] = oxford_analysis()
    if "energy" in args.datasets:
        print("Auditing Oxford energy-trading dataset ...")
        _, results["energy"] = energy_analysis()
    if "samsung" in args.datasets:
        print("Auditing Samsung NMC dataset ...")
        _, results["samsung"] = samsung_analysis()
    if "calce" in args.datasets:
        print("Auditing CALCE CS2 dataset ...")
        _, results["calce"] = calce_analysis()
    summary_path = AUDIT / "stage1_partial_summary.json"
    if summary_path.exists():
        try:
            combined = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            combined = {}
        combined.update(results)
        results = combined
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("done")


if __name__ == "__main__":
    main()
