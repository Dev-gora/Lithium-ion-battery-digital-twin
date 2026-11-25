"""Memory-safe audit of the MIT / Severson fast-charging dataset.

The two cycling files are MATLAB 7.3 (HDF5).  They are never loaded whole:
per-cell summary arrays are read directly and voltage/current/temperature
trajectories are sampled at the first, middle and last cycle.  The small
MATLAB 5.0 files in the same directory are MCOS ``table`` objects and are
documented separately rather than decoded.

Run from the project root:
    python research/mit_audit.py
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw" / "MIT"
AUDIT = PROJECT_ROOT / "data" / "processed" / "audit"
FIGS = PROJECT_ROOT / "data" / "reports" / "figures"
for directory in (AUDIT, FIGS):
    directory.mkdir(parents=True, exist_ok=True)

NOMINAL_AH = 1.1
CAPACITY_LIMIT_AH = 1.5 * NOMINAL_AH
TEMP_MIN_C = -20.0
TEMP_MAX_C = 80.0
FILES = [
    {
        "tag": "B1",
        "file": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
        "label": "batch 2018-02-20",
    },
    {
        "tag": "B4",
        "file": "2018-04-03_varcharge_batchdata_updated_struct_errorcorrect.mat",
        "label": "varcharge 2018-04-03",
    },
]
COLORS = {"B1": "#1f5a7a", "B4": "#c44e52"}


def _chars(dataset: h5py.Dataset) -> str:
    values = np.asarray(dataset[()]).ravel()
    return "".join(
        chr(int(value)) for value in values if 0 < int(value) < 0x110000
    )


def _reference_array(
    handle: h5py.File, group: h5py.Group, field: str, index: int
) -> np.ndarray:
    dataset = group[field]
    reference = dataset[index, 0] if dataset.ndim == 2 else dataset[index]
    return np.asarray(handle[reference][()]).ravel()


def _summary_array(group: h5py.Group, field: str) -> np.ndarray:
    return np.asarray(group[field][()]).ravel().astype(float)


def _safe_float(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "n": int(values.size),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def _valid_temperature(values: np.ndarray) -> tuple[np.ndarray, int]:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    valid = finite & (values >= TEMP_MIN_C) & (values <= TEMP_MAX_C)
    return values[valid], int((finite & ~valid).sum())


def _sustained_eol(
    cycles: np.ndarray, soh: np.ndarray, window: int = 5, run: int = 3
) -> dict:
    frame = pd.DataFrame({"cycle": cycles, "soh": soh}).dropna()
    frame = frame.sort_values("cycle")
    if frame.empty:
        return {
            "cycle": None,
            "first_below": None,
            "confirmed": False,
            "window": window,
            "run": run,
        }
    rolled = frame["soh"].rolling(window=window, min_periods=window).median()
    below = rolled <= 0.8
    first_below = None
    for cycle, value in zip(frame["cycle"], frame["soh"]):
        if value <= 0.8:
            first_below = int(cycle)
            break
    eol = None
    count = 0
    run_start = None
    for cycle, flag in zip(frame["cycle"], below):
        if bool(flag):
            if count == 0:
                run_start = int(cycle)
            count += 1
            if count >= run:
                eol = run_start
                break
        else:
            count = 0
            run_start = None
    return {
        "cycle": eol,
        "first_below": first_below,
        "confirmed": bool(frame["soh"].iloc[-1] <= 0.8),
        "window": window,
        "run": run,
    }


def audit_cell(
    handle: h5py.File, batch: h5py.Group, index: int, tag: str
) -> tuple[dict, pd.DataFrame]:
    summary = handle[batch["summary"][index, 0]]
    cycles_group = handle[batch["cycles"][index, 0]]

    cycle = _summary_array(summary, "cycle")
    qd = _summary_array(summary, "QDischarge")
    qc = _summary_array(summary, "QCharge")
    ir = _summary_array(summary, "IR")
    tavg = _summary_array(summary, "Tavg")
    tmax = _summary_array(summary, "Tmax")
    tmin = _summary_array(summary, "Tmin")
    chargetime = _summary_array(summary, "chargetime")

    lengths = {
        "cycle": len(cycle),
        "QDischarge": len(qd),
        "QCharge": len(qc),
        "IR": len(ir),
        "Tavg": len(tavg),
        "Tmax": len(tmax),
        "Tmin": len(tmin),
        "chargetime": len(chargetime),
    }
    usable = min(lengths.values())
    cycle = cycle[:usable]
    qd = qd[:usable]
    qc = qc[:usable]
    ir = ir[:usable]
    tavg = tavg[:usable]
    tmax = tmax[:usable]
    tmin = tmin[:usable]
    chargetime = chargetime[:usable]

    cycle_life_dataset = handle[batch["cycle_life"][index, 0]]
    cycle_life = _safe_float(np.asarray(cycle_life_dataset[()]).ravel()[0])

    policy = _chars(handle[batch["policy"][index, 0]])
    policy_readable = _chars(handle[batch["policy_readable"][index, 0]])
    barcode_values = np.asarray(handle[batch["barcode"][index, 0]][()]).ravel()
    barcode_raw = " ".join(f"0x{int(value):08x}" for value in barcode_values)

    # A few exports contain ~2.1 Ah discharge values on 1.1 Ah cells, usually
    # at cycle 99/100.  They are physically impossible and are flagged rather
    # than allowed to create >100% SOH or false EOL crossings.
    capacity_valid = (
        np.isfinite(qd) & (qd > 0.1) & (qd <= CAPACITY_LIMIT_AH)
    )
    qd_clean = np.where(capacity_valid, qd, np.nan)
    capacity_outlier_cycles = [
        int(value)
        for value, valid in zip(cycle, capacity_valid)
        if not valid and np.isfinite(value)
    ]
    n_capacity_outliers = int(
        np.sum(~capacity_valid & np.isfinite(qd))
    )

    baseline_values = qd_clean[: min(5, usable)]
    baseline_capacity = (
        float(np.nanmedian(baseline_values))
        if np.isfinite(baseline_values).any()
        else float("nan")
    )
    if not np.isfinite(baseline_capacity) or baseline_capacity <= 0:
        baseline_capacity = NOMINAL_AH
    first_cycle_capacity = next(
        (float(value) for value in qd_clean if np.isfinite(value)), None
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        soh_first = qd_clean / baseline_capacity
        soh_nominal = qd_clean / NOMINAL_AH

    eol_first = _sustained_eol(cycle, soh_first)
    eol_nominal = _sustained_eol(cycle, soh_nominal)

    diffs = np.diff(cycle)
    cycle_gaps = int(np.sum(diffs != 1)) if len(cycle) > 1 else 0
    summary_nan = int(
        np.isnan(qd).sum()
        + np.isnan(qc).sum()
        + np.isnan(ir).sum()
        + np.isnan(tavg).sum()
    )

    tmin_valid, tmin_glitch = _valid_temperature(tmin)
    tmax_valid, tmax_glitch = _valid_temperature(tmax)
    tavg_valid, tavg_glitch = _valid_temperature(tavg)
    temperature_glitch_count = tmin_glitch + tmax_glitch + tavg_glitch
    temperature_glitch_cycles = int(
        (
            ~np.isfinite(tmin)
            | (tmin < TEMP_MIN_C)
            | (tmin > TEMP_MAX_C)
            | ~np.isfinite(tmax)
            | (tmax < TEMP_MIN_C)
            | (tmax > TEMP_MAX_C)
        ).sum()
    )

    sample_indices = sorted({0, max(0, usable // 2), max(0, usable - 1)})
    vmin = vmax = imin = imax = tmin_traj = tmax_traj = float("nan")
    trajectory_nan = 0
    trajectory_out_of_range = 0
    trajectory_temperature_glitch = 0
    stored_spacing_s: list[float] = []
    points_per_cycle: list[int] = []
    sampled_cycles: list[int] = []
    for sample_index in sample_indices:
        if sample_index < 0 or sample_index >= usable:
            continue
        sampled_cycles.append(int(cycle[sample_index]))
        for field in ("V", "I", "T", "t"):
            values = _reference_array(handle, cycles_group, field, sample_index)
            values = values.astype(float)
            trajectory_nan += int(np.isnan(values).sum())
            if field == "V":
                finite = values[np.isfinite(values)]
                if finite.size:
                    vmin = (
                        float(np.nanmin(finite))
                        if not np.isfinite(vmin)
                        else min(vmin, float(np.nanmin(finite)))
                    )
                    vmax = (
                        float(np.nanmax(finite))
                        if not np.isfinite(vmax)
                        else max(vmax, float(np.nanmax(finite)))
                    )
                trajectory_out_of_range += int(
                    ((values < 1.5) | (values > 4.5)).sum()
                )
            elif field == "I":
                finite = values[np.isfinite(values)]
                if finite.size:
                    imin = (
                        float(np.nanmin(finite))
                        if not np.isfinite(imin)
                        else min(imin, float(np.nanmin(finite)))
                    )
                    imax = (
                        float(np.nanmax(finite))
                        if not np.isfinite(imax)
                        else max(imax, float(np.nanmax(finite)))
                    )
                trajectory_out_of_range += int((np.abs(values) > 10.0).sum())
            elif field == "T":
                valid, glitches = _valid_temperature(values)
                trajectory_temperature_glitch += glitches
                if valid.size:
                    tmin_traj = (
                        float(np.nanmin(valid))
                        if not np.isfinite(tmin_traj)
                        else min(tmin_traj, float(np.nanmin(valid)))
                    )
                    tmax_traj = (
                        float(np.nanmax(valid))
                        if not np.isfinite(tmax_traj)
                        else max(tmax_traj, float(np.nanmax(valid)))
                    )
            elif field == "t":
                steps = np.diff(values)
                steps = steps[np.isfinite(steps) & (steps > 0)]
                if steps.size:
                    median_step = float(np.median(steps))
                    # MIT t is in hours; a 1 s sample is ~0.00028 h.
                    stored_spacing_s.append(
                        median_step * 3600.0 if median_step < 1.0 else median_step
                    )
        points_per_cycle.append(
            int(len(_reference_array(handle, cycles_group, "V", sample_index)))
        )

    valid_soh_first = soh_first[np.isfinite(soh_first)]
    valid_soh_nominal = soh_nominal[np.isfinite(soh_nominal)]
    valid_qd = qd_clean[np.isfinite(qd_clean)]
    final_soh_first = (
        float(valid_soh_first[-1]) if valid_soh_first.size else None
    )
    final_soh_nominal = (
        float(valid_soh_nominal[-1]) if valid_soh_nominal.size else None
    )
    cell_id = f"MIT_{tag}_C{index:03d}"
    ir_all_zero = bool(
        np.isfinite(ir).any() and np.all(ir[np.isfinite(ir)] == 0)
    )
    ir_median = (
        float(np.nanmedian(ir[np.isfinite(ir) & (ir != 0)]))
        if np.isfinite(ir).any() and np.any(ir[np.isfinite(ir)] != 0)
        else None
    )

    row = {
        "cell_id": cell_id,
        "batch_tag": tag,
        "provider_row_index": int(index),
        "policy": policy,
        "policy_readable": policy_readable,
        "barcode_raw": barcode_raw,
        "cycle_life_provider": cycle_life,
        "cycles": int(len(cycle)),
        "summary_length_mismatch": bool(len(set(lengths.values())) > 1),
        "summary_lengths": json.dumps(lengths),
        "cycle_gaps": cycle_gaps,
        "first_capacity_Ah": _safe_float(baseline_capacity),
        "first_cycle_capacity_Ah": _safe_float(first_cycle_capacity),
        "baseline_capacity_Ah": _safe_float(baseline_capacity),
        "baseline_method": "median of first 5 valid cycles",
        "final_capacity_Ah": _safe_float(valid_qd[-1]) if valid_qd.size else None,
        "final_capacity_raw_Ah": _safe_float(qd[-1]) if len(qd) else None,
        "min_capacity_Ah": _safe_float(np.nanmin(valid_qd)) if valid_qd.size else None,
        "max_capacity_Ah": _safe_float(np.nanmax(valid_qd)) if valid_qd.size else None,
        "max_capacity_raw_Ah": _safe_float(np.nanmax(qd)) if np.isfinite(qd).any() else None,
        "n_capacity_outliers": n_capacity_outliers,
        "capacity_outlier_cycles": json.dumps(capacity_outlier_cycles),
        "final_soh_first_pct": round(100 * final_soh_first, 2) if final_soh_first is not None else None,
        "final_soh_nominal_pct": round(100 * final_soh_nominal, 2) if final_soh_nominal is not None else None,
        "eol_80_first_cycle": eol_first["cycle"],
        "eol_80_first_single_cycle": eol_first["first_below"],
        "eol_80_first_confirmed": eol_first["confirmed"],
        "eol_80_nominal_cycle": eol_nominal["cycle"],
        "eol_80_nominal_confirmed": eol_nominal["confirmed"],
        "summary_nan": summary_nan,
        "zero_ir_cycles": int((ir == 0).sum()),
        "ir_all_zero": ir_all_zero,
        "ir_median_Ohm": ir_median,
        "charge_time_median_min": _safe_float(np.nanmedian(chargetime)) if np.isfinite(chargetime).any() else None,
        "temperature_avg_C": _safe_float(np.nanmean(tavg_valid)) if tavg_valid.size else None,
        "temperature_min_C": _safe_float(np.nanmin(tmin_valid)) if tmin_valid.size else None,
        "temperature_max_C": _safe_float(np.nanmax(tmax_valid)) if tmax_valid.size else None,
        "temperature_glitch_count_summary": temperature_glitch_count,
        "temperature_glitch_cycles": temperature_glitch_cycles,
        "trajectory_sampled_cycles": json.dumps(sampled_cycles),
        "trajectory_voltage_min_V": _safe_float(vmin),
        "trajectory_voltage_max_V": _safe_float(vmax),
        "trajectory_current_min_A": _safe_float(imin),
        "trajectory_current_max_A": _safe_float(imax),
        "trajectory_temperature_min_C": _safe_float(tmin_traj),
        "trajectory_temperature_max_C": _safe_float(tmax_traj),
        "trajectory_temperature_glitch": trajectory_temperature_glitch,
        "trajectory_nan": trajectory_nan,
        "trajectory_out_of_range": trajectory_out_of_range,
        "stored_point_spacing_min_s": round(min(stored_spacing_s), 4) if stored_spacing_s else None,
        "stored_point_spacing_max_s": round(max(stored_spacing_s), 4) if stored_spacing_s else None,
        "median_sampling_s": round(float(np.median(stored_spacing_s)), 4) if stored_spacing_s else None,
        "points_per_cycle_median": int(np.median(points_per_cycle)) if points_per_cycle else None,
    }

    per_cycle = pd.DataFrame(
        {
            "cell_id": cell_id,
            "batch_tag": tag,
            "cycle": cycle,
            "QDischarge_Ah": qd,
            "QDischarge_valid_Ah": qd_clean,
            "QCharge_Ah": qc,
            "IR_Ohm": ir,
            "Tavg_C": tavg,
            "Tmax_C": tmax,
            "Tmin_C": tmin,
            "chargetime_min": chargetime,
            "capacity_outlier": ~capacity_valid & np.isfinite(qd),
            "soh_first_measured": soh_first,
            "soh_nominal": soh_nominal,
            "eol_first_flag": (soh_first <= 0.8) if len(soh_first) else False,
        }
    )
    return row, per_cycle


def main() -> None:
    cells: dict[str, dict] = {}
    cycle_frames: list[pd.DataFrame] = []
    file_meta: list[dict] = []

    for spec in FILES:
        path = RAW / spec["file"]
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"MIT {spec['tag']}: {path.name} ({path.stat().st_size:,} bytes)", flush=True)
        with h5py.File(path, "r") as handle:
            batch = handle["batch"]
            n_cells = int(batch["cycles"].shape[0])
            batch_date = _chars(handle["batch_date"])
            for index in range(n_cells):
                row, per_cycle = audit_cell(handle, batch, index, spec["tag"])
                cells[row["cell_id"]] = row
                cycle_frames.append(per_cycle)
                if index % 10 == 0 or index == n_cells - 1:
                    print(
                        f"  {index + 1}/{n_cells} {row['cell_id']} "
                        f"cycles={row['cycles']} final={row['final_soh_first_pct']}%",
                        flush=True,
                    )
            file_meta.append(
                {
                    "tag": spec["tag"],
                    "file": spec["file"],
                    "bytes": int(path.stat().st_size),
                    "batch_date": batch_date,
                    "cells": n_cells,
                }
            )

    cells_df = pd.DataFrame(cells.values()).sort_values("cell_id")
    cells_df.to_csv(AUDIT / "mit_cells.csv", index=False)
    cycles_df = pd.concat(cycle_frames, ignore_index=True)
    cycles_df.to_csv(AUDIT / "mit_cycles.csv", index=False)

    cycle_counts = cells_df["cycles"].to_numpy(dtype=float)
    final_first = cells_df["final_soh_first_pct"].dropna().to_numpy(dtype=float)
    final_nominal = cells_df["final_soh_nominal_pct"].dropna().to_numpy(dtype=float)
    aggregate = {
        "cells": int(len(cells_df)),
        "cells_by_batch": {key: int(value) for key, value in cells_df["batch_tag"].value_counts().items()},
        "cycles_total": int(cycles_df.shape[0]),
        "cycles_per_cell": _stats(cycle_counts),
        "final_soh_first_pct": _stats(final_first),
        "final_soh_nominal_pct": _stats(final_nominal),
        "cells_with_provider_cycle_life": int(cells_df["cycle_life_provider"].notna().sum()),
        "cells_with_eol_first": int(cells_df["eol_80_first_cycle"].notna().sum()),
        "cells_with_eol_nominal": int(cells_df["eol_80_nominal_cycle"].notna().sum()),
        "capacity_outlier_rows": int(cells_df["n_capacity_outliers"].sum()),
        "cells_with_capacity_outliers": int((cells_df["n_capacity_outliers"] > 0).sum()),
        "summary_nan_total": int(cells_df["summary_nan"].sum()),
        "trajectory_nan_total": int(cells_df["trajectory_nan"].sum()),
        "trajectory_out_of_range_total": int(cells_df["trajectory_out_of_range"].sum()),
        "temperature_glitch_values_summary": int(cells_df["temperature_glitch_count_summary"].sum()),
        "temperature_glitch_values_trajectory": int(cells_df["trajectory_temperature_glitch"].sum()),
        "cells_with_temperature_glitches": int(((cells_df["temperature_glitch_count_summary"] > 0) | (cells_df["trajectory_temperature_glitch"] > 0)).sum()),
        "cells_with_all_zero_ir": int(cells_df["ir_all_zero"].sum()),
        "median_sampling_s": _stats(cells_df["median_sampling_s"].dropna()),
        "stored_point_spacing_min_s": _stats(cells_df["stored_point_spacing_min_s"].dropna()),
        "stored_point_spacing_max_s": _stats(cells_df["stored_point_spacing_max_s"].dropna()),
        "points_per_cycle_median": _stats(cells_df["points_per_cycle_median"].dropna()),
    }

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    max_cycles = 0
    for tag, group in cycles_df.groupby("batch_tag"):
        for _, cell in group.groupby("cell_id"):
            cell = cell.sort_values("cycle")
            ax.plot(
                cell["cycle"],
                100 * cell["soh_first_measured"],
                color=COLORS.get(tag, "#4d4d4d"),
                alpha=0.35,
                lw=0.8,
            )
            max_cycles = max(max_cycles, int(cell["cycle"].max()))
        ax.plot([], [], color=COLORS.get(tag, "#4d4d4d"), lw=1.4, label=f"{tag} cells")

    grid = np.arange(1, max_cycles + 1)
    matrix = np.full((len(cells_df), max_cycles), np.nan)
    for row_index, (_, cell) in enumerate(cells_df.iterrows()):
        cell_cycles = cycles_df[cycles_df["cell_id"] == cell["cell_id"]].sort_values("cycle")
        values = cell_cycles["soh_first_measured"].to_numpy(dtype=float)
        matrix[row_index, : len(values)] = values
    median = np.nanmedian(matrix, axis=0)
    ax.plot(grid, 100 * median, color="black", lw=2.0, label="median across cells")
    ax.axhline(80, color="#4d4d4d", ls="--", lw=1, label="80% EOL")
    ax.set(
        xlabel="Cycle number",
        ylabel="SOH / % (median of first 5 valid cycles)",
        ylim=(40, 105),
    )
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="lower left")
    ax.set_title("MIT / Severson fast-charging cells: measured capacity fade", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / "fig13_mit_soh.png")
    plt.close(fig)
    print(f"figure: {FIGS / 'fig13_mit_soh.png'}")

    summary = {
        "files": file_meta,
        "cells": cells,
        "aggregate": aggregate,
        "chemistry": "LFP/graphite A123 18650 (Severson et al. 2019)",
        "nominal_capacity_Ah": NOMINAL_AH,
        "native_temperature": True,
        "native_soc_labels": False,
        "native_rul_label": "cycle_life (80% of nominal capacity) provided for some cells",
        "soh_label": "QDischarge / median of first 5 valid QDischarge; provider-nominal SOH also recorded",
        "rul_label": "provider cycle_life plus derived sustained 80% crossing",
        "notes": [
            "The two cycling files are MATLAB 7.3/HDF5; each cell stores a summary table and per-cycle trajectory references.",
            "Voltage/current/temperature ranges and missing-value counts are sampled at the first, middle and last cycle per cell; capacity and SOH use every summary cycle.",
            "Capacity values above 1.65 Ah (1.5 x the 1.1 Ah nominal) occur in 11 rows across 8 cells, mostly at cycles 99-101; they are flagged as capacity_outlier and excluded from SOH/EOL.",
            "Temperature placeholders -270 C and 400 C occur in MIT_B1_C038 and in one summary entry of MIT_B4_C000; they are excluded from valid temperature ranges and counted separately.",
            "IR is exactly zero for all cycles in 9 cells, so IR is missing rather than a measured zero-ohm resistance for those cells.",
            "MIT t is in hours. The MAT trajectories are downsampled to roughly 1000 points per cycle; stored-point spacing (median about 300 s) is not the original sampling rate.",
            "initialdata_all.mat, finaldata_4.mat and finaldata_6and8.mat are MATLAB 5.0 MCOS table objects (charge_8C, final_4C, final_8C) and were not decoded; they are excluded from cell/cycle counts.",
            "The 2017-05-12 batch URL returned HTTP 404 and is not part of the audited set.",
            "No native SOC label is present; SOC must be coulomb-counted and marked as derived.",
        ],
    }
    (AUDIT / "mit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("done")


if __name__ == "__main__":
    main()
