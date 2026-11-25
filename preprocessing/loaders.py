"""Dataset-specific loaders returning measured time series for inspection.

Every function reads the original experimental files read-only and returns a
tidy frame with ``time_s, voltage_V, current_A, temperature_C`` plus whatever
capacity/step columns the dataset provides.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw"
AUDIT = PROJECT_ROOT / "data" / "processed" / "audit"

NASA_CELLS = ["B0005", "B0006", "B0007", "B0018"]


def nasa_cycles(cell: str) -> pd.DataFrame:
    path = RAW / "NASA" / "extracted" / "1_ BatteryAgingARC-FY08Q4" / f"{cell}.mat"
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    rows = []
    discharge_index = 0
    for idx, cycle in enumerate(np.atleast_1d(mat[cell].cycle)):
        op = str(cycle.type)
        if op == "discharge":
            discharge_index += 1
        if op not in {"charge", "discharge"}:
            continue
        d = cycle.data
        n = min(len(d.Time), len(d.Voltage_measured), len(d.Current_measured))
        rows.append(
            pd.DataFrame(
                {
                    "measurement_index": idx,
                    "operation": op,
                    "discharge_number": discharge_index if op == "discharge" else np.nan,
                    "time_s": np.asarray(d.Time, float)[:n],
                    "voltage_V": np.asarray(d.Voltage_measured, float)[:n],
                    "current_A": np.asarray(d.Current_measured, float)[:n],
                    "temperature_C": np.asarray(d.Temperature_measured, float)[:n],
                    "capacity_Ah": float(np.asarray(d.Capacity, float).ravel()[0])
                    if op == "discharge" and hasattr(d, "Capacity")
                    else np.nan,
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def calce_cycle(cell: str, global_cycle: int) -> pd.DataFrame:
    cycles = pd.read_csv(AUDIT / "calce_cycles.csv")
    row = cycles[(cycles.cell_id == cell) & (cycles.global_cycle == int(global_cycle))]
    if row.empty:
        return pd.DataFrame()
    row = row.iloc[0]
    path = RAW / "CALCE" / "extracted" / cell / cell / row["file"]
    frames = []
    for sheet in pd.ExcelFile(path).sheet_names:
        if not str(sheet).startswith("Channel"):
            continue
        frame = pd.read_excel(path, sheet_name=sheet)
        frame.columns = [str(c).strip() for c in frame.columns]
        frame["sheet"] = sheet
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df[df["Cycle_Index"].astype(float) == float(row["local_cycle"])]
    return pd.DataFrame(
        {
            "time_s": pd.to_numeric(df.get("Test_Time(s)"), errors="coerce"),
            "voltage_V": pd.to_numeric(df.get("Voltage(V)"), errors="coerce"),
            "current_A": pd.to_numeric(df.get("Current(A)"), errors="coerce"),
            "temperature_C": np.nan,
            "charge_capacity_Ah": pd.to_numeric(df.get("Charge_Capacity(Ah)"), errors="coerce"),
            "discharge_capacity_Ah": pd.to_numeric(df.get("Discharge_Capacity(Ah)"), errors="coerce"),
            "step_index": df.get("Step_Index"),
        }
    ).reset_index(drop=True)


def oxford_checkup(cell: str, characterisation_cycle: int) -> pd.DataFrame:
    path = RAW / "Oxford" / "Oxford_Battery_Degradation_Dataset_1.mat"
    mat = loadmat(path, squeeze_me=True, struct_as_record=False, variable_names=[cell])
    block = getattr(mat[cell], f"cyc{int(characterisation_cycle):04d}")
    frames = []
    for attr, label in [("C1dc", "1C discharge"), ("C1ch", "1C charge"),
                        ("OCVdc", "pseudo-OCV discharge"), ("OCVch", "pseudo-OCV charge")]:
        if not hasattr(block, attr):
            continue
        b = getattr(block, attr)
        frames.append(
            pd.DataFrame(
                {
                    "segment": label,
                    "time_s": (np.asarray(b.t, float) - np.asarray(b.t, float).min()) * 86400,
                    "voltage_V": np.asarray(b.v, float),
                    "current_A": np.full(np.asarray(b.v).size, np.nan),
                    "temperature_C": np.asarray(b.T, float),
                    "charge_mAh": np.asarray(b.q, float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def samsung_cycle(battery: str, global_cycle: int) -> pd.DataFrame:
    audit = pd.read_csv(AUDIT / "samsung_cycle_summary.csv")
    row = audit[(audit.battery == battery) & (audit.global_cycle == int(global_cycle))]
    if row.empty:
        return pd.DataFrame()
    # Locate the raw block whose local cycle index matches.
    row = row.iloc[0]
    blocks = {"Battery1": range(25, 32), "Battery2": range(32, 39)}[battery]
    offset = 0
    folder = RAW / "additional" / "Samsung_INR18650_35E"
    for number in blocks:
        path = folder / f"VZ5_008_021_VUT_D_{number:04d}_v1.txt"
        if not path.exists():
            continue
        df = pd.read_csv(path, sep="\t", encoding="latin-1", usecols=range(7))
        df.columns = ["cycle", "time_s", "Qdis_mAh", "Qch_mAh", "voltage_V", "current_mA", "temperature_C"]
        df["cycle"] = pd.to_numeric(df["cycle"], errors="coerce")
        local_max = int(df["cycle"].max())
        if offset + 1 <= int(global_cycle) <= offset + local_max + 1:
            local = int(global_cycle) - offset - 1
            sub = df[df["cycle"] == local].copy()
            for c in ["time_s", "Qdis_mAh", "Qch_mAh", "voltage_V", "current_mA", "temperature_C"]:
                sub[c] = pd.to_numeric(sub[c], errors="coerce")
            return pd.DataFrame(
                {
                    "time_s": sub["time_s"],
                    "voltage_V": sub["voltage_V"],
                    "current_A": sub["current_mA"] / 1000.0,
                    "temperature_C": sub["temperature_C"],
                    "discharge_capacity_mAh": sub["Qdis_mAh"],
                    "charge_capacity_mAh": sub["Qch_mAh"],
                }
            ).reset_index(drop=True)
        offset += local_max + 1
    return pd.DataFrame()


def dataset_cells(dataset: str) -> list[str]:
    table = pd.read_csv(AUDIT / {
        "NASA": "nasa_discharge_trajectory.csv",
        "CALCE": "calce_cycles.csv",
        "Oxford": "oxford_capacity_trajectory.csv",
        "Samsung": "samsung_cycle_summary.csv",
    }[dataset])
    column = {"NASA": "cell_id", "CALCE": "cell_id", "Oxford": "cell_id", "Samsung": "battery"}[dataset]
    return sorted(table[column].dropna().unique().tolist())
