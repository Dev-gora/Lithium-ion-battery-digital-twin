"""Build unified, analysis-ready tables from the audited raw datasets.

Only audit outputs (derived from raw measurements) are read here; raw
experimental files are never modified.  Every row keeps ``dataset`` and
``cell_id`` so that battery-wise splits remain possible downstream.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT = PROJECT_ROOT / "data" / "processed" / "audit"
OUT = PROJECT_ROOT / "data" / "processed" / "ml"
OUT.mkdir(parents=True, exist_ok=True)

DATASET_META = {
    "NASA": {"chemistry": "LCO", "rated_capacity_Ah": 2.0},
    "CALCE": {"chemistry": "LCO", "rated_capacity_Ah": 1.1},
    "Oxford": {"chemistry": "NMC", "rated_capacity_Ah": 0.740},
    "Samsung": {"chemistry": "NMC", "rated_capacity_Ah": 3.4},
    "MIT": {"chemistry": "LFP", "rated_capacity_Ah": 1.1},
}


def nasa_cycles() -> pd.DataFrame:
    path = AUDIT / "nasa_discharge_trajectory.csv"
    df = pd.read_csv(path)
    df = df[df["reported_capacity_Ah"].notna()].copy()
    out = pd.DataFrame(
        {
            "dataset": "NASA",
            "cell_id": df["cell_id"],
            "cycle_index": df["discharge_number"],
            "voltage_mean_V": df["voltage_mean_V"],
            "voltage_min_V": df["voltage_min_V"],
            "voltage_max_V": df["voltage_max_V"],
            "current_mean_A": df["current_mean_A"],
            "current_absmean_A": df["current_mean_A"].abs(),
            "temperature_mean_C": df["temperature_mean_C"],
            "temperature_max_C": df["temperature_max_C"],
            "duration_s": df["duration_s"],
            "capacity_Ah": df["reported_capacity_Ah"],
            "soh": df["soh"],
            "temperature_available": True,
        }
    )
    return out


def calce_cycles() -> pd.DataFrame:
    path = AUDIT / "calce_cycles.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["soh"].notna()].copy()
    out = pd.DataFrame(
        {
            "dataset": "CALCE",
            "cell_id": df["cell_id"],
            "cycle_index": df["global_cycle"],
            "voltage_mean_V": (df["voltage_min_V"] + df["voltage_max_V"]) / 2,
            "voltage_min_V": df["voltage_min_V"],
            "voltage_max_V": df["voltage_max_V"],
            "current_mean_A": (df["current_min_A"] + df["current_max_A"]) / 2,
            "current_absmean_A": df[["current_min_A", "current_max_A"]].abs().max(axis=1),
            "temperature_mean_C": np.nan,
            "temperature_max_C": np.nan,
            "duration_s": df["duration_s"],
            "capacity_Ah": df["capacity_Ah"] if "capacity_Ah" in df.columns else df["basis_capacity_Ah"],
            "soh": df["soh"],
            "temperature_available": False,
        }
    )
    return out


def oxford_cycles() -> pd.DataFrame:
    path = AUDIT / "oxford_capacity_trajectory.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return pd.DataFrame(
        {
            "dataset": "Oxford",
            "cell_id": df["cell_id"],
            "cycle_index": df["characterisation_cycle"].replace(0, 1),
            "voltage_mean_V": np.nan,
            "voltage_min_V": np.nan,
            "voltage_max_V": np.nan,
            "current_mean_A": np.nan,
            "current_absmean_A": np.nan,
            "temperature_mean_C": 40.0,  # chamber setpoint, provider readme
            "temperature_max_C": np.nan,
            "duration_s": np.nan,
            "capacity_Ah": df["capacity_pOCV_mAh"] / 1000.0,
            "soh": df["soh_pOCV"],
            "temperature_available": True,
        }
    )


def samsung_cycles() -> pd.DataFrame:
    path = AUDIT / "samsung_cycle_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["valid_discharge"]].copy()
    out = pd.DataFrame(
        {
            "dataset": "Samsung",
            "cell_id": df["battery"],
            "cycle_index": df["global_cycle"],
            "voltage_mean_V": (df["Vmin"] + df["Vmax"]) / 2,
            "voltage_min_V": df["Vmin"],
            "voltage_max_V": df["Vmax"],
            "current_mean_A": np.nan,
            "current_absmean_A": np.nan,
            "temperature_mean_C": df["Tmean"],
            "temperature_max_C": df["Tmax"],
            "duration_s": df["duration_s"],
            "capacity_Ah": df["Qdis_mAh"] / 1000.0,
            "soh": df["soh_first"],
            "temperature_available": True,
        }
    )
    return out


def mit_cycles() -> pd.DataFrame:
    """Fast-charging dataset: 49 LFP cells, per-cycle summary records."""
    path = AUDIT / "mit_cycles.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[~df["capacity_outlier"].astype(bool)].copy()
    return pd.DataFrame(
        {
            "dataset": "MIT",
            "cell_id": df["cell_id"],
            "cycle_index": df["cycle"],
            "voltage_mean_V": np.nan,
            "voltage_min_V": np.nan,
            "voltage_max_V": np.nan,
            "current_mean_A": np.nan,
            "current_absmean_A": np.nan,
            "temperature_mean_C": df["Tavg_C"],
            "temperature_max_C": df["Tmax_C"],
            "duration_s": df["chargetime_min"] * 60.0,
            "capacity_Ah": df["QDischarge_valid_Ah"],
            "soh": df["soh_first_measured"],
            "temperature_available": True,
        }
    )


def unified_cycle_table() -> pd.DataFrame:
    frames = [f() for f in (nasa_cycles, calce_cycles, oxford_cycles, samsung_cycles, mit_cycles)]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    df["chemistry"] = df["dataset"].map({k: v["chemistry"] for k, v in DATASET_META.items()})
    df = df.sort_values(["dataset", "cell_id", "cycle_index"]).reset_index(drop=True)

    # Causal (backward-looking) capacity features, computed per cell.
    def per_cell(group: pd.DataFrame) -> pd.DataFrame:
        g = group.sort_values("cycle_index").copy()
        g["capacity_prev"] = g["capacity_Ah"].shift(1)
        g["capacity_delta"] = g["capacity_Ah"].diff()
        g["capacity_slope_5"] = (
            g["capacity_Ah"].rolling(5, min_periods=2).apply(
                lambda y: np.polyfit(np.arange(len(y)), y, 1)[0], raw=True
            )
        )
        g["soh_slope_5"] = (
            g["soh"].rolling(5, min_periods=2).apply(
                lambda y: np.polyfit(np.arange(len(y)), y, 1)[0], raw=True
            )
        )
        g["cycle_since_start"] = g["cycle_index"] - g["cycle_index"].min()
        return g

    # pandas >= 2.2 may drop the grouping columns inside apply, so iterate
    # explicitly and keep the identifiers on every row.
    parts = [per_cell(group) for _, group in df.groupby(["dataset", "cell_id"], sort=False)]
    df = pd.concat(parts, ignore_index=True)
    return df.sort_values(["dataset", "cell_id", "cycle_index"]).reset_index(drop=True)


def save_unified() -> pd.DataFrame:
    df = unified_cycle_table()
    df.to_csv(OUT / "cycle_features.csv", index=False)
    return df


if __name__ == "__main__":
    table = save_unified()
    print(table.groupby(["dataset", "cell_id"]).size())
    print("rows:", len(table))
