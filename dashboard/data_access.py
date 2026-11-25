"""Cached access to audit outputs, trained models and simulation helpers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT = PROJECT_ROOT / "data" / "processed" / "audit"
ML = PROJECT_ROOT / "data" / "processed" / "ml"
SAVED = PROJECT_ROOT / "models" / "saved"
RESULTS = PROJECT_ROOT / "experiments" / "results"
REPORTS = PROJECT_ROOT / "data" / "reports"
FIGS = REPORTS / "figures"


@st.cache_data(show_spinner=False)
def cycle_table() -> pd.DataFrame:
    path = ML / "cycle_features.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def soc_samples() -> pd.DataFrame:
    path = ML / "soc_samples.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def metrics() -> dict:
    path = RESULTS / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def summary(name: str) -> dict:
    path = AUDIT / f"{name}_summary.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


@st.cache_resource(show_spinner=False)
def load_bundle(path: str):
    return joblib.load(SAVED / path)


def bundles() -> dict[str, str]:
    return {p.name: p.name for p in sorted(SAVED.glob("*.joblib"))}


def model_bundle(name: str, fallback: str | None = None):
    """Load a saved bundle by filename, optionally falling back to another artifact."""
    available = bundles()
    choice = name if name in available else fallback
    if choice is None or choice not in available:
        return None
    return load_bundle(choice)


LEDGERS = {
    "NASA": "nasa_discharge_trajectory.csv",
    "CALCE": "calce_cycles.csv",
    "Oxford": "oxford_capacity_trajectory.csv",
    "Samsung": "samsung_cycle_summary.csv",
}

CYCLE_KEYS = {
    "NASA": ("cell_id", "discharge_number"),
    "CALCE": ("cell_id", "global_cycle"),
    "Oxford": ("cell_id", "characterisation_cycle"),
    "Samsung": ("battery", "global_cycle"),
}


@st.cache_data(show_spinner=False)
def ledger(dataset: str) -> pd.DataFrame:
    """Per-cycle audit table for one dataset, as written to disk."""
    path = AUDIT / LEDGERS[dataset]
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def cycle_options(dataset: str, cell: str) -> list[int]:
    table = ledger(dataset)
    if table.empty:
        return []
    cell_col, cycle_col = CYCLE_KEYS[dataset]
    values = table.loc[table[cell_col] == cell, cycle_col].dropna().unique()
    return sorted(int(v) for v in values)


@st.cache_data(show_spinner="Reading the raw .mat file…")
def nasa_cell(cell: str) -> pd.DataFrame:
    from preprocessing import loaders

    return loaders.nasa_cycles(cell)


@st.cache_data(show_spinner="Reading the raw file…")
def raw_cycle(dataset: str, cell: str, cycle: int) -> pd.DataFrame:
    """Raw samples for one cycle, straight from the original file."""
    from preprocessing import loaders

    if dataset == "NASA":
        full = nasa_cell(cell)
        return full[full.discharge_number == cycle].reset_index(drop=True)
    if dataset == "CALCE":
        return loaders.calce_cycle(cell, cycle)
    if dataset == "Oxford":
        return loaders.oxford_checkup(cell, cycle)
    return loaders.samsung_cycle(cell, cycle)


MOVEMENT_FEATURES = {
    "soc_model.joblib": ["voltage_V", "current_A", "temperature_C", "power_W", "dv_dt",
                         "v_roll_mean", "i_roll_mean", "t_rel_s"],
}


@st.cache_data(show_spinner=False)
def fold_table(name: str) -> pd.DataFrame:
    path = RESULTS / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def predict_soh(bundle: dict, rows: pd.DataFrame) -> np.ndarray:
    features = bundle["features"]
    med = pd.Series(bundle.get("impute", {}))
    x = rows.reindex(columns=features)
    x = x.fillna(med.reindex(features)) if len(med) else x
    return bundle["model"].predict(x.to_numpy())


def uncertainty(bundle: dict, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
    """Tree-ensemble prediction interval (10th/90th percentile of tree outputs)."""
    model = bundle["model"]
    if not hasattr(model, "estimators_") or not hasattr(model.estimators_[0], "predict"):
        return None
    features = bundle["features"]
    med = pd.Series(bundle.get("impute", {}))
    x = rows.reindex(columns=features)
    x = x.fillna(med.reindex(features)) if len(med) else x
    tree_preds = np.stack([tree.predict(x.to_numpy()) for tree in model.estimators_])
    return np.percentile(tree_preds, 10, axis=0), np.percentile(tree_preds, 90, axis=0)
