"""Shared UI helpers and session-level battery configuration."""

from __future__ import annotations

import streamlit as st

from battery.battery_profiles import (
    BUILTIN_PROFILES,
    CHEMISTRY_PRESETS,
    BatteryProfile,
    custom_profile,
    get_profile,
)
from dashboard import data_access as da
from dashboard.theme import ACCENT, CHEM_COLORS, badge

DEFAULT_PROFILE = "EV"


def ensure_profile() -> BatteryProfile:
    if "profile" not in st.session_state:
        st.session_state["profile"] = get_profile(DEFAULT_PROFILE)
    return st.session_state["profile"]


def set_profile(profile: BatteryProfile) -> None:
    st.session_state["profile"] = profile


def profile_header(profile: BatteryProfile, extra: str | None = None) -> None:
    """Silkscreen-style identification line for the active battery."""
    color = CHEM_COLORS.get(profile.chemistry, ACCENT)
    parts = [
        f"{profile.cells_series}s{profile.cells_parallel}p",
        f"{profile.pack_nominal_voltage_V:.1f} V",
        f"{profile.pack_capacity_Ah:.1f} Ah",
        f"{profile.pack_energy_Wh / 1000:.2f} kWh",
        profile.cell.form_factor,
    ]
    if extra:
        parts.append(extra)
    items = "<i>·</i>".join(f"<span>{p}</span>" for p in parts)
    st.markdown(
        f"<div class='readout'><span class='swatch' style='background:{color}'></span>"
        f"<b>{profile.name}</b>{items}</div>",
        unsafe_allow_html=True,
    )


def toggle_rail() -> None:
    st.session_state["sidebar_rail"] = not st.session_state.get("sidebar_rail", False)


def sidebar_profile() -> BatteryProfile:
    """The active battery.

    Navigation and the context block live in the rail (drawn in `app.py`), so pages
    only need the profile itself; nothing is written to Streamlit's own sidebar.
    """
    return ensure_profile()


def status_chip(soh: float) -> str:
    if soh >= 0.9:
        return badge("measured", "Healthy (SOH ≥ 90%)")
    if soh >= 0.8:
        return badge("predicted", "Ageing (80–90% SOH)")
    return badge("simulated", "End of life (< 80% SOH)")


def kpi_row(items: list[tuple[str, str, str | None]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items):
        col.metric(label, value, help=help_text)


def dataset_picker(key_prefix: str = "ds") -> tuple[str, str]:
    table = da.cycle_table()
    datasets = ["NASA", "CALCE", "Oxford", "Samsung"]
    dataset = st.selectbox("Dataset", datasets, key=f"{key_prefix}_dataset")
    cells = sorted(table[table.dataset == dataset]["cell_id"].unique()) if not table.empty else []
    cell = st.selectbox("Battery / cell", cells, key=f"{key_prefix}_cell")
    return dataset, cell


def model_bundle(name: str, fallback: str | None = None):
    available = da.bundles()
    choice = name if name in available else fallback
    if choice is None or choice not in available:
        return None
    return da.load_bundle(choice)


def method_label() -> None:
    st.caption(
        "Colours follow the project convention: "
        "<span class='badge badge-measured'>measured</span> · "
        "<span class='badge badge-simulated'>simulated</span> · "
        "<span class='badge badge-predicted'>ML predicted</span>",
        unsafe_allow_html=True,
    )
