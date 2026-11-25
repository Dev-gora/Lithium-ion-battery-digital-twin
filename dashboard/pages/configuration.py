"""Page 2 — battery configuration."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from battery.battery_profiles import CHEMISTRY_PRESETS, custom_profile, get_profile
from dashboard.ui import ensure_profile, profile_header, set_profile, sidebar_profile
from battery.pack_model import PackModel
from dashboard import diagrams
from dashboard.guide import guide


def render() -> None:
    sidebar_profile()
    st.title("Battery configuration")
    st.caption("Profiles describe classes of devices. Parameters are documented assumptions, not cell datasheets.")
    active = ensure_profile()
    guide(
        "FIG. 02",
        "What this page configures",
        "The battery every other page is modelling. Pick an application profile — EV, laptop, "
        "smartphone, stationary storage, portable electronics — or build a pack from cell "
        "parameters. Series cells raise voltage, parallel cells raise capacity, and the derived "
        "table underneath shows the pack that results.",
        [
            "Choose an application, then adjust initial SOC, initial SOH and ambient temperature.",
            "Or switch to Custom and enter the cell: chemistry, series/parallel count, capacity, "
            "voltage window and DC resistance.",
            "Read the derived pack table: capacity, energy, pack resistance and usable capacity at "
            "the current SOH. These values drive the dashboard and simulator.",
        ],
        read="The drawing is the electrical topology, not a product photo: cells in a string share "
             "one current, parallel columns share it between them, and the BMS taps both busbars. "
             "A 4s2p pack therefore has eight cells, twice the voltage and twice the capacity of one.",
        svg=diagrams.pack_topology(active.cells_series, active.cells_parallel),
    )

    application = st.radio(
        "Application",
        ["EV", "Laptop", "Smartphone", "Energy storage", "Portable electronics", "Custom"],
        horizontal=True,
        index=["EV", "Laptop", "Smartphone", "Energy storage", "Portable electronics"].index(
            ensure_profile().application
        ) if ensure_profile().application in ["EV", "Laptop", "Smartphone", "Energy storage", "Portable electronics"] else 0,
    )

    if application != "Custom":
        profile = get_profile(application)
        profile_header(profile)
        st.info(profile.description)

        cols = st.columns(3)
        with cols[0]:
            initial_soc = st.slider("Initial SOC / %", 5, 100, int(profile.cell.initial_soc * 100), 5) / 100
        with cols[1]:
            initial_soh = st.slider("Initial SOH / %", 60, 100, int(profile.cell.initial_soh * 100), 1) / 100
        with cols[2]:
            ambient = st.number_input("Ambient temperature / °C", -10.0, 55.0,
                                      float(profile.cell.ambient_temperature_C), 1.0)
        profile.cell.initial_soc = initial_soc
        profile.cell.initial_soh = initial_soh
        profile.cell.ambient_temperature_C = ambient
        set_profile(profile)
    else:
        chemistry = st.selectbox("Chemistry", list(CHEMISTRY_PRESETS))
        preset = CHEMISTRY_PRESETS[chemistry]
        st.caption(preset["note"])
        cols = st.columns(3)
        with cols[0]:
            series = st.number_input("Cells in series", 1, 400, 1, 1)
            parallel = st.number_input("Cells in parallel", 1, 40, 1, 1)
        with cols[1]:
            capacity = st.number_input("Cell rated capacity / Ah", 0.05, 500.0, 3.0, 0.05)
            nominal = st.number_input("Cell nominal voltage / V", 2.0, 4.3,
                                      float(preset["nominal_voltage_V"]), 0.05)
        with cols[2]:
            vmax = st.number_input("Cell max voltage / V", 2.5, 4.5, float(preset["max_voltage_V"]), 0.05)
            vmin = st.number_input("Cell min voltage / V", 1.5, 3.6, float(preset["min_voltage_V"]), 0.05)
        c1, c2, c3 = st.columns(3)
        with c1:
            soc = st.slider("Initial SOC / %", 5, 100, 80, 5) / 100
        with c2:
            soh = st.slider("Initial SOH / %", 60, 100, 100, 1) / 100
        with c3:
            ambient = st.number_input("Ambient temperature / °C", -10.0, 55.0, 25.0, 1.0)
        r0 = st.number_input("Cell DC resistance / Ω", 0.0001, 1.0, float(preset["r0_ohm"]), 0.0001, format="%.4f")

        profile = custom_profile(
            chemistry=chemistry, series=int(series), parallel=int(parallel), capacity_Ah=capacity,
            nominal_voltage_V=nominal, max_voltage_V=vmax, min_voltage_V=vmin,
            initial_soc=soc, initial_soh=soh, r0_ohm=r0, ambient_temperature_C=ambient,
        )
        set_profile(profile)
        profile_header(profile)

    st.divider()
    st.subheader("Derived pack quantities")
    pack = PackModel(profile)
    st.dataframe(
        pd.DataFrame(
            [
                {"Quantity": "Configuration", "Value": f"{profile.cells_series}s{profile.cells_parallel}p"},
                {"Quantity": "Pack capacity", "Value": f"{pack.pack_capacity_Ah():.2f} Ah"},
                {"Quantity": "Pack nominal voltage", "Value": f"{profile.pack_nominal_voltage_V:.1f} V"},
                {"Quantity": "Pack energy", "Value": f"{pack.pack_energy_Wh()/1000:.2f} kWh"},
                {"Quantity": "Pack DC resistance", "Value": f"{pack.pack_resistance_ohm()*1000:.2f} mΩ"},
                {"Quantity": "Usable capacity at SOH", "Value": f"{profile.pack_capacity_Ah*profile.cell.initial_soh:.2f} Ah"},
                {"Quantity": "Chemistry", "Value": profile.chemistry},
                {"Quantity": "Cell form factor", "Value": profile.cell.form_factor},
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Cell parameters actually used by the models"):
        st.json({
            "cell": profile.cell.to_dict(),
            "thermal_time_constant_s": round(
                profile.cell.thermal.mass_kg * profile.cell.thermal.cp_J_per_kgK
                / max(profile.cell.thermal.h_area_W_per_K, 1e-9), 1),
        })

    st.caption(
        "OCV(SOC) is a chemistry-level parametric curve, not a measured pseudo-OCV. "
        "A cell-specific twin needs measured OCV and impedance data; the dataset audit "
        "records which sources provide them."
    )
