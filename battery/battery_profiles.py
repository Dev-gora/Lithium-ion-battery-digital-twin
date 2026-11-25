"""Configurable lithium-ion battery profiles.

Every parameter here is a *documented assumption* for a class of device, not a
measurement of a specific commercial cell.  Chemistry-specific OCV shape and
kinetic constants are parametric approximations; a calibrated digital twin for a
particular cell requires measured OCV/entropy data and cell-level impedance.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

Chemistry = Literal["LCO", "LFP", "NMC", "NCA"]


@dataclass
class ThermalParams:
    """Lumped thermal capacitance model of a single cell."""

    mass_kg: float = 0.045
    cp_J_per_kgK: float = 1000.0
    h_area_W_per_K: float = 0.12

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CellParams:
    """Electrical and thermal parameters of one cell."""

    chemistry: Chemistry = "NMC"
    form_factor: str = "18650"
    nominal_voltage_V: float = 3.6
    max_voltage_V: float = 4.2
    min_voltage_V: float = 2.5
    rated_capacity_Ah: float = 3.0
    initial_soc: float = 1.0
    initial_soh: float = 1.0
    r0_ohm: float = 0.045
    r1_ohm: float = 0.020
    c1_farad: float = 1800.0
    ambient_temperature_C: float = 25.0
    thermal: ThermalParams = field(default_factory=ThermalParams)
    max_charge_current_A: float = 3.0
    max_discharge_current_A: float = 6.0

    def to_dict(self) -> dict:
        data = asdict(self)
        data.update({f"thermal_{k}": v for k, v in data.pop("thermal").items()})
        return data


# Nominal chemistry presets.  Values are typical of commercial cells of that
# chemistry, used only as a starting point for simulation.
CHEMISTRY_PRESETS: dict[str, dict] = {
    "LCO": {
        "nominal_voltage_V": 3.7,
        "max_voltage_V": 4.2,
        "min_voltage_V": 3.0,
        "r0_ohm": 0.060,
        "r1_ohm": 0.025,
        "note": "Consumer electronics; Co-rich layered oxide. Thermally least stable.",
    },
    "LFP": {
        "nominal_voltage_V": 3.2,
        "max_voltage_V": 3.65,
        "min_voltage_V": 2.5,
        "r0_ohm": 0.030,
        "r1_ohm": 0.018,
        "note": "Flat OCV plateau, long cycle life, excellent thermal stability.",
    },
    "NMC": {
        "nominal_voltage_V": 3.6,
        "max_voltage_V": 4.2,
        "min_voltage_V": 2.5,
        "r0_ohm": 0.045,
        "r1_ohm": 0.020,
        "note": "EV / ESS layered oxide; Ni-rich variants trade life for energy.",
    },
    "NCA": {
        "nominal_voltage_V": 3.6,
        "max_voltage_V": 4.2,
        "min_voltage_V": 2.5,
        "r0_ohm": 0.050,
        "r1_ohm": 0.022,
        "note": "EV cylindrical cells; high energy, sensitive to high temperature.",
    },
}

# OCV curve shape parameters: normalised SOC at which the cell leaves the
# upper plateau and enters the lower plateau, plus plateau steepness.  Used by
# ``equivalent_circuit.ocv_from_soc``.
OCV_SHAPE: dict[str, dict[str, float]] = {
    "LCO": {"upper": 0.86, "lower": 0.16, "slope": 7.0, "hyst": 0.010},
    "LFP": {"upper": 0.97, "lower": 0.035, "slope": 16.0, "hyst": 0.018},
    "NMC": {"upper": 0.88, "lower": 0.10, "slope": 8.0, "hyst": 0.010},
    "NCA": {"upper": 0.88, "lower": 0.10, "slope": 8.5, "hyst": 0.012},
}


@dataclass
class BatteryProfile:
    """A complete battery (cell or pack) configuration."""

    name: str
    application: str
    chemistry: Chemistry
    cells_series: int
    cells_parallel: int
    cell: CellParams
    description: str = ""

    # ---- derived pack quantities -------------------------------------------------
    @property
    def pack_capacity_Ah(self) -> float:
        return self.cell.rated_capacity_Ah * self.cells_parallel

    @property
    def pack_nominal_voltage_V(self) -> float:
        return self.cell.nominal_voltage_V * self.cells_series

    @property
    def pack_energy_Wh(self) -> float:
        return self.pack_capacity_Ah * self.pack_nominal_voltage_V

    @property
    def pack_r0_ohm(self) -> float:
        return self.cell.r0_ohm * self.cells_series / self.cells_parallel

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "application": self.application,
            "chemistry": self.chemistry,
            "cells_series": self.cells_series,
            "cells_parallel": self.cells_parallel,
            "description": self.description,
            "cell": self.cell.to_dict(),
        }


def _cell(chemistry: Chemistry, **overrides) -> CellParams:
    params = dict(CHEMISTRY_PRESETS[chemistry])
    params.pop("note", None)
    params["chemistry"] = chemistry
    params.update(overrides)
    return CellParams(**params)


def ev_profile() -> BatteryProfile:
    """NMC traction pack: 96s2p of 50 Ah automotive pouch cells (~350 V, 100 Ah)."""
    return BatteryProfile(
        name="EV traction pack (NMC)",
        application="Electric vehicle",
        chemistry="NMC",
        cells_series=96,
        cells_parallel=2,
        description=(
            "96s2p NMC pack, 50 Ah automotive pouch cells.  Pack-level thermal "
            "behaviour is lumped into one cell model for a first-order estimate."
        ),
        cell=_cell(
            "NMC",
            form_factor="pouch",
            rated_capacity_Ah=50.0,
            r0_ohm=0.0009,
            r1_ohm=0.0004,
            c1_farad=45000.0,
            max_charge_current_A=100.0,
            max_discharge_current_A=250.0,
            thermal=ThermalParams(mass_kg=0.85, cp_J_per_kgK=1100.0, h_area_W_per_K=0.9),
        ),
    )


def laptop_profile() -> BatteryProfile:
    """3s2p LCO pack of 4.4 Ah pouch cells (~11.1 V, 8.8 Ah, 98 Wh)."""
    return BatteryProfile(
        name="Laptop pack (LCO, 3s2p)",
        application="Laptop",
        chemistry="LCO",
        cells_series=3,
        cells_parallel=2,
        description="3s2p LCO pouch pack, 4.4 Ah per cell, ~98 Wh.",
        cell=_cell(
            "LCO",
            form_factor="pouch",
            rated_capacity_Ah=4.4,
            r0_ohm=0.030,
            r1_ohm=0.014,
            c1_farad=2500.0,
            max_charge_current_A=4.0,
            max_discharge_current_A=8.0,
            thermal=ThermalParams(mass_kg=0.085, cp_J_per_kgK=1000.0, h_area_W_per_K=0.10),
        ),
    )


def smartphone_profile() -> BatteryProfile:
    """Single LCO pouch cell, 4.0 Ah, 3.85 V nominal."""
    return BatteryProfile(
        name="Smartphone cell (LCO)",
        application="Smartphone",
        chemistry="LCO",
        cells_series=1,
        cells_parallel=1,
        description="Single 4.0 Ah LCO pouch cell, 3.85 V nominal.",
        cell=_cell(
            "LCO",
            form_factor="pouch",
            nominal_voltage_V=3.85,
            rated_capacity_Ah=4.0,
            r0_ohm=0.045,
            r1_ohm=0.020,
            c1_farad=1200.0,
            max_charge_current_A=3.0,
            max_discharge_current_A=4.0,
            thermal=ThermalParams(mass_kg=0.045, cp_J_per_kgK=1000.0, h_area_W_per_K=0.04),
        ),
    )


def ess_profile() -> BatteryProfile:
    """Stationary storage rack: 16s2p LFP, 280 Ah cells (~51.2 V, 560 Ah)."""
    return BatteryProfile(
        name="Stationary storage rack (LFP, 16s2p)",
        application="Energy storage system",
        chemistry="LFP",
        cells_series=16,
        cells_parallel=2,
        description="16s2p LFP prismatic cells, 280 Ah each; ~28.7 kWh nominal.",
        cell=_cell(
            "LFP",
            form_factor="prismatic",
            rated_capacity_Ah=280.0,
            r0_ohm=0.00025,
            r1_ohm=0.00012,
            c1_farad=250000.0,
            max_charge_current_A=280.0,
            max_discharge_current_A=280.0,
            thermal=ThermalParams(mass_kg=5.4, cp_J_per_kgK=1050.0, h_area_W_per_K=2.2),
        ),
    )


def portable_profile() -> BatteryProfile:
    """Portable electronics: single NCA 18650, 3.0 Ah."""
    return BatteryProfile(
        name="Portable electronics cell (NCA)",
        application="Portable electronics",
        chemistry="NCA",
        cells_series=1,
        cells_parallel=1,
        description="Single NCA 18650 cell, 3.0 Ah, used in power tools and packs.",
        cell=_cell(
            "NCA",
            form_factor="18650",
            rated_capacity_Ah=3.0,
            r0_ohm=0.055,
            r1_ohm=0.022,
            c1_farad=1500.0,
            max_charge_current_A=4.0,
            max_discharge_current_A=10.0,
            thermal=ThermalParams(mass_kg=0.047, cp_J_per_kgK=1000.0, h_area_W_per_K=0.05),
        ),
    )


BUILTIN_PROFILES = {
    "EV": ev_profile,
    "Laptop": laptop_profile,
    "Smartphone": smartphone_profile,
    "Energy storage": ess_profile,
    "Portable electronics": portable_profile,
}


def get_profile(application: str) -> BatteryProfile:
    return BUILTIN_PROFILES[application]()


def custom_profile(
    chemistry: Chemistry = "NMC",
    series: int = 1,
    parallel: int = 1,
    capacity_Ah: float = 3.0,
    nominal_voltage_V: float | None = None,
    max_voltage_V: float | None = None,
    min_voltage_V: float | None = None,
    initial_soc: float = 1.0,
    initial_soh: float = 1.0,
    r0_ohm: float | None = None,
    ambient_temperature_C: float = 25.0,
) -> BatteryProfile:
    """Build a user-defined profile, filling unspecified fields from chemistry presets."""
    preset = CHEMISTRY_PRESETS[chemistry]
    cell = _cell(
        chemistry,
        rated_capacity_Ah=capacity_Ah,
        nominal_voltage_V=nominal_voltage_V or preset["nominal_voltage_V"],
        max_voltage_V=max_voltage_V or preset["max_voltage_V"],
        min_voltage_V=min_voltage_V or preset["min_voltage_V"],
        r0_ohm=r0_ohm if r0_ohm is not None else preset["r0_ohm"],
        initial_soc=initial_soc,
        initial_soh=initial_soh,
        ambient_temperature_C=ambient_temperature_C,
    )
    return BatteryProfile(
        name="Custom battery",
        application="Custom",
        chemistry=chemistry,
        cells_series=series,
        cells_parallel=parallel,
        cell=cell,
        description="User-configured battery.",
    )
