"""Time-stepping simulation engine for a single cell (or a pack of identical cells).

Integrates the Thevenin circuit, lumped thermal model and empirical ageing model
together and returns a tidy DataFrame.  All outputs are *simulated*, which the
dashboard labels explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from battery.battery_profiles import BatteryProfile, CellParams
from battery.degradation import EmpiricalDegradationModel
from battery.equivalent_circuit import TheveninModel
from battery.thermal_model import LumpedThermalModel


@dataclass
class SimulationResult:
    frame: pd.DataFrame
    profile: BatteryProfile

    @property
    def final(self) -> dict:
        row = self.frame.iloc[-1]
        return row.to_dict()


class Simulator:
    """Stateful battery digital-twin simulator."""

    def __init__(
        self,
        profile: BatteryProfile,
        dt_s: float = 10.0,
        ageing: bool = True,
        initial_soc: float | None = None,
        ambient_temperature_C: float | None = None,
    ):
        self.profile = profile
        cell: CellParams = profile.cell
        if initial_soc is not None:
            cell.initial_soc = float(np.clip(initial_soc, 0.0, 1.0))
        if ambient_temperature_C is not None:
            cell.ambient_temperature_C = float(ambient_temperature_C)
        self.dt_s = float(dt_s)
        self.ecm = TheveninModel(cell)
        self.thermal = LumpedThermalModel(cell, cell.ambient_temperature_C)
        self.ageing = EmpiricalDegradationModel(
            cell.chemistry, cell.rated_capacity_Ah, cell.initial_soh, cell.ambient_temperature_C
        ) if ageing else None
        self.t = 0.0
        self.rows: list[dict] = []

    # ---- single step ---------------------------------------------------------
    def step(self, current_A: float, dt_s: float | None = None) -> dict:
        dt = float(dt_s if dt_s is not None else self.dt_s)
        out = self.ecm.step(current_A, dt, self.thermal.state.temperature_C)
        # Thermal update uses the ohmic loss at the new state.
        temp = self.thermal.step(current_A, out["r0_ohm"], dt)

        if self.ageing is not None:
            deg = self.ageing.accelerate(dt, current_A, temp)
            self.ecm.soh = deg.soh
            self.ecm.usable_capacity_Ah = max(
                self.profile.cell.rated_capacity_Ah * deg.soh, 1e-6
            )

        self.t += dt
        row = {
            "time_s": self.t,
            "time_min": self.t / 60.0,
            "current_A": out["current_A"],
            "voltage_V": out["voltage_V"],
            "soc": out["soc"],
            "temperature_C": temp,
            "power_W": out["power_W"],
            "r0_ohm": out["r0_ohm"],
            "soh": self.ecm.soh,
            "capacity_Ah": self.ecm.usable_capacity_Ah,
        }
        self.rows.append(row)
        return row

    # ---- voltage-limited operation ------------------------------------------
    def allowed_current(self, requested_A: float) -> float:
        """Clip current so the terminal voltage stays inside the cell window."""
        cell = self.profile.cell
        margin = 0.005 * (cell.max_voltage_V - cell.min_voltage_V)
        # Stop at the SOC window edges as well as the voltage limits.
        if requested_A > 0 and self.ecm.state.soc <= 0.002:
            return 0.0
        if requested_A < 0 and self.ecm.state.soc >= 0.998:
            return 0.0
        v = self.ecm.terminal_voltage(requested_A)
        if requested_A > 0 and v < cell.min_voltage_V + margin:
            return 0.0
        if requested_A < 0 and v > cell.max_voltage_V - margin:
            return 0.0
        return float(requested_A)

    def run(
        self,
        current_fn: Callable[[float], float],
        duration_s: float,
        stop_on_voltage: bool = True,
    ) -> SimulationResult:
        n = int(duration_s / self.dt_s) + 1
        for _ in range(n):
            i = current_fn(self.t)
            if stop_on_voltage:
                i = self.allowed_current(i)
            self.step(i)
            if stop_on_voltage and i == 0.0:
                # Terminal cut-off reached; stop the sweep here.
                break
        frame = pd.DataFrame(self.rows)
        return SimulationResult(frame=frame, profile=self.profile)


def current_from_c_rate(c_rate: float, capacity_Ah: float) -> float:
    return float(c_rate) * float(capacity_Ah)


def pack_scale(frame: pd.DataFrame, profile: BatteryProfile) -> pd.DataFrame:
    """Add pack-level columns to a single-cell simulation."""
    out = frame.copy()
    out["pack_voltage_V"] = out["voltage_V"] * profile.cells_series
    out["pack_current_A"] = out["current_A"] * profile.cells_parallel
    out["pack_power_W"] = out["pack_voltage_V"] * out["pack_current_A"]
    out["pack_energy_Wh"] = profile.pack_energy_Wh
    return out
