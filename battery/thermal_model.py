"""Lumped-parameter thermal model of a cell.

    m * cp * dT/dt = I^2 * R_internal + h*A*(T_amb - T)

This is a first-order model: it captures the self-heating trend and the
approach to ambient, not internal temperature gradients.  Cell-level thermal
runaway is not modelled.
"""

from __future__ import annotations

from dataclasses import dataclass

from .battery_profiles import CellParams


@dataclass
class ThermalState:
    temperature_C: float
    heat_generated_J: float = 0.0


class LumpedThermalModel:
    def __init__(self, params: CellParams, ambient_C: float | None = None):
        self.params = params
        self.ambient_C = float(
            ambient_C if ambient_C is not None else params.ambient_temperature_C
        )
        self.state = ThermalState(temperature_C=self.ambient_C)

    def step(self, current_A: float, internal_resistance_ohm: float, dt_s: float) -> float:
        """Advance cell temperature by ``dt_s`` seconds."""
        thermal = self.params.thermal
        heat = current_A**2 * internal_resistance_ohm * dt_s
        loss = thermal.h_area_W_per_K * (self.state.temperature_C - self.ambient_C) * dt_s
        delta = (heat - loss) / max(thermal.mass_kg * thermal.cp_J_per_kgK, 1e-9)
        self.state.temperature_C += delta
        self.state.heat_generated_J += heat
        return self.state.temperature_C

    def time_constant_s(self) -> float:
        thermal = self.params.thermal
        return (
            thermal.mass_kg * thermal.cp_J_per_kgK / max(thermal.h_area_W_per_K, 1e-9)
        )
