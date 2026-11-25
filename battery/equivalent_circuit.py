"""Thevenin equivalent-circuit cell model.

State: SOC [-] and RC polarisation voltage ``v1`` [V].  Terminal voltage is

    V = OCV(SOC) - I * R0(SOC, T, SOH) - v1,        dv1/dt = -v1/(R1 C1) + I/C1

with ``I > 0`` on discharge.  ``OCV(SOC)`` is a parametric approximation per
chemistry (see ``battery_profiles.OCV_SHAPE``).  It is *not* a measured GITT
curve; fitted cells should supply their own OCV table via ``ocv_table``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .battery_profiles import OCV_SHAPE, CellParams

R_GAS = 8.314


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def ocv_from_soc(soc, chemistry: str) -> np.ndarray:
    """Parametric open-circuit voltage for a chemistry.

    Shape: a high plateau near full charge, a sloped middle region and a steep
    fall near empty, which reproduces the qualitative behaviour of the four
    commercial chemistries without claiming cell-specific accuracy.
    """
    soc = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
    shape = OCV_SHAPE.get(chemistry, OCV_SHAPE["NMC"])
    upper, lower, slope = shape["upper"], shape["lower"], shape["slope"]
    # Normalise the plateau position to a 0-1 scale between the two knees.
    x = (soc - lower) / max(upper - lower, 1e-6)
    base = 0.55 + 0.45 * np.clip(x, 0.0, 1.0)
    # Plateau edges.
    base += 0.08 * _logistic(slope * (soc - lower))
    base -= 0.10 * _logistic(slope * (soc - upper))
    # Ionic resistance near empty state of charge.
    base -= 0.05 * np.exp(-soc / 0.02)
    return np.clip(base, 0.0, 1.0)


def ocv_voltage(soc, params: CellParams) -> np.ndarray:
    """Map normalised OCV shape onto the cell's voltage window."""
    frac = ocv_from_soc(soc, params.chemistry)
    return params.min_voltage_V + frac * (params.max_voltage_V - params.min_voltage_V)


def resistance(params: CellParams, soc: float, temperature_C: float, soh: float) -> float:
    """Ohmic resistance with Arrhenius temperature and SOH dependence.

    ``R0`` grows as ``(1/SOH)^k`` (k = 1.6), a common phenomenological form, and
    doubles roughly every 25 K colder than 25 C.
    """
    soh = max(float(soh), 0.2)
    arrhenius = np.exp(3500.0 * (1.0 / (temperature_C + 273.15) - 1.0 / 298.15))
    soc_factor = 1.0 + 0.6 * np.exp(-max(soc, 1e-4) / 0.05)
    return params.r0_ohm * (1.0 / soh) ** 1.6 * arrhenius * soc_factor


@dataclass
class ECMState:
    soc: float
    v1: float = 0.0
    temperature_C: float = 25.0

    def copy(self) -> "ECMState":
        return ECMState(self.soc, self.v1, self.temperature_C)


class TheveninModel:
    """Single-RC Thevenin model with a state that persists across steps."""

    def __init__(self, params: CellParams, soh: float | None = None):
        self.params = params
        self.soh = float(soh if soh is not None else params.initial_soh)
        self.state = ECMState(
            soc=float(np.clip(params.initial_soc, 0.0, 1.0)),
            v1=0.0,
            temperature_C=params.ambient_temperature_C,
        )
        self.usable_capacity_Ah = max(params.rated_capacity_Ah * self.soh, 1e-6)

    # ---- observables ---------------------------------------------------------
    def ocv(self, soc: float | None = None) -> float:
        return float(ocv_voltage(soc if soc is not None else self.state.soc, self.params))

    def r0(self) -> float:
        return resistance(self.params, self.state.soc, self.state.temperature_C, self.soh)

    def terminal_voltage(self, current_A: float) -> float:
        """Terminal voltage for the current state (I > 0 = discharge)."""
        return self.ocv() - current_A * self.r0() - self.state.v1

    # ---- integration ---------------------------------------------------------
    def step(self, current_A: float, dt_s: float, temperature_C: float | None = None) -> dict:
        """Advance the state by ``dt_s`` seconds under constant current."""
        if temperature_C is not None:
            self.state.temperature_C = float(temperature_C)
        i = float(current_A)
        dt = max(float(dt_s), 1e-9)

        # Charge throughput (Ah) and SOC update.
        d_ah = i * dt / 3600.0
        self.state.soc = float(
            np.clip(self.state.soc - d_ah / self.usable_capacity_Ah, 0.0, 1.0)
        )

        # RC polarisation, exact discretisation of the first-order ODE.
        r1 = self.params.r1_ohm * (1.0 / max(self.soh, 0.2)) ** 1.2
        c1 = max(self.params.c1_farad, 1e-9)
        tau = max(r1 * c1, 1e-9)
        decay = np.exp(-dt / tau)
        self.state.v1 = self.state.v1 * decay + i * r1 * (1.0 - decay)

        v = self.terminal_voltage(i)
        return {
            "voltage_V": float(v),
            "soc": self.state.soc,
            "v1_V": float(self.state.v1),
            "temperature_C": self.state.temperature_C,
            "power_W": float(v * i),
            "r0_ohm": float(self.r0()),
            "current_A": i,
        }
