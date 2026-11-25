"""Empirical ageing model for capacity fade and resistance growth.

    capacity_loss(%) = k_cal * sqrt(t_days) + k_cyc * (Ah throughput / Q_rated)
    resistance_growth(%) = g_cal * sqrt(t_days) + g_cyc * (Ah throughput / Q_rated)

Coefficients are order-of-magnitude literature values for the four chemistries
and are **not** fitted to a particular cell.  The measured audit trajectories
(NASA, CALCE, Oxford, Samsung) are what the ML SOH/RUL models are trained on;
this module only drives the interactive simulator, and the app labels its
output as simulated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# per chemistry: (calendar %/sqrt(day), cycle % per rated-Ah throughput)
CAPACITY_FADE_COEFF = {
    "LCO": (0.030, 0.0220),
    "LFP": (0.012, 0.0065),
    "NMC": (0.020, 0.0130),
    "NCA": (0.024, 0.0160),
}
# Resistance growth is faster than capacity fade and temperature sensitive.
RESISTANCE_GROWTH_COEFF = {
    "LCO": (0.060, 0.0400),
    "LFP": (0.024, 0.0130),
    "NMC": (0.045, 0.0260),
    "NCA": (0.050, 0.0300),
}
# Arrhenius acceleration relative to 25 C (nominal activation energy 30 kJ/mol).
EFFECTIVE_EA_J = 30000.0


def temperature_acceleration(temperature_C: float) -> float:
    t = float(np.clip(temperature_C, -20.0, 70.0)) + 273.15
    return float(np.exp(EFFECTIVE_EA_J / 8.314 * (1.0 / 298.15 - 1.0 / t)))


@dataclass
class DegradationState:
    soh: float = 1.0
    resistance_growth: float = 0.0
    throughput_Ah: float = 0.0
    elapsed_days: float = 0.0

    def copy(self) -> "DegradationState":
        return DegradationState(self.soh, self.resistance_growth, self.throughput_Ah, self.elapsed_days)


class EmpiricalDegradationModel:
    """Time/throughput ageing with optional temperature acceleration."""

    def __init__(
        self,
        chemistry: str,
        rated_capacity_Ah: float,
        initial_soh: float = 1.0,
        temperature_C: float = 25.0,
    ):
        self.chemistry = chemistry
        self.rated_capacity_Ah = rated_capacity_Ah
        self.state = DegradationState(soh=float(initial_soh))
        self.temperature_C = float(temperature_C)

    def accelerate(self, dt_s: float, current_A: float, temperature_C: float | None = None) -> DegradationState:
        """Advance ageing by ``dt_s`` seconds at ``current_A``."""
        if temperature_C is not None:
            self.temperature_C = float(temperature_C)
        dt_days = dt_s / 86400.0
        throughput = abs(current_A) * dt_s / 3600.0
        accel = temperature_acceleration(self.temperature_C)

        k_cal, k_cyc = CAPACITY_FADE_COEFF.get(self.chemistry, CAPACITY_FADE_COEFF["NMC"])
        g_cal, g_cyc = RESISTANCE_GROWTH_COEFF.get(self.chemistry, RESISTANCE_GROWTH_COEFF["NMC"])

        self.state.elapsed_days += dt_days
        self.state.throughput_Ah += throughput

        # Evaluate the fade law incrementally so the state is path dependent.
        loss = (
            k_cal * accel * (self.state.elapsed_days**0.5 - max(self.state.elapsed_days - dt_days, 0.0) ** 0.5)
            + k_cyc * accel * throughput / max(self.rated_capacity_Ah, 1e-9)
        )
        growth = (
            g_cal * accel * (self.state.elapsed_days**0.5 - max(self.state.elapsed_days - dt_days, 0.0) ** 0.5)
            + g_cyc * accel * throughput / max(self.rated_capacity_Ah, 1e-9)
        )
        self.state.soh = float(max(self.state.soh - loss / 100.0, 0.05))
        self.state.resistance_growth += growth / 100.0
        return self.state

    def resistance_multiplier(self) -> float:
        return 1.0 + self.state.resistance_growth

    def remainder_cycles_to_eol(self, cycles_per_day: float, eol_soh: float = 0.8) -> float | None:
        """Very rough projection: cycles until SOH reaches ``eol_soh``.

        Assumes the recent cycle rate (cycles/day) and the current chemistry
        coefficients continue unchanged.  Returns ``None`` if already past EOL.
        """
        if self.state.soh <= eol_soh:
            return 0.0
        if cycles_per_day <= 0:
            return None
        k_cal, k_cyc = CAPACITY_FADE_COEFF.get(self.chemistry, CAPACITY_FADE_COEFF["NMC"])
        accel = temperature_acceleration(self.temperature_C)
        # Solve fade(t) = 100*(soh - eol) for days by fixed-point iteration.
        target = 100.0 * (self.state.soh - eol_soh)
        days = self.state.elapsed_days
        throughput_per_day = cycles_per_day * self.rated_capacity_Ah * 0.8
        for _ in range(60):
            f = k_cal * accel * np.sqrt(max(days, 1e-6)) + k_cyc * accel * throughput_per_day * days / self.rated_capacity_Ah
            deriv = (
                0.5 * k_cal * accel / np.sqrt(max(days, 1e-6))
                + k_cyc * accel * throughput_per_day / self.rated_capacity_Ah
            )
            if deriv <= 0:
                break
            days = max(days - (f - target) / deriv, 1e-6)
        # Subtract already-elapsed time, convert to cycles.
        return float(max(days - self.state.elapsed_days, 0.0) * cycles_per_day)
