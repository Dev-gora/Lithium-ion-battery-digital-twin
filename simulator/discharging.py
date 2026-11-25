"""Discharge load profiles for the simulator."""

from __future__ import annotations

import numpy as np

from battery.battery_profiles import BatteryProfile


def constant_load(profile: BatteryProfile, c_rate: float = 1.0):
    i = c_rate * profile.cell.rated_capacity_Ah
    return lambda t_s: i


def constant_power(profile: BatteryProfile, power_W: float):
    """Constant-power load: current depends on the present terminal voltage."""
    def current(t_s: float, voltage_V: float | None = None) -> float:
        if not voltage_V or voltage_V <= 0:
            return power_W / profile.pack_nominal_voltage_V
        return power_W / voltage_V
    return current


def variable_load(profile: BatteryProfile, base_c_rate: float = 0.5, amplitude: float = 0.3, period_s: float = 1800.0):
    i = base_c_rate * profile.cell.rated_capacity_Ah
    return lambda t_s: max(i * (1.0 + amplitude * np.sin(2 * np.pi * t_s / period_s)), 0.0)


def drive_cycle(
    profile: BatteryProfile,
    cycle: str = "urban",
    peak_c_rate: float = 1.5,
    seed: int = 7,
):
    """Synthetic duty cycle with idle, cruise and acceleration phases.

    This is a *synthetic* profile, not a measured Artemis/WLTP trace; the
    Oxford dataset provides the measured Artemis example used for verification.
    """
    rng = np.random.default_rng(seed)
    cells = profile.chemistry
    # Shape parameters differ slightly by application to keep the profile plausible.
    cruise = {"LFP": 0.4, "LCO": 0.6, "NMC": 0.55, "NCA": 0.65}.get(cells, 0.5)
    if cycle == "highway":
        cruise *= 1.6
        peak_c_rate *= 1.3
    capacity = profile.cell.rated_capacity_Ah

    # Piecewise phases over a 20-minute block.
    def current(t_s: float) -> float:
        phase = (t_s % 1200.0) / 1200.0
        if phase < 0.10:          # idle
            return 0.0
        if phase < 0.55:          # cruise
            return cruise * capacity
        if phase < 0.85:          # accelerations
            return peak_c_rate * capacity * (0.5 + 0.5 * abs(np.sin(6.0 * t_s)))
        return 0.15 * capacity + 0.05 * capacity * rng.random()

    return current
