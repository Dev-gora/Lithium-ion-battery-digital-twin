"""Charge protocols for the simulator."""

from __future__ import annotations

import numpy as np

from battery.battery_profiles import BatteryProfile


def constant_current_voltage(
    profile: BatteryProfile,
    c_rate: float = 0.5,
    taper_cutoff_c_rate: float = 0.05,
):
    """CC-CV charge current as a function of (time, terminal voltage).

    Returns a callable ``f(time_s, voltage_V) -> current_A`` (charge is negative,
    following the discharge-positive convention of the ECM).
    """
    cell = profile.cell
    i_cc = c_rate * cell.rated_capacity_Ah
    i_cut = max(taper_cutoff_c_rate * cell.rated_capacity_Ah, 1e-3)

    def current(t_s: float, voltage_V: float) -> float:
        if voltage_V >= cell.max_voltage_V - 1e-3:
            # Voltage-limited taper: emulate CV by reducing current smoothly.
            excess = voltage_V - (cell.max_voltage_V - 1e-3)
            return -max(i_cut, i_cc * float(np.exp(-excess / 0.01)))
        return -i_cc

    return current


def constant_current(profile: BatteryProfile, c_rate: float = 1.0):
    i = c_rate * profile.cell.rated_capacity_Ah
    return lambda t_s, voltage_V: -i


def pulse_charge(profile: BatteryProfile, c_rate: float = 1.0, period_s: float = 600.0, duty: float = 0.7):
    """Pulsed charge: duty-cycled constant current with rest periods."""
    i = c_rate * profile.cell.rated_capacity_Ah

    def current(t_s: float, voltage_V: float) -> float:
        return -i if (t_s % period_s) < duty * period_s else 0.0

    return current
