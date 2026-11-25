"""Ready-made simulator scenarios used by the dashboard."""

from __future__ import annotations

import numpy as np

from battery.battery_profiles import BatteryProfile
from simulator.engine import Simulator, pack_scale

from .charging import constant_current_voltage, pulse_charge
from .discharging import constant_load, constant_power, drive_cycle, variable_load


def build_simulator(profile: BatteryProfile, dt_s: float = 10.0, **kwargs) -> Simulator:
    return Simulator(profile, dt_s=dt_s, **kwargs)


def run_charge(profile, c_rate: float = 0.5, duration_s: float = 7200.0, dt_s: float = 10.0):
    # A charge simulation starts from a depleted cell unless configured otherwise.
    if profile.cell.initial_soc > 0.95:
        profile.cell.initial_soc = 0.05
    sim = build_simulator(profile, dt_s=dt_s)
    protocol = constant_current_voltage(profile, c_rate=c_rate)
    # Wrap the (time, voltage) protocol into the engine's (time) signature.
    def fn(t_s: float) -> float:
        v = sim.ecm.terminal_voltage(0.0)
        return protocol(t_s, v)
    result = sim.run(fn, duration_s)
    result.frame = pack_scale(result.frame, profile)
    return result


def run_discharge(
    profile,
    mode: str = "Constant C-rate",
    c_rate: float = 1.0,
    power_W: float = 10.0,
    duration_s: float = 7200.0,
    dt_s: float = 10.0,
):
    sim = build_simulator(profile, dt_s=dt_s)
    if mode == "Constant power":
        # The caller passes pack power; the cell model needs the per-cell share.
        cell_power = power_W / max(profile.cells_series * profile.cells_parallel, 1)
        base = constant_power(profile, cell_power)
        fn = lambda t_s: base(t_s, sim.ecm.terminal_voltage(0.0))
    elif mode == "Variable load":
        fn = variable_load(profile, c_rate)
    elif mode == "Drive cycle":
        fn = drive_cycle(profile)
    else:
        fn = constant_load(profile, c_rate)
    result = sim.run(lambda t_s: float(fn(t_s)), duration_s)
    result.frame = pack_scale(result.frame, profile)
    return result


def run_cycles(profile, cycles: int = 50, c_rate: float = 1.0, dt_s: float = 30.0):
    """Charge/discharge cycling to expose simulated capacity fade."""
    sim = build_simulator(profile, dt_s=dt_s)
    cell = profile.cell
    rows = []
    for cycle in range(cycles):
        # Discharge to the lower cut-off.
        guard = 0
        while sim.ecm.state.soc > 0.02 and guard < 20000:
            sim.step(c_rate * cell.rated_capacity_Ah)
            guard += 1
        # Recharge to the upper cut-off.
        guard = 0
        while sim.ecm.state.soc < 0.995 and guard < 20000:
            i = -min(c_rate * cell.rated_capacity_Ah, cell.max_charge_current_A)
            sim.step(i)
            guard += 1
        rows.append(
            {
                "cycle": cycle + 1,
                "soh": sim.ecm.soh,
                "capacity_Ah": sim.ecm.usable_capacity_Ah,
                "temperature_C": sim.thermal.state.temperature_C,
            }
        )
    import pandas as pd

    return pd.DataFrame(rows)
