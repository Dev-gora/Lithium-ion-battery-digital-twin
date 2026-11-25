"""Pack-level series / parallel calculations and imbalance bookkeeping.

Cells are assumed identical except for an optional capacity mismatch vector,
which is used to illustrate how parallel strings share current unevenly.  This
is a design-level estimate, not a substitute for a full pack simulation with
per-cell states.
"""

from __future__ import annotations

import numpy as np

from .battery_profiles import BatteryProfile


class PackModel:
    def __init__(self, profile: BatteryProfile, capacity_mismatch: np.ndarray | None = None):
        self.profile = profile
        s, p = profile.cells_series, profile.cells_parallel
        if capacity_mismatch is None:
            capacity_mismatch = np.ones(p)
        mismatch = np.asarray(capacity_mismatch, dtype=float)
        if mismatch.shape != (p,):
            raise ValueError(f"capacity_mismatch must have {p} entries")
        self.capacity_mismatch = mismatch / mismatch.mean()

    # ---- DC quantities -------------------------------------------------------
    def pack_voltage(self, cell_voltage_V: np.ndarray | float) -> np.ndarray:
        return np.asarray(cell_voltage_V, dtype=float) * self.profile.cells_series

    def pack_capacity_Ah(self) -> float:
        """Effective pack capacity accounting for parallel imbalance."""
        return float(
            self.profile.cell.rated_capacity_Ah
            * self.capacity_mismatch.sum()
            * self.profile.cell.initial_soh
        )

    def pack_energy_Wh(self) -> float:
        return self.pack_capacity_Ah() * self.profile.pack_nominal_voltage_V

    def pack_resistance_ohm(self) -> float:
        return (
            self.profile.cell.r0_ohm
            * self.profile.cells_series
            / self.profile.cells_parallel
        )

    # ---- current sharing -----------------------------------------------------
    def split_current(self, pack_current_A: float) -> np.ndarray:
        """Parallel strings share current inversely with their resistance.

        Resistance is assumed inversely proportional to capacity, so the
        share of string *i* is proportional to its capacity ratio.
        """
        weights = self.capacity_mismatch / np.sum(self.capacity_mismatch)
        return pack_current_A * weights

    def power_at_current(self, pack_current_A: float, cell_soc: float) -> dict:
        from .equivalent_circuit import ocv_voltage

        cell_v = float(ocv_voltage(cell_soc, self.profile.cell))
        pack_v = cell_v * self.profile.cells_series
        i_cell = pack_current_A / self.profile.cells_parallel
        pack_v -= i_cell * self.pack_resistance_ohm()
        return {
            "pack_voltage_V": pack_v,
            "pack_current_A": pack_current_A,
            "pack_power_W": pack_v * pack_current_A,
            "cell_current_A": i_cell,
        }
