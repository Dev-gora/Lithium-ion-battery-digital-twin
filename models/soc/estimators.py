"""Classical SOC estimators: coulomb counting and an extended Kalman filter.

Both are used as physics baselines against the learned SOC models.  The EKF
uses a first-order Thevenin model with an OCV(SOC) lookup built from *training*
cells only, so evaluating it on a held-out cell stays honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def coulomb_count(
    current_A: np.ndarray,
    time_s: np.ndarray,
    capacity_Ah: float,
    initial_soc: float = 1.0,
) -> np.ndarray:
    """Integrate current to obtain SOC (I > 0 discharges)."""
    i = np.asarray(current_A, dtype=float)
    t = np.asarray(time_s, dtype=float)
    if i.size != t.size:
        raise ValueError("current and time must have equal length")
    dt = np.diff(t, prepend=t[0])
    dt = np.where(np.isfinite(dt) & (dt >= 0), dt, 0.0)
    used = np.cumsum(i * dt) / 3600.0
    return np.clip(initial_soc - used / max(capacity_Ah, 1e-9), 0.0, 1.0)


def build_ocv_lookup(
    soc: np.ndarray,
    voltage: np.ndarray,
    bins: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Monotone OCV(SOC) table from (near-)constant-current training data."""
    soc = np.asarray(soc, dtype=float)
    voltage = np.asarray(voltage, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    centres, means = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (soc >= lo) & (soc < hi) & np.isfinite(voltage)
        if mask.sum() >= 5:
            centres.append((lo + hi) / 2)
            means.append(float(np.mean(voltage[mask])))
    centres = np.asarray(centres)
    means = np.asarray(means)
    order = np.argsort(centres)
    centres, means = centres[order], means[order]
    # Enforce monotonicity (isotonic-like running max from empty to full).
    means = np.maximum.accumulate(means)
    return centres, means


@dataclass
class EKFResult:
    soc: np.ndarray
    voltage_estimate: np.ndarray
    innovation: np.ndarray


@dataclass
class SOCEKF:
    """Extended Kalman filter for SOC with a 2-state Thevenin model.

    State x = [soc, v1]; input u = current (A, positive on discharge).
    """

    ocv_soc: np.ndarray
    ocv_v: np.ndarray
    capacity_Ah: float
    r0: float = 0.05
    r1: float = 0.02
    c1: float = 2000.0
    process_soc: float = 1e-8
    process_v1: float = 1e-6
    measurement_var: float = 2.5e-3
    soc: float = 1.0
    v1: float = 0.0
    p: np.ndarray = field(default_factory=lambda: np.diag([1e-3, 1e-4]))

    def ocv(self, soc: float) -> float:
        return float(np.interp(soc, self.ocv_soc, self.ocv_v))

    def docv(self, soc: float) -> float:
        # Finite-difference derivative of the lookup table.
        h = 1e-3
        return (self.ocv(min(soc + h, 1.0)) - self.ocv(max(soc - h, 0.0))) / (2 * h)

    def step(self, current_A: float, voltage_V: float, dt_s: float) -> tuple[float, float]:
        i = float(current_A)
        dt = max(float(dt_s), 1e-9)

        # ---- predict -------------------------------------------------------
        soc_pred = self.soc - i * dt / 3600.0 / max(self.capacity_Ah, 1e-9)
        decay = np.exp(-dt / max(self.r1 * self.c1, 1e-9))
        v1_pred = self.v1 * decay + i * self.r1 * (1.0 - decay)

        f = np.array([[1.0, 0.0], [0.0, decay]])
        q = np.diag([self.process_soc, self.process_v1])
        p_pred = f @ self.p @ f.T + q

        # ---- update --------------------------------------------------------
        h = np.array([self.docv(soc_pred), -1.0])
        v_hat = self.ocv(soc_pred) - i * self.r0 - v1_pred
        innovation = voltage_V - v_hat
        s = float(h @ p_pred @ h.T + self.measurement_var)
        k = p_pred @ h / max(s, 1e-12)
        x = np.array([soc_pred, v1_pred]) + k * innovation
        x[0] = float(np.clip(x[0], 0.0, 1.0))

        self.p = (np.eye(2) - np.outer(k, h)) @ p_pred
        self.soc, self.v1 = float(x[0]), float(x[1])
        return self.soc, innovation

    def run(self, current_A, voltage_V, time_s) -> EKFResult:
        i = np.asarray(current_A, dtype=float)
        v = np.asarray(voltage_V, dtype=float)
        t = np.asarray(time_s, dtype=float)
        socs, vhat, innov = [], [], []
        for k in range(i.size):
            dt = t[k] - t[k - 1] if k else 0.0
            s, e = self.step(i[k], v[k], dt)
            socs.append(s)
            innov.append(e)
            vhat.append(self.ocv(s) - i[k] * self.r0 - self.v1)
        return EKFResult(np.asarray(socs), np.asarray(vhat), np.asarray(innov))
