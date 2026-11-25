"""Live numbers for the assistant's worked examples.

Every figure the assistant quotes is read here from the same artifacts the pages
use — `metrics.json`, the fold tables, the cycle table and the active profile —
so an example cannot drift from what the app is showing.

Nothing is ever shipped half-filled: :func:`fill` raises :class:`PlaceholderError`
when a placeholder has no live value, when an artifact is missing (the key
resolves to ``n/a``) or when the template cannot be formatted.  Callers decide how
loud to be; the assistant page renders the failure as a red error block.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import streamlit as st

from battery.battery_profiles import get_profile
from battery.pack_model import PackModel
from dashboard import data_access as da

NA = "n/a"

PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


class PlaceholderError(RuntimeError):
    """A knowledge-base template could not be resolved from live artifacts."""


def values() -> dict[str, str]:
    """Formatted strings for `str.format_map` in the knowledge base."""
    return dict(_collect())


@st.cache_data(show_spinner=False)
def _collect() -> dict[str, str]:
    v: dict[str, str] = {}

    try:
        table = da.cycle_table()
    except Exception:
        table = pd.DataFrame()

    if not table.empty:
        v["cells"] = f"{table.cell_id.nunique()}"
        v["rows"] = f"{len(table):,}"

    metrics = da.metrics()

    # ---------------------------------------------------------------- SOC
    soc = metrics.get("soc", {}).get("summary", {})
    if soc:
        best = min(soc.items(), key=lambda kv: kv[1].get("mae_mean", 1e9))
        v["soc_best"] = best[0].split(" (")[0]
        v["soc_mae"] = f"{best[1]['mae_mean'] * 100:.1f} %"
        v["soc_mae_mah"] = f"{best[1]['mae_mean'] * 2.0 * 1000:.0f} mAh"
        for key, label in (("EKF (physics)", "ekf_mae"),
                           ("Coulomb counting (2.0 Ah nominal)", "coulomb_mae")):
            if key in soc:
                v[label] = f"{soc[key]['mae_mean'] * 100:.1f} %"

    soc_folds = da.fold_table("soc_folds.csv")
    if not soc_folds.empty:
        rf = soc_folds[soc_folds.model.astype(str).str.contains("Random Forest")]
        rf = rf.sort_values("mae")
        if not rf.empty:
            v["soc_fold_spread"] = ", ".join(f"{x * 100:.1f} %" for x in rf.mae)
            v["soc_fold_range"] = f"{rf.mae.min() * 100:.1f}–{rf.mae.max() * 100:.1f} %"

    # ---------------------------------------------------------------- SOH
    for kind, prefix in (("capacity_aware", "soh_cap"), ("sensor_only", "soh_sens")):
        summary = metrics.get("soh", {}).get(kind, {}).get("summary", {})
        if not summary:
            continue
        best = min(summary.items(), key=lambda kv: kv[1].get("mae_mean", 1e9))
        v[f"{prefix}_mae"] = f"{best[1]['mae_mean'] * 100:.1f} pp"
        v[f"{prefix}_model"] = best[0].split(" (")[0]

    # ---------------------------------------------------------------- RUL
    rul = metrics.get("rul", {}).get("summary", {})
    if rul:
        best = min(rul.items(), key=lambda kv: kv[1].get("mae_mean", 1e9))
        v["rul_best"] = best[0].split(" (")[0]
        v["rul_best_mae"] = f"{best[1]['mae_mean']:.0f} cycles"

    # ------------------------------------------------ one concrete cell
    if not table.empty:
        nasa = table[table.dataset == "NASA"] if "NASA" in set(table.dataset) else table
        cell_id = "B0005" if (nasa.cell_id == "B0005").any() else str(nasa.cell_id.iloc[0])
        dataset = "NASA" if "NASA" in set(table.dataset) else str(nasa.dataset.iloc[0])
        rows = (table[(table.dataset == dataset) & (table.cell_id == cell_id)]
                .sort_values("cycle_index"))
        # Fold tables key held-out cells as "NASA/B0005".
        label = f"{dataset}/{cell_id}"
        v["cell_label"] = label
        v["cell_ah_rated"] = "2.0 Ah"

        if not rows.empty:
            start = rows.iloc[0]
            v["cell_cap_start"] = f"{start.capacity_Ah:.3f} Ah"
            targeted = rows.iloc[(rows.cycle_index - 100).abs().argsort()[:1]]
            probe = targeted.iloc[0]
            v["cell_cycle"] = f"{int(probe.cycle_index)}"
            v["cell_soh"] = f"{probe.soh * 100:.2f} %"
            v["cell_cap_now"] = f"{probe.capacity_Ah:.3f} Ah"

            bundle = da.model_bundle("soh_model_capacity_aware.joblib")
            if bundle is not None:
                pred = float(da.predict_soh(bundle, targeted)[0])
                v["cell_pred"] = f"{pred * 100:.2f} %"
                v["cell_err"] = f"{abs(pred - float(probe.soh)) * 100:.2f} pp"

            folds = da.fold_table("soh_folds_capacity_aware.csv")
            if not folds.empty:
                hit = folds[(folds.held_out == label)
                            & folds.model.astype(str).str.contains("Random Forest")]
                if not hit.empty:
                    v["cell_fold_mae"] = f"{hit.mae.iloc[0] * 100:.1f} pp"
                    v["cell_fold_r2"] = f"{hit.r2.iloc[0]:.2f}"

            # trajectory-level ageing facts
            soh = rows.soh.to_numpy(float)
            cycle = rows.cycle_index.to_numpy(float)
            if len(rows) > 3:
                slope = np.polyfit(cycle, soh, 1)[0] * 100
                v["age_fade"] = f"{abs(slope):.3f} pp/cycle"
            below = rows[rows.soh <= 0.8]
            if not below.empty:
                v["age_eol"] = f"{int(below.cycle_index.iloc[0])}"
            v["age_n"] = f"{len(rows)}"
            v["age_end_soh"] = f"{rows.soh.iloc[-1] * 100:.1f} %"
            v["age_cap_end"] = f"{rows.capacity_Ah.iloc[-1]:.3f} Ah"
            v["age_last_cycle"] = f"{int(rows.cycle_index.iloc[-1])}"

            # RUL fold for the same cell, if it was scored
            rul_folds = da.fold_table("rul_folds.csv")
            if not rul_folds.empty:
                hit = rul_folds[(rul_folds.held_out == label)
                                & rul_folds.model.astype(str).str.contains("Random Forest")]
                if not hit.empty:
                    v["rul_cell_mae"] = f"{hit.mae.iloc[0]:.0f}"
                    v["rul_cell_n"] = f"{int(hit.n.iloc[0])}"

    # ---------------------------------------------------- active EV profile
    try:
        profile = get_profile("EV")
        pack = PackModel(profile)
        resistance = pack.pack_resistance_ohm()
        v["p_s"] = f"{profile.cells_series}"
        v["p_p"] = f"{profile.cells_parallel}"
        v["p_cells"] = f"{profile.cells_series * profile.cells_parallel}"
        v["p_v"] = f"{profile.pack_nominal_voltage_V:.1f} V"
        v["p_ah"] = f"{profile.pack_capacity_Ah:.0f} Ah"
        v["p_kwh"] = f"{profile.pack_energy_Wh / 1000:.2f} kWh"
        v["p_r"] = f"{resistance * 1000:.1f} mΩ"
        v["cell_v"] = f"{profile.cell.nominal_voltage_V:.2f} V"
        cell_ah = float(getattr(profile.cell, "rated_capacity_Ah", 0.0))
        v["cell_ah"] = f"{cell_ah:.0f} Ah"
        v["cell_r0"] = f"{profile.cell.r0_ohm * 1000:.1f} mΩ"
        v["sim_current"] = f"{0.5 * profile.pack_capacity_Ah:.0f} A"
        v["sim_sag"] = f"{0.5 * profile.pack_capacity_Ah * resistance:.2f} V"
        v["sim_cell_sag"] = f"{0.5 * cell_ah * profile.cell.r0_ohm * 1000:.1f} mV"
    except Exception:
        pass

    for key in KEYS:
        v.setdefault(key, NA)
    return v


KEYS = (
    "cells", "rows", "soc_best", "soc_mae", "soc_mae_mah", "ekf_mae", "coulomb_mae",
    "soc_fold_spread", "soc_fold_range",
    "soh_cap_model", "soh_cap_mae", "soh_sens_model", "soh_sens_mae",
    "rul_best", "rul_best_mae",
    "cell_label", "cell_ah_rated", "cell_cycle", "cell_soh", "cell_cap_start",
    "cell_cap_now", "cell_pred", "cell_err", "cell_fold_mae", "cell_fold_r2",
    "age_fade", "age_eol", "age_n", "age_end_soh", "age_cap_end", "age_last_cycle",
    "rul_cell_mae", "rul_cell_n",
    "p_s", "p_p", "p_cells", "p_v", "p_ah", "p_kwh", "p_r",
    "cell_v", "cell_ah", "cell_r0", "sim_current", "sim_sag", "sim_cell_sag",
)


def missing(text: str, values_map: dict[str, str] | None = None) -> list[str]:
    """Placeholders in `text` that have no usable live value."""
    values_map = values() if values_map is None else values_map
    return sorted({name for name in PLACEHOLDER.findall(text or "")
                   if name not in values_map or values_map[name] == NA})


def fill(text: str) -> str:
    """Substitute live values, raising :class:`PlaceholderError` on any gap."""
    if not text:
        return text
    values_map = values()
    unresolved = missing(text, values_map)
    if unresolved:
        raise PlaceholderError(
            "no live value for " + ", ".join("{" + name + "}" for name in unresolved)
        )
    try:
        return text.format_map(values_map)
    except (KeyError, IndexError, ValueError) as exc:
        raise PlaceholderError(f"template could not be formatted: {exc}") from exc
