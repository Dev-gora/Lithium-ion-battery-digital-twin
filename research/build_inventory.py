"""Write ``data/reports/dataset_inventory.md`` from the audit outputs.

This replaces only the inventory; the quality report, compatibility matrix and
preprocessing summary are maintained alongside it and must not be overwritten
by a report run.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT = PROJECT_ROOT / "data" / "processed" / "audit"
REPORTS = PROJECT_ROOT / "data" / "reports"
RAW = PROJECT_ROOT / "data" / "raw"


def load(name: str) -> dict:
    """Read ``<name>_summary.json`` (or the given name if it already is one)."""
    stem = name if name.endswith("_summary") else f"{name}_summary"
    path = AUDIT / f"{stem}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def f(value, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}{suffix}"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if not np.isfinite(value) else f"{value:,.{digits}f}{suffix}"


def size(path: Path) -> str:
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return f"{total/1e9:.2f} GB" if total > 1e9 else f"{total/1e6:.0f} MB"


def main() -> None:
    nasa = load("nasa")
    calce = load("calce")
    oxford = load("oxford")
    energy = load("oxford_energy")
    samsung = load("samsung")
    mit = load("mit_summary")
    mit_agg = mit.get("aggregate", {})
    stage1 = load("stage1_summary")

    cycles = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "ml" / "cycle_features.csv") \
        if (PROJECT_ROOT / "data" / "processed" / "ml" / "cycle_features.csv").exists() else pd.DataFrame()

    lines = [
        "# Dataset inventory",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} from the audit outputs in `data/processed/audit/`.",
        "Raw provider files are unchanged; extraction writes only to `data/raw/<source>/extracted/`.",
        "",
        "## At a glance",
        "",
        "| Dataset | Raw size | Cells | Cycles / records | Chemistry | Sampling | SOH labels | RUL labels |",
        "|---|---|---|---|---|---|---|---|",
        f"| NASA PCoE | {size(RAW/'NASA')} | 4 audited (34 in archive) | {f(nasa.get('operations', {}).get('discharge'))} discharges | LCO, 2 Ah 18650 | "
        f"18.69 s (B0005–B7), 9.39 s (B0018) | measured capacity | 4/4 cross 80% |",
        f"| CALCE CS2 | {size(RAW/'CALCE')} | 4 | {f(calce.get('measurement_rows_unique'))} rows / "
        f"{sum(v['cycles_total'] for v in calce.get('cells', {}).values())} cycle records | LCO + trace Mn prismatic | "
        f"30.02 s (CS2_35/36/24), 10.02 s (CS2_3) | measured capacity | 4/4 cross 80% |",
        f"| Oxford Degradation 1 | {size(RAW/'Oxford')} | {f(oxford.get('cells'))} | check-ups every 100 drive cycles | NMC, 740 mAh pouch | "
        f"{f(oxford.get('median_checkup_sampling_s'))} s | pseudo-OCV capacity | "
        f"{sum(1 for v in oxford.get('soh_final_pct', {}).values() if v < 80):d}/8 cross 80% |",
        f"| Oxford energy trading | {size(RAW/'additional'/'Oxford_Energy_Trading')} | {f(energy.get('cells'))} | 13 monthly check-ups | LFP, 16 Ah | "
        f"10–600 s | measured 1C capacity | not reached |",
        f"| Samsung INR18650-35E | {size(RAW/'additional'/'Samsung_INR18650_35E')} | {f(samsung.get('cells'))} | 600 cycles each | NMC, 3.4 Ah 18650 | "
        f"{f(samsung.get('median_sampling_s'))} s | provider SOH, 580 cycles | not reached |",
        f"| MIT / Severson | {size(RAW/'MIT')} | {f(mit_agg.get('cells'))} | {f(mit_agg.get('cycles_total'))} cycle records | LFP, 1.1 Ah 18650 | "
        f"{f(mit_agg.get('median_sampling_s', {}).get('median'))} s stored | provider cycle life (39 cells) | 39 cells cross 80% |",
        "",
        "## Unified analysis table",
        "",
    ]
    if not cycles.empty:
        lines += ["| Dataset | Cells | Rows | Chemistry |", "|---|---|---|---|"]
        for dataset, group in cycles.groupby("dataset"):
            lines.append(f"| {dataset} | {group.cell_id.nunique()} | {len(group):,} | {group.chemistry.iloc[0]} |")
        lines += ["", f"Total: **{len(cycles):,} cycle-level rows** across "
                      f"**{cycles.groupby(['dataset','cell_id']).ngroups} batteries** "
                      "(`data/processed/ml/cycle_features.csv`).", ""]

    lines += [
        "## NASA PCoE battery ageing set",
        "",
        "- Source: <https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip>.",
        f"- Cells analysed: {', '.join(nasa.get('cells_analyzed', []))}; 34 unique cells exist in the archive (38 `.mat` files, "
        "B0025–B0028 duplicated across nested batches).",
        f"- Per-cell discharge cycles: " + ", ".join(f"{k} {v}" for k, v in nasa.get("cycle_counts", {}).items()) + ".",
        f"- Measurement rows: {f(nasa.get('total_measurement_rows'))}; EIS records: {f(nasa.get('impedance_records'))} (Re, Rct, full spectra).",
        f"- Initial → final capacity: " + "; ".join(
            f"{c} {f(nasa.get('capacity_initial_Ah', {}).get(c), 3)} → {f(nasa.get('capacity_final_Ah', {}).get(c), 3)} Ah"
            for c in nasa.get("cells_analyzed", [])) + ".",
        f"- Final SOH: " + ", ".join(f"{k} {f(v, 1)}%" for k, v in nasa.get("soh_final_pct", {}).items()) + ".",
        f"- Voltage observed {f(nasa.get('voltage_range_V', {}).get('min'), 3)}–{f(nasa.get('voltage_range_V', {}).get('max'), 3)} V "
        "(extremes include sensor spikes outside the 2.6–4.2 V window).",
        f"- Temperature {f(nasa.get('temperature_range_C', {}).get('min'), 1)}–{f(nasa.get('temperature_range_C', {}).get('max'), 1)} °C.",
        f"- Nulls {f(nasa.get('nan_values'))}, out-of-range flags {f(nasa.get('out_of_range_values'))}.",
        "- Citation: Saha & Goebel, NASA Ames Prognostics Center of Excellence (2007).",
        "",
        "## CALCE CS2 prismatic cells",
        "",
        "- Source: <https://calce.umd.edu/data> (CS2_35, CS2_36, CS2_24, CS2_3). All four ZIPs pass CRC verification.",
        f"- {calce.get('files')} spreadsheets, {calce.get('files_unique')} unique trajectories, "
        f"{f(calce.get('measurement_rows_unique'))} measurement rows.",
        "- No temperature channel and no native SOC label in any CS2 export.",
    ]
    for cell, info in calce.get("cells", {}).items():
        lines.append(
            f"- **{cell}** — {info['cycles_total']} cycle records ({info['cycles_eligible']} eligible), "
            f"capacity {f(info['first_capacity_Ah'], 3)} → {f(info['final_capacity_Ah'], 3)} Ah, "
            f"final SOH {f(info['final_soh_pct'], 1)}%, sustained 80% crossing at global cycle {info['eol_80pct_cycle']}; "
            f"basis: {info['soh_basis']}."
        )
    lines += [
        "- Citation: CALCE Battery Research Group, University of Maryland.",
        "",
        "## Oxford Battery Degradation Dataset 1",
        "",
        f"- 8 Kokam SLPB533459H4 pouch cells (740 mAh), 40 °C chamber, Urban Artemis drive cycles.",
        f"- Check-ups per cell: " + ", ".join(f"{k} {v}" for k, v in oxford.get("checkups_per_cell", {}).items()) + ".",
        f"- Drive cycles to: " + ", ".join(f"{k} {v}" for k, v in oxford.get("max_drive_cycles", {}).items()) + ".",
        f"- Final SOH: " + ", ".join(f"{k} {f(v, 1)}%" for k, v in oxford.get("soh_final_pct", {}).items()) + ".",
        f"- Temperature glitches below 20 °C: {f(oxford.get('temperature_below_20_C'))} of "
        f"{f(oxford.get('temperature_samples'))} samples.",
        "- Citation: dataset DOI via Oxford ORA; C. R. Birkl PhD thesis, University of Oxford (2017).",
        "",
        "## Oxford energy-trading (stationary storage)",
        "",
        "- 6 cells (BMP, BMR, SPM × 2), 16 Ah LFP, field-like cycling for ~363 days, monthly capacity check-ups.",
        f"- Final retention: " + ", ".join(f"{k} {f(v, 1)}%" for k, v in energy.get("capacity_retention_final_pct", {}).items()) + ".",
        "- Zero-coded missing values exist for voltage and temperature; see the quality report.",
        "",
        "## Samsung INR18650-35E",
        "",
        f"- 2 cells, {f(samsung.get('rated_capacity_Ah'))} Ah, {f(samsung.get('cycles_measured'))} valid cycles each "
        f"({f(samsung.get('measurement_rows'))} rows at {f(samsung.get('median_sampling_s'), 1)} s).",
        f"- Provider SOH for {f(samsung.get('cycles_provider_soh'))} cycles; provider vs remeasured MAE "
        f"{f(samsung.get('provider_vs_measured_mae_pct'), 3)} pp.",
        f"- Voltage {f(samsung.get('voltage_range_V', [None])[0], 2)}–{f(samsung.get('voltage_range_V', [None, None])[1], 2)} V, "
        f"temperature {f(samsung.get('temperature_range_C', [None])[0], 1)}–{f(samsung.get('temperature_range_C', [None, None])[1], 1)} °C.",
        "- Licence CC BY 4.0, Zenodo 10.5281/zenodo.14182610; cite Sedlařík & Vyroubal.",
        "",
        "## MIT / Severson fast-charging dataset",
        "",
        f"- 49 cells ({mit_agg.get('cells_by_batch')}), {f(mit_agg.get('cycles_total'))} cycle records; "
        f"cycles per cell min {f(mit_agg.get('cycles_per_cell', {}).get('min'), 0)}, median "
        f"{f(mit_agg.get('cycles_per_cell', {}).get('median'), 0)}, max {f(mit_agg.get('cycles_per_cell', {}).get('max'), 0)}.",
        f"- {f(mit_agg.get('cells_with_provider_cycle_life'))} cells carry a provider cycle-life (EOL) label.",
        f"- Final SOH range {f(mit_agg.get('final_soh_first_pct', {}).get('min'), 1)}–"
        f"{f(mit_agg.get('final_soh_first_pct', {}).get('max'), 1)}%.",
        "- Stored cycle traces are downsampled by the provider (median spacing reported in `mit_cells.csv`).",
        "- Citation: Severson et al., *Nature Energy* 4, 466–477 (2019), <https://doi.org/10.1038/s41560-019-0356-8>.",
        "",
        "## Requires manual acquisition",
        "",
        "- CALCE CX2 / PL pouch / additional LCO-LFP-NMC sets: each archive is requested individually from "
        "<https://calce.umd.edu/data>; no direct link is invented here.",
        "- Battery Data Commons records: selected metadata saved in `data/metadata/bdc_selected_records.json`; "
        "per-dataset download still follows each provider's own access route.",
        "",
        "## Integrity manifests",
        "",
        "`nasa_integrity.json`, `calce_integrity.json`, `mit_integrity.json` and `other_integrity.json` record file "
        "sizes, hashes where computed, and archive checks performed during the audit.",
        "",
    ]
    if stage1:
        lines.append("`stage1_summary.json` aggregates every dataset summary used by the reports and the dashboard.")
    (REPORTS / "dataset_inventory.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", REPORTS / "dataset_inventory.md")


if __name__ == "__main__":
    main()
