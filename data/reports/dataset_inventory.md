# Dataset inventory

Generated 2026-11-19 14:19 from the audit outputs in `data/processed/audit/`.
Raw provider files are unchanged; extraction writes only to `data/raw/<source>/extracted/`.

## At a glance

| Dataset | Raw size | Cells | Cycles / records | Chemistry | Sampling | SOH labels | RUL labels |
|---|---|---|---|---|---|---|---|
| NASA PCoE | 629 MB | 4 audited (34 in archive) | 636 discharges | LCO, 2 Ah 18650 | 18.69 s (B0005–B7), 9.39 s (B0018) | measured capacity | 4/4 cross 80% |
| CALCE CS2 | 932 MB | 4 | 3,140,366 rows / 7724 cycle records | LCO + trace Mn prismatic | 30.02 s (CS2_35/36/24), 10.02 s (CS2_3) | measured capacity | 4/4 cross 80% |
| Oxford Degradation 1 | 266 MB | 8 | check-ups every 100 drive cycles | NMC, 740 mAh pouch | 1.00 s | pseudo-OCV capacity | 6/8 cross 80% |
| Oxford energy trading | 30 MB | 6 | 13 monthly check-ups | LFP, 16 Ah | 10–600 s | measured 1C capacity | not reached |
| Samsung INR18650-35E | 206 MB | 2 | 600 cycles each | NMC, 3.4 Ah 18650 | 10.00 s | provider SOH, 580 cycles | not reached |
| MIT / Severson | 2.13 GB | 49 | 27,957 cycle records | LFP, 1.1 Ah 18650 | 300.31 s stored | provider cycle life (39 cells) | 39 cells cross 80% |

## Unified analysis table

| Dataset | Cells | Rows | Chemistry |
|---|---|---|---|
| CALCE | 4 | 2,068 | LCO |
| MIT | 49 | 27,946 | LFP |
| NASA | 4 | 636 | LCO |
| Oxford | 8 | 519 | NMC |
| Samsung | 2 | 1,200 | NMC |

Total: **32,369 cycle-level rows** across **67 batteries** (`data/processed/ml/cycle_features.csv`).

## NASA PCoE battery ageing set

- Source: <https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip>.
- Cells analysed: B0005, B0006, B0007, B0018; 34 unique cells exist in the archive (38 `.mat` files, B0025–B0028 duplicated across nested batches).
- Per-cell discharge cycles: B0005 168, B0006 168, B0007 168, B0018 132.
- Measurement rows: 2,089,050; EIS records: 887 (Re, Rct, full spectra).
- Initial → final capacity: B0005 1.856 → 1.325 Ah; B0006 2.035 → 1.186 Ah; B0007 1.891 → 1.432 Ah; B0018 1.855 → 1.341 Ah.
- Final SOH: B0005 71.4%, B0006 58.3%, B0007 75.7%, B0018 72.3%.
- Voltage observed -0.001–8.393 V (extremes include sensor spikes outside the 2.6–4.2 V window).
- Temperature 21.8–42.3 °C.
- Nulls 6, out-of-range flags 15.
- Citation: Saha & Goebel, NASA Ames Prognostics Center of Excellence (2007).

## CALCE CS2 prismatic cells

- Source: <https://calce.umd.edu/data> (CS2_35, CS2_36, CS2_24, CS2_3). All four ZIPs pass CRC verification.
- 142 spreadsheets, 141 unique trajectories, 3,140,366 measurement rows.
- No temperature channel and no native SOC label in any CS2 export.
- **CS2_24** — 4989 cycle records (52 eligible), capacity 1.472 → 0.786 Ah, final SOH 53.4%, sustained 80% crossing at global cycle 3065; basis: full check-up discharge capacity (Vmin <= 3.2 V, Qd >= 0.45 Ah); partial 3.78-4.2 V cycles excluded.
- **CS2_3** — 873 cycle records (168 eligible), capacity 1.104 → 0.411 Ah, final SOH 37.2%, sustained 80% crossing at global cycle 147; basis: 0.11 A (C/10) discharge capacity of each six-rate block; block charge throughput recorded separately.
- **CS2_35** — 886 cycle records (880 eligible), capacity 1.139 → 0.304 Ah, final SOH 26.7%, sustained 80% crossing at global cycle 548; basis: full-cycle discharge capacity (Vmin <= 2.8 V); partial cycles excluded.
- **CS2_36** — 976 cycle records (968 eligible), capacity 1.145 → 0.172 Ah, final SOH 15.1%, sustained 80% crossing at global cycle 505; basis: full-cycle discharge capacity (Vmin <= 2.8 V); partial cycles excluded.
- Citation: CALCE Battery Research Group, University of Maryland.

## Oxford Battery Degradation Dataset 1

- 8 Kokam SLPB533459H4 pouch cells (740 mAh), 40 °C chamber, Urban Artemis drive cycles.
- Check-ups per cell: Cell1 78, Cell2 73, Cell3 76, Cell4 47, Cell5 46, Cell6 46, Cell7 77, Cell8 76.
- Drive cycles to: Cell1 8200, Cell2 7700, Cell3 8100, Cell4 5100, Cell5 5000, Cell6 5000, Cell7 8100, Cell8 8100.
- Final SOH: Cell1 76.9%, Cell2 77.4%, Cell3 78.3%, Cell4 80.0%, Cell5 69.9%, Cell6 80.1%, Cell7 80.9%, Cell8 78.3%.
- Temperature glitches below 20 °C: n/a of n/a samples.
- Citation: dataset DOI via Oxford ORA; C. R. Birkl PhD thesis, University of Oxford (2017).

## Oxford energy-trading (stationary storage)

- 6 cells (BMP, BMR, SPM × 2), 16 Ah LFP, field-like cycling for ~363 days, monthly capacity check-ups.
- Final retention: BMP_cell1 97.5%, BMP_cell2 97.6%, BMR_cell1 86.0%, BMR_cell2 85.2%, SPM_cell1 98.3%, SPM_cell2 98.2%.
- Zero-coded missing values exist for voltage and temperature; see the quality report.

## Samsung INR18650-35E

- 2 cells, 3.40 Ah, {'Battery1': 600, 'Battery2': 600} valid cycles each (1,404,621 rows at 10.0 s).
- Provider SOH for 580 cycles; provider vs remeasured MAE 0.284 pp.
- Voltage 2.65–4.20 V, temperature 17.6–35.5 °C.
- Licence CC BY 4.0, Zenodo 10.5281/zenodo.14182610; cite Sedlařík & Vyroubal.

## MIT / Severson fast-charging dataset

- 49 cells ({'B1': 47, 'B4': 2}), 27,957 cycle records; cycles per cell min 101, median 493, max 1,252.
- 39 cells carry a provider cycle-life (EOL) label.
- Final SOH range 74.6–100.1%.
- Stored cycle traces are downsampled by the provider (median spacing reported in `mit_cells.csv`).
- Citation: Severson et al., *Nature Energy* 4, 466–477 (2019), <https://doi.org/10.1038/s41560-019-0356-8>.

## Requires manual acquisition

- CALCE CX2 / PL pouch / additional LCO-LFP-NMC sets: each archive is requested individually from <https://calce.umd.edu/data>; no direct link is invented here.
- Battery Data Commons records: selected metadata saved in `data/metadata/bdc_selected_records.json`; per-dataset download still follows each provider's own access route.

## Integrity manifests

`nasa_integrity.json`, `calce_integrity.json`, `mit_integrity.json` and `other_integrity.json` record file sizes, hashes where computed, and archive checks performed during the audit.

`stage1_summary.json` aggregates every dataset summary used by the reports and the dashboard.