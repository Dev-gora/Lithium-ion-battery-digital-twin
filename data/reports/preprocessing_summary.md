# Stage 1 Preprocessing Summary

**Project:** Lithium-ion battery digital twin  
**Audit date:** 19 November 2026  
**Scope:** Parsing, cycle reconstruction, label derivation, and artifact handling for the six Stage 1 datasets. This document describes what was done and how to reproduce it. No model training or feature engineering was performed.

---

## 1. Principles

1. **Raw data is read-only.** No archive, MAT file, xlsx file, or CSV in `data/raw/` was rewritten, renamed, or normalised in place.
2. **Derived data is separate.** All parsed tables and summaries live in `data/processed/audit/`; figures live in `data/reports/figures/`.
3. **No silent deletion.** Invalid, partial, or flagged rows are retained in the audit tables with flags or blank labels.
4. **No imputation or interpolation in Stage 1.** Missing values are counted and masked, not filled.
5. **No cross-dataset merging.** Each dataset is parsed independently; compatibility is documented, not assumed.
6. **Labels are explicit.** Derived SOC, SOH, and RUL are named as derived. Provider labels are marked as provider labels.

---

## 2. Environment

| Component | Version / path |
|---|---|
| Python | 3.14.4 |
| numpy | 2.4.4 |
| pandas | 3.0.2 |
| scipy | 1.17.1 |
| h5py | 3.16.0 |
| openpyxl | 3.1.5 |
| Project root | `C:\Users\devgo\OneDrive\Desktop\ai ml\battery-digital-twin` |
| Audit code | `research/stage1_audit.py`, `research/mit_audit.py` |
| Audit outputs | `data/processed/audit/` |

MATLAB 7.3 files are read with h5py. MATLAB 5.0 files are read with scipy where possible. The three small MIT MCOS table files are not decoded and are excluded from counts.

---

## 3. Directory layout

```
battery-digital-twin/
├── data/
│   ├── raw/
│   │   ├── NASA/                     # 5.+Battery+Data+Set.zip + extracted/
│   │   ├── CALCE/                    # CS2_3.zip, CS2_24.zip, CS2_35.zip, CS2_36.zip + extracted/
│   │   ├── Oxford/                   # Oxford_Battery_Degradation_Dataset_1.mat, ExampleDC_C1.mat, Readme.txt
│   │   ├── MIT/                      # 2 cycling MAT files + 3 supplementary MAT files
│   │   └── additional/
│   │       ├── Oxford_Energy_Trading/
│   │       └── Samsung_INR18650_35E/
│   ├── processed/audit/              # parsed CSVs, summaries, integrity manifests
│   ├── metadata/                     # Battery Data Commons records
│   └── reports/                      # this report and Stage 1 deliverables
├── research/
│   ├── stage1_audit.py               # NASA, CALCE, Oxford, energy, Samsung
│   └── mit_audit.py                  # MIT (memory-safe HDF5 reads)
└── models/                           # untouched; no Stage 1 training
```

---

## 4. NASA PCoE parsing

### Input

- `data/raw/NASA/5.+Battery+Data+Set.zip` → `data/raw/NASA/extracted/*/B*.mat`
- Four audited cells: `B0005`, `B0006`, `B0007`, `B0018`.

### Rules

1. MAT files are loaded with `scipy.io.loadmat(..., squeeze_me=True, struct_as_record=False)`.
2. Each cell struct contains a `cycle` array of operations: `charge`, `discharge`, `impedance`.
3. One summary row is produced per charge/discharge operation with:
   - start time from the MATLAB datenum;
   - duration from the `Time` vector;
   - voltage, current, and temperature min/mean/max;
   - reported capacity from the first finite `Capacity` value;
   - coulomb count by trapezoidal integration of `|I| dt` (diagnostic only);
   - `n_nan` and `out_of_range` counts.
4. Discharge cycles are numbered consecutively per cell.
5. SOH = reported discharge capacity / first reported discharge capacity of that cell.
6. EIS records are extracted from `impedance` operations: `Re`, `Rct`, and the impedance vector (`Rectified_Impedance` when present, otherwise `Battery_impedance`).
7. No resampling. Original sampling intervals are preserved.

### Flags

- Voltage outside 1.5–4.6 V, temperature outside −40–80 °C, or |current| > 15 A.
- Out-of-range samples are counted; they are not used to alter discharge capacity.
- Duplicate archive files `B0025.mat`–`B0028.mat` are recorded, not merged.

### Outputs

`nasa_cycle_summary.csv` (1,280 operation rows), `nasa_discharge_trajectory.csv` (636 cycles), `nasa_eis_summary.csv` (887 records), `nasa_summary.json`.

---

## 5. CALCE CS2 parsing

### Input

- `data/raw/CALCE/CS2_35.zip`, `CS2_36.zip`, `CS2_24.zip`, `CS2_3.zip`
- Extracted xlsx files under `data/raw/CALCE/extracted/`.

### Rules

1. **Archive check:** `zipfile.testzip()` on each ZIP; no bad member accepted.
2. **Columns:** `Test_Time(s)`, `Date_Time`, `Step_Index`, `Cycle_Index`, `Current(A)`, `Voltage(V)`, `Charge_Capacity(Ah)`, `Discharge_Capacity(Ah)`.
3. **Sheet concatenation:** all channel sheets in a workbook are concatenated. Repeated boundary rows are removed by `Date_Time` (keep first).
4. **Cycle ordering:** cycles are sorted by measured `Date_Time`, not by filename. Filename dates are used only to break exact duplicate timestamps.
5. **Cycle capacity:** `qd_span = max(Discharge_Capacity) − min(Discharge_Capacity)` within the cycle. This is robust to counter resets across files.
6. **CS2_3 block capacity:** each `Cycle_Index` is a six-rate characterisation block. Discharge steps are grouped by `Step_Index`; each step contributes `max(Qd) − min(Qd)`. The lowest-mean-current step (≈0.11 A, C/10) is the SOH basis. Block throughput is recorded as a diagnostic only.
7. **Cycle classification:**

| Cell | Classified as eligible when | Otherwise |
|---|---|---|
| CS2_35 | `Vmin ≤ 2.8 V` and capacity `≥ 0.15 Ah` | `partial` |
| CS2_36 | `Vmin ≤ 2.8 V` and capacity `≥ 0.15 Ah` | `partial` |
| CS2_24 | `Vmin ≤ 3.2 V` and capacity `≥ 0.45 Ah` (one check-up per file) | `partial` |
| CS2_3 | at least 5 discharge steps and low-rate capacity `≥ 0.05 Ah` | `partial_block` |

8. **SOH:** eligible-cycle capacity / first eligible-cycle capacity of that cell.
9. **Duplicate file:** `CS2_35_2_10_11.xlsx` is an exact measured-trajectory duplicate of `CS2_35_2_4_11.xlsx`; the later file is excluded via `duplicate_file_of` in `calce_file_inventory.csv`.
10. **No temperature channel exists** in the CS2 exports. No temperature is imputed.

### EOL rule

Sustained trailing rolling-median SOH ≤ 80%:

- CS2_35, CS2_36, CS2_3: window 5 eligible cycles, require 3 consecutive below-threshold values.
- CS2_24: window 3 check-ups, require 2 consecutive below-threshold values.

The first single-cycle crossing is also recorded for audit but is **not** used as the EOL label.

### Outputs

`calce_cycles.csv` (7,724 rows: 2,068 eligible, 5,656 partial), `calce_file_inventory.csv` (142 files, duplicate mapping), `calce_summary.json`, `calce_integrity.json`.

---

## 6. Oxford Battery Degradation Dataset 1 parsing

### Input

- `data/raw/Oxford/Oxford_Battery_Degradation_Dataset_1.mat`
- `data/raw/Oxford/ExampleDC_C1.mat` (first-cycle example only)

### Rules

1. The main MAT is loaded one cell at a time to limit memory.
2. Characterisation fields match `cyc\d{4}` (e.g. `cyc0100`, `cyc0200`).
3. Each check-up contains four segments: `C1ch` (1C charge), `C1dc` (1C discharge), `OCVch` (pseudo-OCV charge), `OCVdc` (pseudo-OCV discharge).
4. Segment capacity = `max(q) − min(q)` in mAh. Time is stored in days and converted to seconds (`× 86400`).
5. One summary row per segment with capacity, duration, median `dt`, voltage range, temperature range, sub-20 °C and sub-5 °C counts, and NaN count.
6. SOH uses the **pseudo-OCV discharge** capacity divided by the first check-up pseudo-OCV discharge capacity. The 1C SOH is retained for comparison.
7. No resampling; the native 1 s sampling is preserved in the summary statistics.

### Flags

- Four rows at Cell1 check-up 1600 report ~0.52 °C; treated as a sensor glitch.
- 24 pseudo-OCV segments at check-ups 3500 and 3700 report 21.3–25.6 °C across all cells (8 cells × one charge segment at 3500; 8 cells × charge and discharge at 3700); treated as a chamber/ambient excursion.
- Capacity labels are retained because the local SOH residual at those check-ups is only −0.09% to −0.42%.
- Temperature values at those segments must be masked in thermal analysis.

### Outputs

`oxford_characterisation_summary.csv` (2,076 segments), `oxford_capacity_trajectory.csv` (519 check-ups), `oxford_summary.json`.

---

## 7. Oxford energy-trading parsing

### Input

- `data/raw/additional/Oxford_Energy_Trading/`
- Six `*_capacityData.csv`, six `*_profileData.csv`, `optimal_current_profiles.csv`, `Readme.txt`.

### Rules

1. Capacity check-ups: read `time_s`, `profile_time_s`, `capacity_Ah`. SOH = `capacity_Ah / first capacity_Ah` ordered by `time_s`.
2. Profiles: read `time_s`, `current_A`, `voltage_V`, `cell_temperature_C`, `environment_temperature_C`.
3. Profile statistics are computed on raw values (for audit); zero voltage and zero temperature are **not** treated as physical measurements.
4. `0` temperature is defined by the provider readme as “no measurement available.”
5. `−1000 °C` in BMR_cell2 is a sentinel, not a temperature.
6. Profiles and check-ups are not point-aligned; no attempt was made to align them sample by sample.

### Flags and required downstream masking

| Signal | Placeholder | Count | Rule |
|---|---|---:|---|
| Voltage | `0 V` | 19,971 (2.56%) | Mask before integration |
| Cell temperature | `0 °C` | 384,530 | Mask; provider-defined non-measurement |
| Environment temperature | `0 °C` | 413,892 | Mask; provider-defined non-measurement |
| Environment temperature | `−1000 °C` | 41,766 | Mask; sentinel |
| Cell temperature | `NaN` | 15 | Mask |
| Environment temperature | `NaN` | 15 | Mask |

### Outputs

`oxford_energy_capacity.csv` (78 check-ups), `oxford_energy_summary.json`.

---

## 8. Samsung INR18650-35E parsing

### Input

- `data/raw/additional/Samsung_INR18650_35E/`
- 14 cycling files: `VZ5_008_021_VUT_D_0025_v1.txt`–`0038_v1.txt`.
- Battery1 = 0025–0031; Battery2 = 0032–0038.
- Provider SOH file: `VZ5_008_021_VUT_D_0043_v1.txt` (tab-separated: cycle, Battery1, Battery2).

### Rules

1. Read tab-separated text with Latin-1 encoding and the first seven columns:
   `cycle number`, `time/s`, `Q discharge/mA.h`, `Q charge/mA.h`, `Ecell/V`, `<I>/mA`, `Temperature/°C`.
2. Each raw block restarts its cycle counter at zero. A global cycle index is built by accumulating the **actual** local maximum of each block, not a fixed block length.
3. Per cycle: `Qdis = max(Qdis_mAh)`, `Qch = max(Qch_mAh)`, `Vmin`, `Vmax`, `Tmean`, `Tmax`, sample count, duration.
4. A cycle is valid when `Qdis > 0.5 × 3400 mAh` (1,700 mAh).
5. Provider SOH = `Qdis / 3400 mAh`.
6. Harmonised SOH = `Qdis / first valid Qdis` per battery.
7. The partial discharge at Battery2 global cycle 130 (366.5 mAh) keeps its global slot but is excluded from both SOH series.

### Outputs

`samsung_cycle_summary.csv` (1,201 rows; 1,200 valid), `samsung_reported_soh.csv` (580 provider cycles), `samsung_summary.json`.

---

## 9. MIT / Severson parsing

### Input

- `data/raw/MIT/2018-02-20_batchdata_updated_struct_errorcorrect.mat` (47 cells)
- `data/raw/MIT/2018-04-03_varcharge_batchdata_updated_struct_errorcorrect.mat` (2 cells)
- Three supplementary MATLAB 5.0 MCOS files, not decoded.

### Rules

1. Both cycling files are MATLAB 7.3/HDF5. Cell summaries are read directly; per-cycle trajectories are read through HDF5 references.
2. Summary fields: `cycle`, `QDischarge`, `QCharge`, `IR`, `Tavg`, `Tmax`, `Tmin`, `chargetime`.
3. Trajectory fields: `V`, `I`, `T`, `t` (hours). Stored spacing is converted to seconds (`× 3600`).
4. Memory-safe reading: only the selected cell and selected cycle references are loaded; the 2 GB batch file is never fully materialised.
5. Capacity validity: `0.1 Ah < QDischarge ≤ 1.65 Ah` (1.5 × nominal 1.1 Ah). Values outside this range are flagged `capacity_outlier` and excluded from SOH/EOL.
6. Baseline capacity = median of the first five valid `QDischarge` values (or nominal 1.1 Ah if none are valid).
7. SOH = `QDischarge / baseline_capacity`. Nominal SOH = `QDischarge / 1.1 Ah` is also recorded.
8. EOL = sustained trailing rolling median (window 5, run 3) ≤ 80%. Provider `cycle_life` is retained as the native cell-level label.
9. IR exactly zero for all cycles in a cell is treated as **missing IR**, not zero resistance.
10. Temperature valid range is −20 to 80 °C; values outside are counted as placeholders and excluded from ranges.
11. Trajectory voltage is checked against 1.5–4.5 V; current against |I| ≤ 10 A. No out-of-range trajectory samples were found in the sampled cycles.
12. No resampling of the stored trajectories; the stored downsampling is documented, not reversed.

### Outputs

`mit_cells.csv` (49 rows), `mit_cycles.csv` (27,957 rows), `mit_summary.json`, `mit_integrity.json`.

---

## 10. Label definitions

### 10.1 SOC (derived only)

No dataset provides native SOC. If SOC is generated in Stage 2:

```
SOC(t) = SOC(t0) + (1 / Q_rated) * ∫ I(τ) dτ
```

Requirements:

- Record the assumed initial SOC and the integration method.
- Mask zero-voltage and zero-temperature placeholders before integration.
- Report coulomb-counting drift; do not present derived SOC as ground truth.
- Use check-up capacities as periodic anchors where available (Oxford, Samsung, CALCE check-ups).

### 10.2 SOH

| Dataset | Formula | Denominator |
|---|---|---|
| NASA | `Qdis / Qdis_first` | First discharge capacity per cell |
| CALCE | `capacity / first_eligible_capacity` | Per-cell first eligible cycle |
| Oxford degradation | `capacity_pOCV / first_capacity_pOCV` | First check-up pseudo-OCV discharge |
| Oxford energy | `capacity_Ah / first_capacity_Ah` | First monthly check-up |
| Samsung | Provider `Qdis / 3.4 Ah`; harmonised `Qdis / Qdis_first` | Rated 3.4 Ah; first valid measured |
| MIT | `Qdis / median(first 5 valid Qdis)` | Cell-specific baseline |

### 10.3 RUL / EOL

| Dataset | EOL rule | Resolution |
|---|---|---|
| NASA | Sustained measured SOH ≤ 80% | Cycle |
| CALCE | Sustained trailing rolling-median SOH ≤ 80% (5/3 or 3/2) | Cycle |
| Oxford degradation | SOH ≤ 80% at check-up | ~100 drive cycles |
| Oxford energy | Not reached | — |
| Samsung | Not reached | — |
| MIT | Provider `cycle_life` (39 cells) and sustained 80% crossing (49 cells) | Cycle |

RUL labels are **derived** except MIT provider `cycle_life`. They must not be treated as independent ground truth.

---

## 11. Corrections applied during Stage 1

These are the parsing decisions that changed relative to a naive first pass:

1. **CALCE cycle order:** sorted by measured `Date_Time`, not filename. Filename dates only break duplicate ties.
2. **CALCE multi-channel sheets:** concatenated; 5,872 repeated boundary rows removed by timestamp.
3. **CS2_3 SOH basis:** changed from total block throughput to the 0.11 A (C/10) step capacity. Block throughput varies with where a six-rate block is split across files and is not a stable SOH basis.
4. **CS2_24 SOH:** restricted to the single full check-up at cycle 1 of each file; 4,937 partial cycles carry blank SOH.
5. **CS2_35/36 partials:** 6 and 8 partial cycles excluded by voltage/capacity guards; zero-capacity artifacts not labelled as full cycles.
6. **CALCE duplicate:** `CS2_35_2_10_11.xlsx` excluded as an exact trajectory duplicate.
7. **Samsung block offsets:** accumulated from each block’s actual local maximum, not a fixed 100-cycle assumption; the 30- and 70-cycle blocks are handled correctly.
8. **Samsung partial:** Battery2 global cycle 130 excluded from SOH while keeping its slot for provider numbering.
9. **MIT baseline:** median of the first five valid capacities, robust to the 2.0–2.2 Ah export outliers.
10. **MIT temperature:** −270 °C and 400 °C treated as placeholders; counted separately.
11. **MIT IR:** all-zero IR treated as missing, not zero resistance.
12. **Oxford energy:** zeros and −1000 °C treated as placeholders, not physical measurements.
13. **Oxford degradation:** low-temperature segments identified; capacity labels retained after checking the residual is ≤ 0.42%.

---

## 12. Cross-dataset harmonisation rules

These rules apply only if Stage 2 combines datasets. They do **not** authorise raw-cycle pooling.

1. **Never merge raw trajectories.** Use a canonical cycle-level schema and keep `dataset_id`, `cell_id`, and `protocol_id`.
2. **Normalise SOH per cell**, using the documented denominator for that dataset. Do not compare absolute Ah across chemistries.
3. **Keep both Samsung SOH bases.** Provider SOH is the native benchmark; harmonised SOH is for cross-dataset comparison.
4. **Mask placeholders before any modelling.** Oxford energy zeros/−1000 °C; MIT temperature placeholders; Oxford degradation thermal excursions.
5. **Do not pool NASA or Oxford degradation by chemistry.** Their positive chemistry is not stated by the provider.
6. **Treat MIT as LFP and CALCE as LCO.** They are chemistry-distinct from the NMC datasets and from each other.
7. **Samsung + Oxford energy** is the only defensible NMC/graphite cycle-level pairing. Even there: 3.4 Ah 18650 vs 16 Ah pouch, C/2 lab cycling vs grid profiles, 580 vs 13 check-ups, and no EOL in either.
8. **Split by cell, not by cycle.** This applies to every dataset.

The full pairwise compatibility record is in `data/reports/dataset_compatibility_matrix.csv`.

---

## 13. Proposed canonical cycle schema for Stage 2

| Field | Type | Meaning |
|---|---|---|
| `dataset_id` | string | `nasa_pcoe`, `calce_cs2`, `oxford_degradation`, `oxford_energy`, `samsung_inr18650`, `mit_severson` |
| `cell_id` | string | Original cell/battery identifier |
| `protocol_id` | string | Dataset-specific protocol or check-up type |
| `cycle_index` | int | Dataset-native cycle counter |
| `global_cycle` | int | Audit-order cycle index where defined |
| `timestamp_start` | ISO 8601 | Measured start time where available |
| `capacity_Ah` | float | Measured discharge capacity, if eligible |
| `soh` | float | Derived or provider SOH, blank if not eligible |
| `soh_basis` | string | `first_discharge`, `first_eligible`, `provider_rated`, `first_checkup`, `first_5_median` |
| `soh_is_native` | bool | True only for Samsung provider SOH |
| `eol_cycle` | int | Sustained 80% crossing, blank if censored |
| `eol_is_native` | bool | True only for MIT provider `cycle_life` |
| `temperature_C` | float | Representative temperature, blank if unavailable/placeholder |
| `valid` | bool | Cycle eligible for the relevant label |
| `flag` | string | `partial`, `capacity_outlier`, `temperature_glitch`, `placeholder`, etc. |

This schema is intentionally cycle-level. Trajectory-level tensors should remain dataset-specific until a sampling-aware adapter is defined.

---

## 14. Reproducibility commands

From the project root:

```powershell
# NASA, Oxford degradation, Oxford energy, Samsung, CALCE
python research/stage1_audit.py

# CALCE only (re-ran after the cell-rule fixes)
python research/stage1_audit.py --datasets calce

# MIT (memory-safe HDF5 audit)
python research/mit_audit.py
```

Outputs are written to `data/processed/audit/`. Figures are written to `data/reports/figures/`.

Integrity manifests can be re-checked with SHA-256 against:

- `data/processed/audit/calce_integrity.json`
- `data/processed/audit/nasa_integrity.json`
- `data/processed/audit/mit_integrity.json`
- `data/processed/audit/other_integrity.json`

Example PowerShell hash check for one file:

```powershell
Get-FileHash "data/raw/CALCE/CS2_35.zip" -Algorithm SHA256
```

Compare the result with `calce_integrity.json`. The NASA ZIP hash is in `other_integrity.json`; `nasa_integrity.json` records the structural ZIP test.

---

## 15. Output inventory

### 15.1 Parsed raw volume

| Dataset | Parsed volume |
|---|---:|
| NASA | 2,089,050 measurement rows; 1,280 charge/discharge operations; 636 discharge cycles; 887 EIS records |
| CALCE | 3,146,238 measurement rows (3,140,366 unique); 7,724 cycles; 2,068 eligible |
| Oxford degradation | 2,076 characterisation segments; 519 check-ups |
| Oxford energy | 781,062 profile rows; 78 capacity check-ups |
| Samsung | 1,404,621 measurement rows; 1,200 valid cycles; 580 provider-SOH cycles |
| MIT | 27,957 cycles; 49 cells |

### 15.2 Audit files

| File | Rows / contents |
|---|---|
| `nasa_cycle_summary.csv` | 1,280 operation rows |
| `nasa_discharge_trajectory.csv` | 636 discharge cycles |
| `nasa_eis_summary.csv` | 887 impedance records |
| `calce_cycles.csv` | 7,724 cycles (2,068 eligible) |
| `calce_file_inventory.csv` | 142 files with duplicate mapping |
| `oxford_characterisation_summary.csv` | 2,076 segments |
| `oxford_capacity_trajectory.csv` | 519 check-ups |
| `oxford_energy_capacity.csv` | 78 check-ups |
| `samsung_cycle_summary.csv` | 1,201 rows (1,200 valid) |
| `samsung_reported_soh.csv` | 580 provider SOH cycles |
| `mit_cells.csv` | 49 cells |
| `mit_cycles.csv` | 27,957 cycles |
| `stage1_summary.json` | Combined six-dataset summary |
| `report_stats.json` | Report statistics, EOL differences, placeholder totals |
| `*_integrity.json` | SHA-256 and structural integrity manifests |

---

## 16. Do-not-do list for Stage 2

- Do **not** treat derived SOC as measured ground truth.
- Do **not** fill Oxford energy zeros with interpolation without documenting it.
- Do **not** use MIT stored spacing as the measurement rate.
- Do **not** use all-zero MIT IR as a feature indicating zero resistance.
- Do **not** pool CALCE partial cycles with full cycles when fitting SOH.
- Do **not** use CS2_3 block throughput as the SOH basis.
- Do **not** use the first cycle as an input feature when the SOH denominator is the first cycle.
- Do **not** split cycles randomly within a cell.
- Do **not** claim NASA or Oxford degradation positive chemistry.
- Do **not** sum cycle counts across datasets as if they were equivalent.

---

## 17. Verification checklist

- [x] Raw files unchanged; SHA-256 manifests recorded.
- [x] CALCE and NASA ZIPs CRC-tested.
- [x] All six datasets parsed into audit tables.
- [x] Cycle ordering and duplicate handling documented.
- [x] Partial and non-eligible cycles retained with blank labels.
- [x] SOH/SOC/RUL definitions explicit and per-dataset.
- [x] Placeholder and artifact counts recorded.
- [x] Reproducibility commands recorded.
- [x] No model training performed.
