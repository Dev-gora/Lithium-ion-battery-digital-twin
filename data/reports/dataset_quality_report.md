# Stage 1 Dataset Quality Report

**Project:** Lithium-ion battery digital twin  
**Audit date:** 19 November 2026  
**Scope:** Data integrity, completeness, sampling consistency, artifact detection, and label quality for the six Stage 1 datasets. No models were trained.  
**Evidence base:** `data/processed/audit/*.json`, `data/processed/audit/*.csv`, provider readmes retained with the raw data, and SHA-256 manifests. All counts below are reproducible from those files.

---

## 1. Executive summary

All six datasets pass structural integrity checks: no corrupt archive or unreadable primary data file was found. The main quality risks are **placeholders and artifacts**, not file corruption.

Key conclusions:

1. **No dataset has native SOC ground truth.** Any SOC target must be coulomb-counted and labelled derived/pseudo-ground-truth.
2. **Only Samsung has a native SOH label.** NASA, CALCE, Oxford degradation, Oxford energy, and MIT SOH values are derived from measured capacity.
3. **RUL labels exist only as provider cycle life for 39/49 MIT cells**; everywhere else RUL is derived from a sustained 80% SOH crossing.
4. **Oxford energy is the lowest-quality dataset for signal-level work:** 51.11% of temperature readings are zero placeholders, 2.56% of voltage readings are zero dropouts, and BMR_cell2 contains 41,766 values of −1000 °C.
5. **MIT trajectories are downsampled**, not raw 1 Hz. Stored spacing is ~300 s median with 750–1,590 points per cycle. It remains suitable for cycle-level SOH/RUL, not for high-frequency electrical dynamics.
6. **CALCE has no temperature channel** and heterogeneous protocols; it is the strongest SOH/RUL stress-test set for partial and rate-varied cycling.
7. **No two datasets are directly poolable.** Chemistry, cell format, protocol, capacity scale, and sampling differ. The only defensible cross-dataset pairing is NMC/graphite cycle-level capacity work between Samsung and Oxford energy, and even that requires harmonisation and caveats.

### 1.1 Quality scorecard

Grade definitions:

- **A** — passes integrity; no unexplained gaps or placeholders affecting the audited signals.
- **B** — minor, quantified, documented artifacts; usable with masking or exclusion.
- **C** — material limitation that constrains the applicable tasks or requires domain-specific handling.
- **D** — unusable without re-acquisition or major repair. Not assigned to any dataset.

| Dataset | Integrity | Missing / placeholders | Temporal consistency | Label availability | Artifacts | Overall readiness |
|---|---|---|---|---|---|---|
| NASA PCoE | A | B | B | C | B | **B** |
| CALCE CS2 | A | A | B | C | B | **B** |
| Oxford degradation | A | B | A | C | B | **B** |
| Oxford energy trading | A | C | B | C | C | **C** |
| Samsung INR18650-35E | A | A | A | B | A | **B** |
| MIT / Severson | A | B | B | B | B | **B** |

The grades are task-relative. Oxford energy is still useful for stationary-storage domain validation and sensor-robustness work, but not for trajectory-level SOC supervision or RUL.

---

## 2. Method

- **Read-only audit.** Raw archives and extracted files were opened read-only. No raw file was rewritten, renamed, or normalised in place.
- **Integrity.** SHA-256 was computed for every raw archive/data file in scope. CALCE and NASA ZIPs additionally passed `zipfile.testzip()` with no bad member. MATLAB 7.3 files were opened with HDF5 readers; MATLAB 5.0 MCOS files were inspected but not decoded.
- **Parsing.** Per-dataset audit scripts (`research/stage1_audit.py`, `research/mit_audit.py`) produced the JSON/CSV outputs in `data/processed/audit/`.
- **Range checks.** Flags are dataset-specific and conservative:
  - NASA: voltage outside 1.5–4.6 V, temperature outside −40–80 °C, or |current| > 15 A.
  - CALCE: voltage outside 2.0–4.5 V or |current| > 3 A.
  - MIT trajectories: voltage outside 1.5–4.5 V, |current| > 10 A, temperature outside −20–80 °C.
  - Oxford energy: zero voltage and zero temperature treated as provider-defined non-measurements, not physical values.
- **Labels.** SOH and RUL are derived only where the measured capacity supports them. Partial or non-eligible cycles remain in the audit tables with blank labels rather than being silently discarded.

---

## 3. Integrity verification

### 3.1 Artifact-level results

| Dataset | Artifact(s) | Size | Structural result | Manifest |
|---|---|---:|---|---|
| NASA | `5.+Battery+Data+Set.zip` | 209,708,670 B | 7 entries; `testzip()` = no bad member; 38 MAT files, 34 unique cells | `nasa_integrity.json` |
| CALCE | `CS2_3.zip` | 235,649,043 B | 40 entries, 39 xlsx; no bad member | `calce_integrity.json` |
| CALCE | `CS2_24.zip` | 127,488,412 B | 54 entries, 52 xlsx; no bad member | `calce_integrity.json` |
| CALCE | `CS2_35.zip` | 37,056,519 B | 26 entries, 25 xlsx; no bad member | `calce_integrity.json` |
| CALCE | `CS2_36.zip` | 37,881,685 B | 27 entries, 26 xlsx; no bad member | `calce_integrity.json` |
| Oxford degradation | `Oxford_Battery_Degradation_Dataset_1.mat` | 266,164,290 B | Readable MAT; 8 cells | `other_integrity.json` |
| Oxford degradation | `ExampleDC_C1.mat` | 72,593 B | Readable MAT; first-cycle example | `other_integrity.json` |
| Oxford degradation | `Readme.txt` | 5,643 B | Readable; provider protocol description | `other_integrity.json` |
| Oxford energy | 12 profile/capacity CSVs + optimal profiles + readme | 14 files | All parsed as CSV/text | `other_integrity.json` |
| Samsung | 49 VZ5 data files + readme | 50 files | All read with Latin-1; 14 cycling files parsed | `other_integrity.json` |
| MIT | `2018-02-20_batchdata_updated_struct_errorcorrect.mat` | 2,022,599,329 B | MATLAB 7.3/HDF5 readable; 47 cells | `mit_integrity.json` |
| MIT | `2018-04-03_varcharge_batchdata_updated_struct_errorcorrect.mat` | 89,125,795 B | MATLAB 7.3/HDF5 readable; 2 cells | `mit_integrity.json` |
| MIT | `initialdata_all.mat`, `finaldata_4.mat`, `finaldata_6and8.mat` | 12,611,247 / 74,727 / 150,568 B | MATLAB 5.0 MCOS table objects; not decoded | `mit_integrity.json` |

**Result:** all primary cycling files are intact and readable. The three small MIT files are supplementary MATLAB 5.0 tables, not primary cycling data; they are excluded from cell and cycle counts.

### 3.2 Duplicate artifacts

| Dataset | Duplicate finding | Handling |
|---|---|---|
| NASA | `B0025.mat`–`B0028.mat` duplicate other cell files; 38 files → 34 unique cells | Not merged; duplicate files noted in `nasa_summary.json` |
| CALCE | `CS2_35_2_10_11.xlsx` is an exact measured-trajectory duplicate of `CS2_35_2_4_11.xlsx`; 142 files → 141 unique | Later duplicate excluded; mapping retained in `calce_file_inventory.csv` |
| CALCE | 5,872 repeated boundary rows across multi-channel sheets | De-duplicated by timestamp; count retained in the summary |
| Samsung | 35 non-cycling VZ5 files (ML outputs, SOH tables, analyses) | Excluded from cycling counts; not deleted |
| MIT | No duplicate cells or files found in the two cycling batches | — |
| Oxford energy | No duplicate files found | — |

---

## 4. Missing values and placeholders

### 4.1 Consolidated table

| Dataset | Parsed measurement rows / units | Explicit NaN | Zero / sentinel placeholders | Duplicate or excluded rows | Net usable for labels |
|---|---|---:|---|---:|---|
| NASA | 2,089,050 rows; 636 discharge cycles | 6 | 15 out-of-range samples (voltage excursions) | 4 duplicate cell files not merged | 636 discharge cycles; 4 cells |
| CALCE | 3,146,238 rows (3,140,366 unique); 7,724 cycles | 0 | 2 out-of-range samples (non-eligible blocks) | 5,872 boundary rows; 1 duplicate file | 2,068 eligible + 5,656 partial cycles; 4 cells |
| Oxford degradation | 519 check-ups; 2,076 segments | 0 | 24 low-temperature segments; 4-row 0.52 °C glitch (28 rows total) | 0 | 519 check-ups; 8 cells; labels retained |
| Oxford energy | 781,062 profile rows; 78 check-ups | 15 cell-T + 15 env-T | 19,971 zero V; 798,422 zero T; 41,766 × −1000 °C | 0 | 78 check-ups; 6 cells; profiles require masking |
| Samsung | 1,404,621 rows; 1,200 valid cycles | 0 | 1 partial discharge (366.5 mAh) | 1 invalid cycle excluded | 1,200 cycles; 2 cells; 580 provider-SOH cycles each |
| MIT | 27,957 cycles; 49 cells | 0 in parsed summary and sampled trajectories | 11 capacity outliers; 203 summary + 6,981 trajectory temperature placeholders; 9 all-zero-IR cells | 3 supplementary MAT files undecoded | 27,957 cycles; 49 cells; 39 provider RUL labels |

### 4.2 NASA

- **NaN:** 6 values in one B0018 charge operation. They are flagged in `nasa_cycle_summary.csv`; the operation is not used for discharge-capacity labels.
- **Out-of-range:** 15 samples across six charge operations:
  - One operation per cell in B0005/B0006/B0007 reaches 8.08–8.39 V, far above the 4.6 V audit bound. This is a sensor/step-transition artifact, not a physical cell voltage.
  - Four operations per cell show near-zero voltage minima and ~4.99 V maxima at step boundaries.
  - No current or temperature sample breached its audit bound.
- **Impact:** discharge operations — the basis of SOH — are unaffected. The flagged charge rows are retained with `out_of_range` counts.

### 4.3 CALCE

- **NaN:** 0 in parsed measurement columns.
- **Out-of-range:** 2 samples, both in non-eligible CS2_3 partial blocks:
  - Global cycle 3: 1.9627 V (below the 2.0 V bound), file `CS2_3_9_16_11.xlsx`.
  - Global cycle 47: −5.918 A (above the 3 A bound), file `CS2_3_2_6_12.xlsx`.
- **Impact:** neither row receives an SOH label. Both remain in `calce_cycles.csv` with `out_of_range = 1` for auditability.
- **Duplicate boundary rows:** 5,872 repeated rows were removed by timestamp, mostly at channel/file boundaries. This is de-duplication, not missing data.

### 4.4 Oxford degradation

- **NaN:** 0 in the 2,076 characterisation segments.
- **Temperature artifacts:**
  - **Cell1, check-up 1600:** all four segments report 0.519–0.522 °C instead of ~40 °C. This is a sensor failure on that channel; four rows affected.
  - **All eight cells, check-ups 3500 and 3700:** pseudo-OCV segments report 21.3–25.6 °C while the 1C segments at the same check-ups remain near 40 °C. This is a systematic chamber/ambient excursion during the low-rate tests, not a per-cell sensor fault. 24 segments are affected: one pseudo-OCV charge segment per cell at 3500, and charge plus discharge per cell at 3700.
- **Impact:** the pseudo-OCV SOH values at check-ups 3500 and 3700 differ from their neighbours by only −0.09% to −0.42%, so the capacity labels are retained. The affected temperature values must be masked in any thermal analysis.
- **Provider statement:** the readme says cells were tested in a 40 °C thermal chamber; the ~40 °C median agrees. The anomalous check-ups above are the exception.

### 4.5 Oxford energy trading

This is the most incomplete signal-level dataset.

- **Zero voltage:** 19,971 samples (2.56% of 781,062 profile rows). Treated as dropout, not 0 V.
- **Zero temperature:** 384,530 cell-temperature and 413,892 environment-temperature zeros; **798,422 of 1,562,124 temperature readings (51.11%)** are zero. The provider readme defines temperature `0` as “no measurement available.”
- **Sentinel:** BMR_cell2 environment temperature contains 41,766 values of **−1000 °C**.
- **NaN:** 15 cell-temperature and 15 environment-temperature values.
- **Impact:** no raw profile row should be integrated or thermally analysed until these values are masked. Capacity check-ups (78 rows) are complete and usable.
- **Minor:** BMR_cell1 reaches 4.2400 V, about 40 mV above the nominal 4.2 V window, consistent with CCCV overshoot/measurement tolerance; retained but noted.

### 4.6 Samsung

- **NaN:** none in the parsed cycling summary.
- **Partial discharge:** Battery2 global cycle 130 (block `0033`) ends with a 366.5 mAh discharge, about 12% of a normal cycle. It is excluded from SOH (`valid_discharge = False`) but retained in `samsung_cycle_summary.csv`.
- **Block stitching:** raw blocks restart `cycle number` at zero and have unequal lengths (100/30/70/100/100/100/100 cycles). A global cycle index was constructed during parsing; no cycles were lost.
- **Impact:** 1,200 valid cycles; 580 provider-SOH cycles per battery.

### 4.7 MIT

- **NaN:** 0 in the parsed summary arrays and in the sampled cycle trajectories.
- **Capacity outliers:** 11 rows across 8 cells with `QDischarge` 2.01–2.17 Ah, physically implausible for 1.1 Ah cells. Most occur at cycles 99–101. They are flagged `capacity_outlier` and excluded from SOH/EOL.
- **Temperature placeholders:** −270 °C and 400 °C occur in `MIT_B1_C038` and in one summary entry of `MIT_B4_C000`. 203 summary values and 6,981 trajectory values in sampled cycles fall outside −20–80 °C; they are excluded from ranges and counted separately.
- **IR missing:** IR is exactly zero for all cycles in 9 cells (`MIT_B1_C040`–`C046`, `MIT_B4_C000`, `MIT_B4_C001`). This is treated as missing, not zero resistance.
- **Undecoded files:** `initialdata_all.mat`, `finaldata_4.mat`, `finaldata_6and8.mat` are MATLAB 5.0 MCOS table objects. They are excluded from all cell/cycle counts.
- **Missing batch:** the `2017-05-12` batch URL returned HTTP 404 and is not in the audited set.

---

## 5. Out-of-range and physically implausible values

| Dataset | Flagged values | Nature | Used for labels? |
|---|---|---:|---|
| NASA | 15 | 8.08–8.39 V spikes and near-zero/4.99 V step transitions in charge operations | No (charge only) |
| CALCE | 2 | 1.9627 V and −5.918 A in non-eligible CS2_3 partial blocks | No |
| Oxford degradation | 28 segments | 4-row 0.52 °C sensor glitch; 24 low-temperature pseudo-OCV segments | Yes, capacity retained; temperature masked |
| Oxford energy | 860,159 placeholders/sentinels | Zero V, zero T, −1000 °C | Capacity labels unaffected; profiles masked |
| Samsung | 1 cycle | 366.5 mAh partial discharge | No |
| MIT | 11 capacity rows + 7,184 temperature values + 9 IR-missing cells | Export artifacts and placeholders | Outliers excluded; IR treated as missing |

**Principle applied:** flag and retain wherever possible; exclude only from the specific label they would corrupt. This preserves auditability and avoids silently shrinking the raw record.

---

## 6. Sampling and temporal consistency

| Dataset | Nominal / stored sampling | Temporal checks | Result |
|---|---|---|---|
| NASA | 18.687 s median (B0005–B0007); 9.391 s (B0018) | Operations ordered by cycle index; duplicate files identified | Consistent within cell; sampling differs between cells |
| CALCE | 30.015 s (CS2_35/36/24); 10.015 s (CS2_3) | Cycles ordered by measured `Date_Time`; no overlapping cycles; no duplicate start timestamps | Consistent; counters restart across files and are reconstructed |
| Oxford degradation | 1.0 s | Check-ups every ~100 drive cycles; 519 check-ups over 5,000–8,200 drive cycles | Consistent; check-up spacing gives coarse RUL resolution |
| Oxford energy | 22.29–600 s, profile-dependent | Profiles cover 362 days; check-ups monthly; profiles exclude check-up and downtime | Not point-aligned with capacity check-ups |
| Samsung | 10.0 s | 7 blocks per battery stitched to 600 cycles; one partial excluded | Consistent after stitching |
| MIT | Stored spacing median 195–300 s; 750–1,590 points/cycle (median 1,058) | No cycle gaps; no summary-length mismatches | Downsampled; not original 1 Hz |

**Important:** MIT stored-point spacing is an artifact of downsampling. It must not be described as the measurement rate, and derivatives (dI/dt, dV/dt) computed from it are not comparable to 1 Hz data.

---

## 7. Label quality

### 7.1 SOC labels

| Dataset | Native SOC? | What is available | Consequence |
|---|---|---|---|
| NASA | No | Current, voltage, temperature, capacity | SOC must be coulomb-counted; initial SOC and drift are unverified |
| CALCE | No | Current, voltage, capacity; no temperature | SOC must be coulomb-counted; no thermal correction possible |
| Oxford degradation | No | Current, voltage, temperature, check-up capacity | SOC must be coulomb-counted; check-ups give periodic anchors |
| Oxford energy | No | Current, voltage, temperature (with placeholders), check-up capacity | SOC integration is compromised by 2.56% zero-voltage dropouts |
| Samsung | No | Current, voltage, temperature, capacity | SOC must be coulomb-counted |
| MIT | No | Current, voltage, temperature, capacity | SOC must be coulomb-counted; downsampled trajectories add integration error |

**Conclusion:** there is **zero native SOC supervision** in Stage 1. Any SOC model must be trained against derived labels and evaluated as such.

### 7.2 SOH labels

| Dataset | Native SOH? | Audit definition | Denominator / basis | Final SOH range | Main caveat |
|---|---|---|---|---|---|
| NASA | No | `Qdis / Qdis_first` | First discharge capacity per cell (1.855–2.035 Ah) | 58.25–75.75% | Capacity reported per discharge; no temperature correction |
| CALCE | No | Measured capacity / first eligible capacity | Per-protocol first eligible capacity (1.1043–1.4722 Ah) | 15.05–53.41% | Different bases per cell; no temperature |
| Oxford degradation | No | Pseudo-OCV discharge capacity / first check-up | First check-up pOCV capacity (726.97–739.82 mAh) | 69.91–80.94% | Coarse check-up spacing; two thermal excursions |
| Oxford energy | No | 1C capacity / first measured capacity | First monthly 1C capacity (17.153–17.262 Ah) | 85.15–98.35% | Only 13 check-ups; no EOL |
| Samsung | **Yes** | Provider `Qdis / 3.4 Ah`; harmonised `Qdis / Qdis_first` | Rated 3.4 Ah; harmonised first measured capacity | 82.13–83.10% provider; 84.37–85.02% harmonised | Only two cells; provider and harmonised bases differ by ~2 pp |
| MIT | No | `Qdis / median(first 5 valid Qdis)` | Cell-specific first-five median (0.967–1.109 Ah) | 74.62–100.11% | 10 censored cells; 11 capacity outliers excluded |

SOH values slightly above 100% occur in CALCE CS2_3 (100.38% at global cycle 2) and MIT (`MIT_B1_C037`, 100.11%). These are measurement noise around the baseline, not capacity gain, and should be clipped or treated as 100% for modelling.

### 7.3 RUL labels

| Dataset | Native RUL? | EOL definition | Cells reaching EOL | Censoring / resolution |
|---|---|---|---|---|
| NASA | No | Sustained measured SOH ≤ 80% | 4/4 | Cycle-level resolution |
| CALCE | No | Sustained trailing rolling-median SOH ≤ 80% (5/3 window for CS2_35/36/3; 3/2 for CS2_24) | 4/4 | Cycle-level; partial cycles excluded |
| Oxford degradation | No | SOH ≤ 80% at check-up | 6/8 | ~100 drive-cycle resolution; Cell6 80.06%, Cell7 80.94% |
| Oxford energy | No | Not reached | 0/6 | No RUL labels |
| Samsung | No | Not reached | 0/2 | No RUL labels |
| MIT | Partial | Provider `cycle_life` for 39 cells; derived sustained 80% crossing | 39/49 | 10 censored; derived vs provider median +4 cycles, range −1 to +36 |

### 7.4 Leakage risks

- **Baseline leakage:** defining SOH as capacity divided by the first cycle makes the first cycle exactly 100%. Do not use the first cycle as an input feature when predicting SOH.
- **Target leakage:** RUL is derived from the same SOH series used as a target. Split by cell, not by random cycle, or the derived EOL will leak across folds.
- **Provider-label leakage:** MIT `cycle_life` is a cell-level outcome. It must never be an input feature.
- **Cross-cell leakage:** Samsung has only two cells; any train/test split that puts cycles from the same cell on both sides overstates performance.

---

## 8. Per-dataset findings and required handling

### 8.1 NASA PCoE

**Strengths:** clean discharge capacity series, 636 cycles across four cells, 887 EIS records, cell and ambient temperature.

**Issues:**
- 6 NaN values in one B0018 charge operation.
- 15 out-of-range voltage samples in charge operations.
- Sampling differs between B0018 (9.39 s) and the other three cells (18.69 s).
- Chemistry not stated by provider.
- Four duplicate cell files in the archive.

**Required handling:** use discharge operations for SOH; ignore flagged charge artifacts; keep B0018 sampling separate; do not claim chemistry.

### 8.2 CALCE CS2

**Strengths:** large, well-documented cycling set; four distinct protocols; full and partial cycling; six-rate blocks; sustained EOL crossings.

**Issues:**
- No temperature channel.
- No native SOC.
- Counters and `Data_Point` restart across files.
- CS2_24 is 98.9% partial cycles; only 52 check-ups carry SOH.
- CS2_3 block boundaries split across files; block throughput varies with split position.
- One duplicate file and 5,872 boundary rows.

**Required handling:** order by `Date_Time`; use per-cell SOH bases; retain partial cycles with blank labels; use the 0.11 A step for CS2_3 SOH; never use block throughput as the SOH basis.

### 8.3 Oxford degradation

**Strengths:** 1 s electrical data, 519 check-ups, eight cells, clear EOL for six cells.

**Issues:**
- Positive-electrode chemistry not stated.
- RUL resolution is ~100 drive cycles.
- Four-row 0.52 °C sensor glitch at Cell1 check-up 1600.
- 24 low-temperature pseudo-OCV segments at check-ups 3500/3700 across all cells.
- Capacity labels are pseudo-OCV based, not 1C based (both are recorded).

**Required handling:** mask the 28 temperature-affected segments (4 glitch + 24 excursion); retain capacity labels (residual ≤ 0.42%); treat EOL timing as interval-censored at ~100 cycles.

### 8.4 Oxford energy trading

**Strengths:** realistic stationary-storage profiles, six cells, monthly capacity check-ups, 362 days of operation.

**Issues:**
- 51.11% zero temperature placeholders.
- 2.56% zero voltage dropouts.
- 41,766 × −1000 °C sentinel values in BMR_cell2.
- Profiles are not point-aligned with check-ups.
- No cell reaches 80% EOL.
- BMP cells use a 10–90% SOC voltage window; BMR/SPM use the full window.

**Required handling:** mask zeros and −1000 before any integration; never impute zero as a physical temperature or voltage; treat BMP and BMR/SPM as separate protocols; use only for domain validation and robustness, not RUL.

### 8.5 Samsung INR18650-35E

**Strengths:** native provider SOH, 600 cycles per cell, 10 s sampling, temperature, clean cycling summary.

**Issues:**
- Only two cells.
- No EOL.
- One partial discharge excluded.
- Provider SOH uses rated 3.4 Ah; harmonised SOH differs by up to ~2 pp.
- 35 of 50 files are non-cycling outputs and must not be mistaken for raw cycling data.

**Required handling:** record both SOH bases; use the harmonised SOH for cross-dataset work; keep the provider SOH as the native benchmark; do not claim EOL.

### 8.6 MIT / Severson

**Strengths:** 49 cells, 27,957 cycles, provider cycle life for 39 cells, diverse fast-charge policies, temperature and IR channels.

**Issues:**
- Trajectories are downsampled (~300 s stored spacing).
- 11 capacity outliers across 8 cells.
- 9 cells have all-zero IR (missing, not zero resistance).
- Temperature placeholders in two cells.
- 10 cells are censored.
- Three supplementary MAT files are undecoded.
- One 2017 batch URL was unavailable (404).

**Required handling:** use cycle-level summary for SOH/RUL; exclude capacity outliers; treat zero IR as missing; mask temperature placeholders; do not use stored spacing as the measurement rate; split by cell.

---

## 9. Fitness for Stage 2 tasks

Legend: **P** = primary, **S** = secondary/with caveats, **N** = not recommended.

| Dataset | SOC estimation | SOH estimation | RUL prediction | Anomaly detection | Physics-informed modelling |
|---|---|---|---|---|---|
| NASA | S (coulomb-count only) | P | P | P (EIS + thermal) | S (EIS, thermal) |
| CALCE | N | P | P | S (voltage/current artifacts) | S (rate-dependent aging) |
| Oxford degradation | S (check-up anchors) | P | P (coarse) | P (thermal glitches) | S (drive-cycle aging) |
| Oxford energy | S (masked integration) | S | N | P (placeholders/dropouts) | P (grid profiles) |
| Samsung | S (coulomb-count only) | P (native) | N | S (partial cycle) | S (NMC 18650) |
| MIT | S (downsampled) | P | P (39/49) | P (thermal/IR/capacity) | P (fast-charge policies) |

**SOC:** no primary dataset. Use derived labels only, and report them as pseudo-ground-truth.
**SOH:** five datasets are primary; Oxford energy is secondary.
**RUL:** NASA, CALCE, Oxford degradation, and MIT are usable; Samsung and Oxford energy are not.
**Anomaly detection:** every dataset contributes a different artifact class; this is a strength for robustness testing but not for pooled supervision.
**Physics-informed modelling:** Oxford energy for grid profiles; NASA for EIS; CALCE for rate-dependent ageing; MIT for fast-charge policy effects.

---

## 10. Residual risks and mandatory Stage 2 actions

1. **Do not pool raw cycles across datasets.** Build per-dataset adapters and a canonical schema; harmonise only after explicit mapping.
2. **Mask before integrating.** Oxford energy zeros and −1000 °C; MIT temperature placeholders; all out-of-range samples.
3. **Keep derived labels marked.** SOC is always derived. SOH and RUL are derived except Samsung SOH and MIT provider cycle life.
4. **Split by cell.** Random cycle splits leak baseline and EOL information.
5. **Report per-dataset metrics first.** Pooled metrics are only valid after protocol and chemistry harmonisation, and should carry a domain-shift caveat.
6. **Preserve the raw evidence.** SHA-256 manifests and the audit CSVs are the reference for every number in Stage 2.
7. **Do not overstate chemistry.** NASA and Oxford degradation positive chemistry remain “not stated by provider.”

---

## 11. Quality checklist

- [x] All six datasets integrity-checked; SHA-256 manifests recorded.
- [x] CALCE and NASA ZIP CRC tests passed with no bad members.
- [x] MIT cycling files opened as HDF5 and audited cell by cell.
- [x] Missing values, placeholders, and out-of-range samples counted and located.
- [x] Duplicate files and duplicate boundary rows identified and mapped.
- [x] Sampling rates and downsampling documented.
- [x] SOH bases and denominators documented per dataset.
- [x] RUL definitions, censoring, and resolution documented.
- [x] Leakage risks identified.
- [x] No raw file modified; no model training performed.
- [x] Findings agree with `stage1_summary.json`, `calce_summary.json`, `mit_summary.json`, `report_stats.json`, and the integrity manifests.
