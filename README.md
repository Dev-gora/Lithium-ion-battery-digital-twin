# Universal Lithium-Ion Battery Digital Twin

State-of-charge estimation, health monitoring and remaining-useful-life prediction for
lithium-ion batteries, built on four public experimental datasets plus a configurable
physics-based simulator.

**Status: dataset discovery and quality control are complete; SOC, SOH and RUL models are
trained and validated with leave-one-battery-out splits; the interactive dashboard is
functional.** See `data/reports/` for the audit deliverables.

---

## What is inside

| Layer | Contents |
|---|---|
| Data | NASA PCoE, CALCE CS2, Oxford Degradation 1, Oxford energy-trading, Samsung INR18650-35E, MIT/Severson — 67 batteries, 32,369 cycle-level rows |
| Physics | Thevenin equivalent circuit, lumped thermal model, empirical ageing model, pack series/parallel model, cell profiles for EV / laptop / smartphone / ESS / portable / custom |
| ML | SOC (Ridge, RF, Gradient Boosting, XGBoost + EKF and coulomb-counting baselines), SOH (same regressors, two feature sets), RUL (same regressors + degradation-curve regression), anomaly detection (Isolation Forest) |
| App | Streamlit dashboard with 10 pages, line-art schematics, Plotly charts, a grounded project assistant, CSV/PDF export |

### Validated results (leave-one-battery-out)

Measured on batteries the model never saw, from `experiments/results/metrics.json`:

| Task | Best model | Metric |
|---|---|---|
| SOC | Random Forest | MAE ≈ 1.7 % SOC (EKF baseline ≈ 3.6 %, naive coulomb counting ≈ 11.4 %) |
| SOH | Random Forest (capacity-aware features) | MAE ≈ 1.1 pp; sensor-only ≈ 2.5 pp (XGBoost) |
| RUL | Random Forest | MAE ≈ 69 cycles at 80 % EOL (Gradient Boosting ≈ 80, XGBoost ≈ 74) |

Numbers are printed by `models/training/train_all.py` and recorded in
`experiments/results/metrics.json`, averaged over 67 leave-one-cell-out folds of the 32,369-row
cycle table. Re-run the training script to regenerate them.

---

## Repository layout

```
app.py                     Streamlit entry point
requirements.txt
battery/                   cell profiles, equivalent circuit, thermal, ageing, pack model
preprocessing/             feature engineering + per-dataset loaders
research/                  stage-1 audit, MIT audit, archive extraction, inventory builder
models/
  soc/estimators.py        coulomb counting + extended Kalman filter
  training/train_all.py    trains and evaluates every model
  saved/                   trained .joblib artifacts
simulator/                 charge/discharge protocols and time-stepping engine
dashboard/                 theme, cached data access, diagrams, assistant knowledge base, 10 pages
  static_export.py         renders the same pages to site/ for Netlify
deploy/                    stylesheet and resize helper for the static build
netlify.toml               publishes site/ (no build step runs on Netlify)
site/                      generated static site: 10 pages, charts, downloads
data/
  raw/                     original provider files (never modified)
  processed/audit/         per-dataset audit outputs
  processed/ml/            unified cycle table and SOC samples
  reports/                 dataset inventory, quality report, compatibility matrix,
                           preprocessing summary, publication figures
experiments/results/       metrics.json, fold tables, predictions
```

---

## Running

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt

streamlit run app.py
```

Rebuild the derived data and models:

```bash
python research/extract_archives.py --sources NASA CALCE
python research/stage1_audit.py --datasets nasa oxford energy samsung calce
python research/mit_audit.py
python research/build_inventory.py
python preprocessing/feature_engineering.py
python models/training/train_all.py
```

The dashboard reads only the derived artifacts, so it starts instantly and never touches the
raw files.

### Deploying the dashboard on Netlify

Netlify serves files, not Python processes, so it cannot run `app.py`. The same pages are
exported to static HTML instead:

```bash
python -m dashboard.static_export    # writes site/
netlify deploy --prod                # netlify.toml publishes site/
```

`dashboard/static_export.py` puts an HTML backend in place of Streamlit, imports the unchanged
page modules and calls each `render()`, so the static pages come from the same code as the app:
Plotly charts stay interactive, tables and schematics are the real ones, and every download
button becomes a file under `site/downloads/`. `netlify.toml` publishes `site/`, adds the
extensionless URLs (`/ageing`, `/ml-lab`, …) and a 404 page.

What is fixed is the widgets. Selectboxes, sliders and buttons keep the values each page
documents as its defaults, because a static file has no server to recompute with; the simulator
shows one precomputed 0.5C discharge of the configured pack, and the assistant shows three
knowledge-base answers with their worked examples. The header bar on every page says this, and
the controls are drawn as disabled rather than silently ignored. For live interaction, run the
Streamlit app.

`site/` is a build product, regenerated whenever `dashboard/` changes. It is not built on
Netlify because the export needs `data/` and `models/`, which stay local; the export uses no
network access and no Streamlit server.

---

## Data provenance and caveats

- **No dataset provides native SOC ground truth.** SOC references are coulomb-integrated from
  measured current and normalised by that cycle's measured capacity; the app states this
  wherever SOC accuracy is reported.
- **CALCE has no temperature channel.** Temperature features are blank for CALCE rows and are
  median-imputed from the training fold only.
- **Oxford check-ups are every 100 drive cycles**, so its RUL labels are coarse.
- **MIT main batch** finished downloading (2.02 GB); 49 cells are audited. `mit_cells.csv` and
  `mit_cycles.csv` hold per-cell and per-cycle records.
- **Samsung** has 2 cells only, so it is a weak standalone validation set.
- **Physics simulator parameters are documented assumptions**, not datasheet values. Ageing
  coefficients are literature-scale; measured fade comes from the datasets, simulated fade
  from the model. The app distinguishes measured / simulated / predicted values by colour and
  label.
- `data/raw` is never written to by any processing script.

## Licences

NASA PCoE: public research distribution. CALCE: academic use, cite the CALCE group.
Oxford: cite the dataset DOI and Birkl (2017). Samsung: CC BY 4.0
(10.5281/zenodo.14182610). MIT/Severson: cite Severson et al., *Nature Energy* 4 (2019).
