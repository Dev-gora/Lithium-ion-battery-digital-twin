"""Grounding material for the project assistant.

The assistant is retrieval over this file, not a generative model: every answer
is written here, quotes only artifacts that exist in the project, and names the
page where the number can be checked.  That keeps it honest on a lab machine
with no API key and no network.
"""

from __future__ import annotations

import re

STOP = {
    "what", "whats", "is", "are", "the", "a", "an", "how", "does", "do", "did", "in", "of",
    "to", "and", "for", "on", "with", "can", "i", "you", "me", "my", "we", "it", "this",
    "that", "these", "those", "why", "when", "where", "which", "about", "tell", "explain",
    "detail", "detailed", "example", "examples", "please", "show", "give", "there", "be",
    "use", "used", "using", "work", "works", "mean", "means", "from", "at", "by", "or",
}

KB: list[dict] = [
    {
        "id": "overview",
        "title": "The project in one page",
        "keywords": ["digital twin", "overview", "what is everything", "what is this app",
                     "what does this do", "project", "purpose", "summary", "start", "big picture"],
        "body": (
            "This is a **universal lithium-ion battery digital twin**. A digital twin is a "
            "model of a physical asset that keeps answering questions about it: how much charge "
            "is left, how much capacity has been lost, and how many cycles remain before the cell "
            "must be replaced.\n\n"
            "It has three layers, and the app is careful to keep them apart:\n\n"
            "- **Measured** — four public laboratory archives (NASA PCoE, CALCE CS2, Oxford, "
            "Samsung INR18650-35E), read unchanged from disk: {cells} batteries and {rows} "
            "cycle-level rows.\n"
            "- **Simulated** — a Thevenin equivalent circuit with a lumped thermal model and an "
            "empirical ageing term. You choose a battery, a load and an ambient temperature; the "
            "model integrates voltage, current, SOC and temperature over time.\n"
            "- **Predicted** — scikit-learn / XGBoost regressors trained for SOC, SOH and RUL, "
            "validated by leaving a whole battery out of every fold.\n\n"
            "The sidebar rail walks the pipeline in order: *Dashboard → Configuration → Simulator "
            "→ AI prediction → Ageing → Data sources → Dataset explorer → ML lab → Research → "
            "Assistant*."
        ),
        "example": (
            "Take the built-in EV profile: {p_s}s{p_p}p of {cell_ah} cells. The configuration page "
            "derives {p_v} nominal, {p_ah}, {p_kwh} and {p_r} pack resistance from those inputs. "
            "The simulator discharges it at 0.5C — {sim_current} at pack level — and reports where "
            "the terminal voltage crosses the cut-off. The AI page then estimates SOH from a cycle "
            "record, and the ageing page shows where the real laboratory cells of that chemistry "
            "ended up."
        ),
        "page": "Dashboard (and this assistant for the rest)",
    },
    {
        "id": "soc",
        "title": "State of charge (SOC)",
        "keywords": ["soc", "state of charge", "charge left", "battery percentage", "fuel gauge",
                     "coulomb counting", "ekf", "kalman"],
        "body": (
            "SOC is the fraction of usable charge currently in the cell, 0–100 %. The app "
            "estimates it three ways so you can see what the learning buys you:\n\n"
            "- **Naive coulomb counting** — integrates current and divides by a fixed 2.0 Ah "
            "nominal capacity. Drift is unavoidable because real capacity fades.\n"
            "- **Extended Kalman filter (EKF)** — an equivalent-circuit state observer that "
            "corrects the integration with terminal-voltage feedback. Also assumes nominal "
            "capacity.\n"
            "- **Learned regressors** — Ridge, Random Forest, Gradient Boosting and XGBoost map "
            "instantaneous voltage, current, temperature, power, dV/dt and short rolling means to "
            "SOC.\n\n"
            "There is no native SOC channel in any of the four datasets, so the reference is "
            "coulomb-integrated from *measured* current and normalised by that cycle's *measured* "
            "discharge capacity. That is the most defensible reference available, and it is why "
            "absolute SOC error should be read as a comparison between estimators rather than a "
            "calibrated accuracy claim."
        ),
        "example": (
            "The audited benchmark puts the best learned estimator ({soc_best}) at {soc_mae} mean "
            "absolute error, against {ekf_mae} for the EKF and {coulomb_mae} for fixed-capacity "
            "coulomb counting. On the 2.0 Ah nominal capacity those counters divide by, {soc_mae} "
            "of SOC is about {soc_mae_mah} of charge — arithmetic, not a measurement: it shows why "
            "the naive counter drifts once the cell's real capacity leaves nominal, while the "
            "learned model reads the voltage sag instead of assuming a capacity."
        ),
        "page": "AI prediction → SOC estimation",
    },
    {
        "id": "soh",
        "title": "State of health (SOH)",
        "keywords": ["soh", "state of health", "capacity fade", "capacity loss", "degradation",
                     "health", "capacity aware", "sensor only"],
        "body": (
            "SOH is current capacity divided by rated capacity. Every audited discharge cycle "
            "gets a measured capacity by integrating current over the discharge, so SOH is a "
            "direct observable here rather than a guess.\n\n"
            "Two feature sets are trained deliberately:\n\n"
            "- **capacity_aware** — includes the previous discharge capacity, i.e. what a "
            "laboratory cycler knows.\n"
            "- **sensor_only** — voltage, current, temperature, duration and cycle index only, "
            "i.e. what a BMS with no reference discharge can actually see. Its error is larger, "
            "and that gap is the honest cost of not having a lab.\n\n"
            "The chart on the AI page draws measured SOH as a solid line, the prediction as a "
            "dotted line, and a shaded band that is the 10th–90th percentile across the forest's "
            "individual trees. That band is a spread indicator, not a calibrated confidence "
            "interval."
        ),
        "example": (
            "{cell_label} delivers {cell_cap_start} on its first recorded discharge and "
            "{cell_cap_now} at cycle {cell_cycle} — a measured SOH of {cell_soh}. Held out "
            "completely, the capacity-aware Random Forest predicts {cell_pred} at that cycle, "
            "{cell_err} from the measurement: one easy cycle, not the whole story. Across the "
            "cell's {age_n}-cycle trajectory the fold's mean absolute error is {cell_fold_mae}, "
            "so that is the error to quote. Pick {cell_label} on the SOH tab to reproduce both."
        ),
        "page": "AI prediction → SOH estimation",
    },
    {
        "id": "rul",
        "title": "Remaining useful life (RUL)",
        "keywords": ["rul", "remaining useful life", "end of life", "eol", "when replace",
                     "how many cycles", "lifetime", "replacement"],
        "body": (
            "RUL is the number of cycles left before SOH crosses an end-of-life threshold, 80 % "
            "by default. The page lets you move the threshold between 70 % and 90 %; each "
            "threshold has its own trained artifact, because the label changes with it.\n\n"
            "Two things are shown side by side: the forest model's predicted remaining cycles, "
            "and the true remaining cycles computed from the measured trajectory. A cell that "
            "never reaches the threshold is **excluded** rather than labelled with an optimistic "
            "extrapolation — if there is no ground truth, no validated number is printed.\n\n"
            "A linear degradation-curve regression is included as a baseline: fit SOH against "
            "cycle, extrapolate to the threshold. On cells with a clear knee the learned model "
            "wins; on a short, near-linear history they agree closely."
        ),
        "example": (
            "{cell_label} reaches 80 % SOH at measured cycle {age_eol} and has {age_n} recorded "
            "cycles. Held out entirely, the Random Forest's remaining-life estimate for that cell "
            "carries a mean absolute error of {rul_cell_mae} cycles over the {rul_cell_n} cycles "
            "scored for it. That error is the resolution of the answer: a prediction made 20 "
            "cycles before the threshold is a maintenance-planning input, not a cycle-exact count."
        ),
        "page": "AI prediction → RUL prediction",
    },
    {
        "id": "validation",
        "title": "How the numbers are validated",
        "keywords": ["validation", "leave one battery out", "lobbo", "cross validation", "fold",
                     "overfit", "generalisation", "test set", "mae", "rmse", "r2", "accuracy"],
        "body": (
            "Random train/test splits leak: consecutive cycles of the same cell are almost "
            "identical, so a shuffled split lets the model memorise a battery and look brilliant. "
            "This project uses **leave-one-battery-out**: for each fold, every cycle of one cell "
            "is removed, the model trains on the remaining cells, and the held-out cell's full "
            "trajectory is predicted.\n\n"
            "What is reported is the **mean and per-fold spread** of MAE, RMSE and R². The ML lab "
            "page plots each fold as a point, which is the honest view — a good mean with one "
            "terrible fold means the model depends on a dataset it happened to see.\n\n"
            "Headline results from `experiments/results/metrics.json`: SOC {soc_best} "
            "{soc_mae} MAE (EKF {ekf_mae}, naive coulomb counting {coulomb_mae}); SOH "
            "{soh_cap_model} {soh_cap_mae} capacity-aware and {soh_sens_mae} sensor-only; RUL "
            "{rul_best} {rul_best_mae} at 80 % EOL."
        ),
        "example": (
            "Across the four SOC folds the Random Forest's MAE ranges over {soc_fold_range}: "
            "{soc_fold_spread}. Quoting the mean ({soc_mae}) alone is fine for a table, but the "
            "spread is the honest part: the estimator is much steadier on some cells than others, "
            "which is exactly what the box plot on the ML lab page refuses to hide."
        ),
        "page": "ML lab → Compare algorithms",
    },
    {
        "id": "datasets",
        "title": "The four datasets",
        "keywords": ["dataset", "nasa", "calce", "oxford", "samsung", "data source", "where does "
                     "the data come from", "labs", "provenance", "raw"],
        "body": (
            "| Source | Cells | Format | What it contributes |\n"
            "|---|---|---|---|\n"
            "| NASA PCoE | 18650 LCO, 2 Ah | `.mat` structures | charge/discharge/impedance records, "
            "EIS at several check-ups |\n"
            "| CALCE CS2 | prismatic LCO, 1.1–1.5 Ah | Arbin workbooks | long cycle life, no "
            "temperature channel |\n"
            "| Oxford | pouch, ~0.74 Ah | MATLAB check-up structs | characterisation every ~100 "
            "drive cycles, plus pseudo-OCV |\n"
            "| Samsung INR18650-35E | NMC, 3.4 Ah | tab-separated text | C/2 cycling at room "
            "temperature, 10 s sampling |\n\n"
            "Together: {cells} batteries, {rows} cycle-level rows. `data/raw` is opened "
            "read-only; "
            "every derived table lives under `data/processed/` and is regenerated by the scripts in "
            "`research/` and `preprocessing/`.\n\n"
            "Why four? Cross-chemistry and cross-laboratory coverage. A model that only ever saw "
            "NASA LCO cells would look excellent and fail on an LFP pack."
        ),
        "example": (
            "NASA B0005 ships as a `.mat` dictionary with one record per discharge. CALCE ships as "
            "an Excel workbook where each channel is a sheet of rows. The Oxford file is a check-up "
            "structure containing a 1C cycle and a pseudo-OCV sweep. The Data sources page shows "
            "each of these exactly as loaded, so you can confirm the parsers did not invent columns."
        ),
        "page": "Data sources / Dataset explorer",
    },
    {
        "id": "caveats",
        "title": "Known data limitations",
        "keywords": ["limitation", "caveat", "missing", "temperature calce", "no temperature",
                     "soc ground truth", "bias", "problem with data", "quality"],
        "body": (
            "These are stated in the app wherever they matter, and they are not cosmetic:\n\n"
            "- **No dataset provides native SOC ground truth.** References are coulomb-integrated "
            "and normalised per cycle.\n"
            "- **CALCE has no temperature channel.** Its temperature features are blank and are "
            "median-imputed from the training fold only, never from the test fold.\n"
            "- **Oxford check-ups are ~100 drive cycles apart**, so its RUL labels are coarse.\n"
            "- **Samsung has two cells**, which is far too few for standalone validation; it is "
            "used as a cross-chemistry check, not a benchmark.\n"
            "- **Simulator parameters are documented assumptions**, not datasheet values. "
            "Absolute fade rates from the simulator are illustrative; measured fade comes from the "
            "datasets."
        ),
        "example": (
            "Plot a CALCE cell on the ageing page and the temperature column is empty. That is not "
            "a parsing bug — the Arbin export simply has no thermocouple channel. Any model that "
            "reports a CALCE temperature prediction is imputing, and the app labels it that way."
        ),
        "page": "Data sources / Dataset explorer",
    },
    {
        "id": "simulator",
        "title": "The physics simulator",
        "keywords": ["simulator", "simulation", "equivalent circuit", "thevenin", "ecm", "r0",
                     "rc", "thermal", "drive cycle", "c rate", "constant power", "discharge"],
        "body": (
            "The simulator integrates a **Thevenin equivalent circuit**: an open-circuit voltage "
            "that depends on SOC and temperature, a series ohmic resistance R0, and one RC pair "
            "for charge-transfer and diffusion lag. Terminal voltage is "
            "`V = OCV(SOC) − I·R0 − V_RC`.\n\n"
            "Around that sits a **lumped thermal model** — one node, heat capacity from cell mass "
            "and specific heat, losses I²R, cooling to ambient through a surface conductance "
            "area — and an **empirical ageing term** that fades capacity and grows resistance with "
            "charge throughput, C-rate and temperature.\n\n"
            "Loads: constant C-rate, constant power, a variable profile and a drive cycle. Charge, "
            "discharge and multi-cycle ageing are all supported. Pack-level output multiplies cell "
            "voltage by the series count and divides current by the parallel count."
        ),
        "example": (
            "At 0.5C the {p_ah} pack moves {sim_current}. With {p_r} pack resistance that is a "
            "{sim_sag} step in terminal voltage the instant the load is applied — {sim_cell_sag} "
            "per cell on a {cell_ah} cell at {cell_r0}. Go to 1.0C and both the sag and the I²R "
            "heating double, so the cut-off arrives sooner and warmer."
        ),
        "page": "Battery simulator",
    },
    {
        "id": "ageing",
        "title": "Ageing and end of life",
        "keywords": ["ageing", "aging", "fade", "sei", "impedance", "resistance growth", "eis",
                     "knee", "projection", "degradation mechanism"],
        "body": (
            "Two measured signatures are shown together because they fail differently: "
            "**capacity fade** (how much charge the cell still stores) and **resistance growth** "
            "(how much voltage it loses under the same current). NASA's periodic EIS check-ups "
            "give ohmic `Re` and charge-transfer `Rct` against discharge number.\n\n"
            "The dominant mechanisms are SEI growth on the anode (slow, near-linear, consumes "
            "lithium), loss of active material, electrolyte dry-out and — at high rate or low "
            "temperature — lithium plating, which produces the sharp *knee* in the fade curve.\n\n"
            "The projection is a least-squares fade rate extrapolated to the threshold. It is "
            "labelled as a convenience estimate; the model-based RUL with measured error lives on "
            "the AI prediction page."
        ),
        "example": (
            "{cell_label} fades on a least-squares fit at {age_fade}: {cell_cap_start} on the first "
            "discharge down to {age_cap_end} at cycle {age_last_cycle}, with 80 % crossed at cycle "
            "{age_eol} and {age_end_soh} left at the end of the record. The projection on this page "
            "continues that fitted slope from the last measured point — and on this cell the "
            "trajectory is already bending, which is why the measured EOL cycle sits beside it."
        ),
        "page": "Battery ageing",
    },
    {
        "id": "configuration",
        "title": "Packs, series and parallel",
        "keywords": ["configuration", "pack", "series", "parallel", "kwh", "capacity", "voltage",
                     "profile", "ev", "laptop", "smartphone", "energy storage", "custom"],
        "body": (
            "A pack is cells in series (voltage adds, capacity stays) and parallel (capacity adds, "
            "voltage stays). So:\n\n"
            "- `V_pack = s · V_cell`\n"
            "- `C_pack = p · C_cell`\n"
            "- `E_pack = V_pack · C_pack`\n"
            "- `R_pack = s · R_cell / p`\n\n"
            "Five built-in profiles cover EV, laptop, smartphone, stationary storage and portable "
            "electronics; the Custom mode exposes every parameter. All of them are *documented "
            "assumptions* about a class of device, not datasheets for a specific product.\n\n"
            "Changing the profile changes the rest of the app: the dashboard KPIs, the simulator "
            "pack scaling and the header readout all follow the active profile."
        ),
        "example": (
            "The built-in EV profile is {p_s}s{p_p}p of {cell_ah} cells: about {p_v} nominal, "
            "{p_ah}, {p_kwh}, and {p_r} pack resistance. Resistance follows s·R_cell/p with "
            "R_cell at {cell_r0} — so doubling the parallel count doubles energy and halves "
            "resistance, which is why large packs sag less at the same current."
        ),
        "page": "Battery configuration",
    },
    {
        "id": "pages",
        "title": "What each page is for",
        "keywords": ["pages", "tabs", "navigation", "where do i find", "which page", "tour",
                     "sidebar", "rail", "menu"],
        "body": (
            "| Page | Purpose |\n"
            "|---|---|\n"
            "| Dashboard | live pack readouts, one simulated discharge, all measured fade curves |\n"
            "| Battery configuration | choose or build the pack; see derived pack quantities |\n"
            "| Battery simulator | run charge, discharge or multi-cycle ageing |\n"
            "| AI prediction | SOC / SOH / RUL tabs with per-cell predictions and errors |\n"
            "| Battery ageing | measured fade, EIS resistance growth, end-of-life projection |\n"
            "| Data sources | the audit ledger, raw samples and the files on disk, per dataset |\n"
            "| Dataset explorer | one cycle of one cell, plotted channel by channel |\n"
            "| AI / ML lab | algorithm comparison, retraining, SHAP attribution, model download |\n"
            "| Research analytics | reports, comparison tables, figures, PDF/CSV export |\n"
            "| Project assistant | this page |\n\n"
            "The rail is always visible; the control at its foot switches it between icon-and-name "
            "and icons only."
        ),
        "example": (
            "Looking for a raw voltage trace from an Oxford pouch cell? Data sources → Oxford → "
            "pick a check-up, or Dataset explorer → Oxford → cell → cycle. Want the same cycle's "
            "SOH as the models see it? Data sources → cycle ledger, which is the exact table the "
            "training script consumed."
        ),
        "page": "Every page",
    },
    {
        "id": "ml",
        "title": "Models, features and attribution",
        "keywords": ["model", "algorithm", "random forest", "gradient boosting", "xgboost",
                     "ridge", "feature", "shap", "importance", "train", "retrain", "joblib"],
        "body": (
            "Four estimators are compared per task: Ridge (a linear control), Random Forest, "
            "Gradient Boosting and XGBoost. Features come from cycle-level summaries — mean, min "
            "and max voltage, mean absolute current, mean and max temperature, duration, cycle "
            "index — plus, for SOC, instantaneous signals: voltage, current, temperature, power, "
            "dV/dt and short rolling means.\n\n"
            "SHAP values on the saved SOH model rank which features actually move a prediction. "
            "They are associations within this dataset, not causal claims: a high mean temperature "
            "matters because it co-varies with cycling history here, not because temperature is "
            "the only lever.\n\n"
            "Artifacts are `.joblib` bundles containing the estimator, the ordered feature list "
            "and the training-fold medians used for imputation — so a prediction can be "
            "reproduced without the training script."
        ),
        "example": (
            "If SHAP ranks `voltage_min_V` first for SOH, that says the depth of the voltage sag "
            "at end of discharge carries most of the health information in these cycles. Drop it "
            "and the error usually grows, which the Train-a-model tab lets you test directly by "
            "holding out a different cell."
        ),
        "page": "AI / ML lab",
    },
    {
        "id": "chem",
        "title": "Chemistries: LCO, LFP, NMC, NCA",
        "keywords": ["chemistry", "lco", "lfp", "nmc", "nca", "lithium iron phosphate", "cathode",
                     "voltage range", "which battery"],
        "body": (
            "- **LCO** (LiCoO₂) — 3.6 V nominal, high energy density, the classic 18650 laptop "
            "cell; thermally the least forgiving. NASA and CALCE cells are LCO.\n"
            "- **LFP** (LiFePO₄) — 3.2 V nominal, very flat OCV, long cycle life, excellent "
            "thermal stability; the flat curve is why SOC estimation from voltage alone struggles.\n"
            "- **NMC** (LiNiMnCoO₂) — 3.6–3.7 V, the EV workhorse; the nickel fraction trades "
            "energy density against stability. Samsung 35E is NMC.\n"
            "- **NCA** (LiNiCoAlO₂) — 3.6 V, highest practical energy density, used in long-range "
            "EVs; sensitive to temperature.\n\n"
            "The OCV curve in the simulator is a chemistry-level parametric shape, not a measured "
            "pseudo-OCV. A cell-specific twin needs measured OCV and impedance — the audit records "
            "which sources supply them."
        ),
        "example": (
            "An LFP cell holds roughly 3.2–3.3 V across most of its SOC range, moving only a few "
            "tens of millivolts. So an error of 10 mV maps to a large SOC error, while the same "
            "10 mV on an NMC cell near 3.7 V is far less ambiguous. That single fact explains most "
            "of the difficulty in LFP SOC estimation."
        ),
        "page": "Battery configuration / simulator",
    },
    {
        "id": "exports",
        "title": "Reports, tables and exports",
        "keywords": ["report", "export", "pdf", "csv", "download", "figure", "inventory",
                     "publication", "markdown"],
        "body": (
            "Everything the research write-up needs is reachable from the app:\n\n"
            "- **Markdown reports** — dataset inventory, quality report, preprocessing summary "
            "(`data/reports/`), readable in place and downloadable.\n"
            "- **Comparison table** — every task and model's MAE/RMSE/R² as one CSV.\n"
            "- **Figures** — the publication PNGs generated by the analysis stage.\n"
            "- **PDF** — assembled on demand from the live metrics plus those figures.\n\n"
            "Per-cycle and per-sample CSVs are also offered on the data pages, and trained "
            "`.joblib` bundles on the ML lab page."
        ),
        "example": (
            "To hand a supervisor the current state: Research analytics → Export → Build PDF. The "
            "first page prints the validated metrics from `metrics.json` and the footer states the "
            "data provenance and that simulated values are not measurements."
        ),
        "page": "Research analytics",
    },
    {
        "id": "setup",
        "title": "Running and rebuilding",
        "keywords": ["run", "install", "setup", "python", "streamlit", "pip", "error", "not found",
                     "missing", "troubleshoot", "rebuild", "venv"],
        "body": (
            "From `battery-digital-twin/`:\n\n"
            "```\n"
            ".venv\\Scripts\\activate          # Windows\n"
            "pip install -r requirements.txt\n"
            "streamlit run app.py\n"
            "```\n\n"
            "Rebuild order matters — each stage consumes the previous one's output:\n\n"
            "```\n"
            "python research/extract_archives.py --sources NASA CALCE\n"
            "python research/stage1_audit.py --datasets nasa oxford energy samsung calce\n"
            "python research/mit_audit.py\n"
            "python research/build_inventory.py\n"
            "python preprocessing/feature_engineering.py\n"
            "python models/training/train_all.py\n"
            "```\n\n"
            "The dashboard only reads derived artifacts, so it starts instantly and never touches "
            "the raw files."
        ),
        "example": (
            "If a page says *No cycle features found*, `data/processed/ml/cycle_features.csv` is "
            "missing — run the feature-engineering stage. If the SOH tab says *model not found*, "
            "run the training script; it writes `models/saved/soh_model_*.joblib`."
        ),
        "page": "README",
    },
    {
        "id": "glossary",
        "title": "Glossary",
        "keywords": ["glossary", "definition", "what does mean", "terminology", "cc cv", "c rate",
                     "ocv", "eis", "pp", "nominal"],
        "body": (
            "- **C-rate** — current normalised to capacity. 1C on a 3.4 Ah cell is 3.4 A.\n"
            "- **CC-CV** — constant current until the upper voltage, then constant voltage until "
            "current tapers.\n"
            "- **OCV** — open-circuit voltage, the rested terminal voltage at a given SOC.\n"
            "- **EIS** — electrochemical impedance spectroscopy; separates ohmic `Re` from "
            "charge-transfer `Rct`.\n"
            "- **SEI** — solid electrolyte interphase; grows on the anode and consumes lithium.\n"
            "- **SOH / EOL** — capacity relative to rated, and the threshold (80 % here) at which a "
            "cell is considered spent.\n"
            "- **pp** — percentage points, used for SOH error so it is not confused with a relative "
            "error.\n"
            "- **LOBO** — leave-one-battery-out validation."
        ),
        "example": (
            "Saying SOH MAE is {soh_cap_mae} means a cell at 86.0 % is typically predicted "
            "somewhere inside 86.0 ± {soh_cap_mae} — a band roughly that wide in percentage "
            "points. "
            "Saying 2 % relative would mean something much smaller, and the distinction matters "
            "in a report."
        ),
        "page": "Anywhere",
    },
    {
        "id": "limits",
        "title": "What this twin cannot do",
        "keywords": ["limitation", "cannot", "not supported", "wrong", "trust", "accuracy limit",
                     "valid", "extrapolate", "real battery"],
        "body": (
            "- It is **not a cell-level digital twin of a specific serial number**. It is a "
            "class-level twin calibrated on public data plus documented assumptions.\n"
            "- The simulator's ageing coefficients are literature-scale. Use measured fade for "
            "claims, simulated fade for intuition.\n"
            "- Predictions are interpolations within the audited chemistries and C-rates. A "
            "chemistry or load outside that envelope is extrapolation and is not validated.\n"
            "- SOC accuracy rests on a coulomb-integrated reference, not a calibrated sensor.\n"
            "- No on-line BMS streaming, no thermal runaway, no cell-to-cell balancing dynamics.\n\n"
            "What it does support: honest error bars, reproducible artifacts, and a clear "
            "separation of measured, simulated and predicted numbers."
        ),
        "example": (
            "Asking for the RUL of a brand-new LFP cell outside the audited set will produce a "
            "number, but nothing in the project validates it. The app's own rule is to refuse that "
            "claim rather than print a confident figure — the RUL tab hides output for cells whose "
            "measured trajectory never reaches the threshold."
        ),
        "page": "AI prediction",
    },
]


TITLE_INDEX = [entry["title"] for entry in KB]


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOP and len(w) > 1]


def search(question: str) -> list[tuple[float, dict]]:
    """Rank knowledge entries against a free-text question."""
    q = question.lower().strip()
    tokens = set(_tokens(question))
    ranked: list[tuple[float, dict]] = []
    for entry in KB:
        haystack = (entry["title"] + " " + entry["body"] + " " + " ".join(entry["keywords"])).lower()
        keys = set(_tokens(entry["title"] + " " + " ".join(entry["keywords"])))
        score = 3.0 * len(tokens & keys)
        for keyword in entry["keywords"]:
            if " " in keyword and keyword in q:
                score += 4.0
            elif keyword in q:
                score += 1.5
        score += sum(0.3 for token in tokens if token in haystack)
        if entry["id"] == "overview" and any(
            phrase in q for phrase in ("what is this", "what is everything", "overview", "start")
        ):
            score += 5.0
        ranked.append((round(score, 2), entry))
    ranked.sort(key=lambda pair: -pair[0])
    return ranked


def answer(question: str) -> dict | None:
    """Best entry plus the runner-up, or None when nothing is close."""
    q = question.strip()
    if not q:
        return None
    ranked = search(q)
    if not ranked or ranked[0][0] < 2.0:
        return None
    best = dict(ranked[0][1])
    best["_score"] = ranked[0][0]
    best["_related"] = [e["title"] for score, e in ranked[1:3] if score > 1.5]
    return best
