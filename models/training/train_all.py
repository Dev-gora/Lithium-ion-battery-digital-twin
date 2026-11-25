"""Train and evaluate the SOC, SOH and RUL models on the audited datasets.

Design rules
------------
* Battery-wise splits only: leave-one-cell-out (LOCO) for SOH/RUL, and a
  train-on-three/test-on-one rotation for SOC.  No random row splits.
* Every metric is computed on rows the model never saw.
* Baselines (Ridge, coulomb counting, EKF) are reported next to the ML models
  so that "ML is better" is only claimed when the numbers support it.

Artifacts are written to ``models/saved`` and ``experiments/results``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

RAW = PROJECT_ROOT / "data" / "raw"
SAVED = PROJECT_ROOT / "models" / "saved"
RESULTS = PROJECT_ROOT / "experiments" / "results"
ML = PROJECT_ROOT / "data" / "processed" / "ml"
for directory in (SAVED, RESULTS, ML):
    directory.mkdir(parents=True, exist_ok=True)

NASA_CELLS = ["B0005", "B0006", "B0007", "B0018"]
SEED = 42


def metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    y_true, y_pred = y_true[mask], y_pred[mask]
    r2 = r2_score(y_true, y_pred) if len(y_true) > 2 else float("nan")
    return {
        "n": int(mask.sum()),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2),
    }


def make_models(seed: int = SEED) -> dict:
    return {
        "Ridge (linear baseline)": Pipeline(
            [("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, min_samples_leaf=3, n_jobs=-1, random_state=seed
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=180, learning_rate=0.05, max_depth=3, random_state=seed
        ),
        "XGBoost": XGBRegressor(
            n_estimators=220, learning_rate=0.06, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=4,
        ),
    }


# ---------------------------------------------------------------- SOH
SOH_FEATURES_SENSOR = [
    "cycle_index",
    "voltage_mean_V",
    "voltage_min_V",
    "voltage_max_V",
    "current_absmean_A",
    "temperature_mean_C",
    "temperature_max_C",
    "duration_s",
]
SOH_FEATURES_CAPACITY = SOH_FEATURES_SENSOR + ["capacity_prev", "capacity_slope_5"]


def load_cycle_table() -> pd.DataFrame:
    path = ML / "cycle_features.csv"
    if path.exists():
        return pd.read_csv(path)
    from preprocessing.feature_engineering import save_unified

    return save_unified()


def train_soh(df: pd.DataFrame) -> dict:
    df = df.dropna(subset=["soh"])
    results: dict[str, dict] = {}
    for feature_set, features in [
        ("sensor_only", SOH_FEATURES_SENSOR),
        ("capacity_aware", SOH_FEATURES_CAPACITY),
    ]:
        present = [f for f in features if f in df.columns]
        fold_rows, preds = [], []
        for (dataset, cell), test in df.groupby(["dataset", "cell_id"]):
            train = df[~((df.dataset == dataset) & (df.cell_id == cell))]
            med = train[present].median(numeric_only=True)
            x_train = train[present].fillna(med).to_numpy()
            x_test = test[present].fillna(med).to_numpy()
            for name, model in make_models().items():
                model.fit(x_train, train["soh"].to_numpy())
                pred = model.predict(x_test)
                fold_rows.append({"model": name, "held_out": f"{dataset}/{cell}", **metrics(test["soh"], pred)})
                preds.append(pd.DataFrame({
                    "model": name, "dataset": dataset, "cell_id": cell,
                    "cycle_index": test["cycle_index"].to_numpy(),
                    "y_true": test["soh"].to_numpy(), "y_pred": pred,
                }))
        folds = pd.DataFrame(fold_rows)
        summary = {}
        for model in folds.model.unique():
            sub = folds[folds.model == model]
            summary[model] = {
                "mae_mean": float(sub.mae.mean()), "mae_std": float(sub.mae.std(ddof=1)),
                "rmse_mean": float(sub.rmse.mean()), "r2_mean": float(sub.r2.mean()),
                "folds": int(len(sub)),
            }
        pd.concat(preds).to_csv(RESULTS / f"soh_predictions_{feature_set}.csv", index=False)
        folds.to_csv(RESULTS / f"soh_folds_{feature_set}.csv", index=False)
        best = min(summary.items(), key=lambda kv: kv[1]["mae_mean"])[0]
        med = df[present].median(numeric_only=True)
        final = make_models()[best]
        final.fit(df[present].fillna(med).to_numpy(), df["soh"].to_numpy())
        joblib.dump(
            {"model": final, "features": present, "impute": med.to_dict(), "kind": feature_set},
            SAVED / f"soh_model_{feature_set}.joblib",
        )
        results[feature_set] = {"summary": summary, "best_model": best, "features": present}
        print(f"SOH [{feature_set}] best={best} MAE={summary[best]['mae_mean']:.4f}")
    return results


# ---------------------------------------------------------------- RUL
RUL_FEATURES = ["soh", "capacity_slope_5", "soh_slope_5", "cycle_since_start", "capacity_prev"]


def build_rul_table(df: pd.DataFrame, eol: float = 0.8) -> pd.DataFrame:
    rows = []
    for (dataset, cell), group in df.dropna(subset=["soh"]).groupby(["dataset", "cell_id"]):
        g = group.sort_values("cycle_index").reset_index(drop=True)
        below = np.where(g["soh"].to_numpy() <= eol)[0]
        if below.size == 0:
            continue
        g = g.iloc[: int(below[0]) + 1].copy()
        g["rul"] = np.arange(len(g), 0, -1) - 1
        g["dataset"], g["cell_id"] = dataset, cell
        rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def train_rul(df: pd.DataFrame, eol: float = 0.8) -> dict:
    rul_df = build_rul_table(df, eol)
    if rul_df.empty:
        return {"status": "no cell reaches the EOL threshold"}
    present = [f for f in RUL_FEATURES if f in rul_df.columns]
    folds, preds = [], []
    for (dataset, cell), test in rul_df.groupby(["dataset", "cell_id"]):
        train = rul_df[~((rul_df.dataset == dataset) & (rul_df.cell_id == cell))]
        if train.empty:
            continue
        med = train[present].median(numeric_only=True)
        for name, model in make_models().items():
            model.fit(train[present].fillna(med).to_numpy(), train["rul"].to_numpy())
            pred = model.predict(test[present].fillna(med).to_numpy())
            folds.append({"model": name, "held_out": f"{dataset}/{cell}", **metrics(test["rul"], pred)})
            preds.append(pd.DataFrame({
                "model": name, "dataset": dataset, "cell_id": cell,
                "cycle_index": test["cycle_index"].to_numpy(),
                "rul_true": test["rul"].to_numpy(), "rul_pred": pred,
            }))
    folds_df = pd.DataFrame(folds)
    preds_df = pd.concat(preds) if preds else pd.DataFrame()

    curve_rows = []
    for (dataset, cell), group in df.dropna(subset=["soh"]).groupby(["dataset", "cell_id"]):
        g = group.sort_values("cycle_index")
        x = g["cycle_index"].to_numpy(dtype=float)
        y = g["soh"].to_numpy(dtype=float)
        if len(x) < 10:
            continue
        slope, intercept = np.polyfit(x, y, 1)
        actual = x[y <= eol]
        if slope < 0 and actual.size:
            predicted = (eol - intercept) / slope
            curve_rows.append({
                "dataset": dataset, "cell_id": cell,
                "predicted_eol_cycle": float(predicted),
                "actual_eol_cycle": float(actual.min()),
                "error_cycles": float(predicted - actual.min()),
            })
    curve_df = pd.DataFrame(curve_rows)
    folds_df.to_csv(RESULTS / "rul_folds.csv", index=False)
    if not preds_df.empty:
        preds_df.to_csv(RESULTS / "rul_predictions.csv", index=False)
    if not curve_df.empty:
        curve_df.to_csv(RESULTS / "rul_degradation_curve.csv", index=False)

    summary = {
        model: {
            "mae_mean": float(folds_df[folds_df.model == model].mae.mean()),
            "rmse_mean": float(folds_df[folds_df.model == model].rmse.mean()),
        }
        for model in folds_df.model.unique()
    }
    best = min(summary.items(), key=lambda kv: kv[1]["mae_mean"])[0]
    med = rul_df[present].median(numeric_only=True)
    final = make_models()[best]
    final.fit(rul_df[present].fillna(med).to_numpy(), rul_df["rul"].to_numpy())
    joblib.dump(
        {"model": final, "features": present, "impute": med.to_dict(), "eol": eol},
        SAVED / f"rul_model_eol{int(eol * 100)}.joblib",
    )
    print(f"RUL best={best} MAE={summary[best]['mae_mean']:.1f} cycles over {len(folds_df.held_out.unique())} folds")
    return {
        "summary": summary, "best_model": best,
        "folds": folds_df.to_dict("records"),
        "curve_baseline": curve_df.to_dict("records"),
        "cells_with_eol": sorted(f"{d}/{c}" for d, c in rul_df.groupby(["dataset", "cell_id"]).groups),
        "eol_threshold": eol,
    }


# ---------------------------------------------------------------- SOC
SOC_FEATURES = ["voltage_V", "current_A", "temperature_C", "power_W", "dv_dt", "v_roll_mean", "i_roll_mean", "t_rel_s"]


def extract_nasa_soc(stride: int = 4) -> pd.DataFrame:
    cache = ML / "soc_samples.csv"
    if cache.exists():
        return pd.read_csv(cache)
    frames = []
    for cell in NASA_CELLS:
        mat = loadmat(
            RAW / "NASA" / "extracted" / "1_ BatteryAgingARC-FY08Q4" / f"{cell}.mat",
            squeeze_me=True, struct_as_record=False,
        )
        cycle_no = 0
        for cycle in np.atleast_1d(mat[cell].cycle):
            if str(cycle.type) != "discharge":
                continue
            cycle_no += 1
            d = cycle.data
            t = np.asarray(d.Time, float)
            # NASA logs discharge current as negative; convert to the
            # discharge-positive convention used everywhere else.
            i = -np.asarray(d.Current_measured, float)
            v = np.asarray(d.Voltage_measured, float)
            temp = np.asarray(d.Temperature_measured, float)
            cap = np.asarray(d.Capacity, float).ravel()
            if not cap.size or not np.isfinite(cap[0]):
                continue
            dt = np.diff(t, prepend=t[0])
            dt = np.where(np.isfinite(dt) & (dt >= 0), dt, 0.0)
            ah = np.cumsum(i * dt) / 3600.0
            soc = np.clip(1.0 - ah / cap[0], 0.0, 1.0)
            frames.append(pd.DataFrame({
                "cell_id": cell, "cycle_id": f"{cell}:{cycle_no}",
                "time_s": t, "voltage_V": v, "current_A": i,
                "temperature_C": temp, "soc_ref": soc, "capacity_Ah": cap[0],
            }).iloc[::stride])
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(cache, index=False)
    return df


def add_soc_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["power_W"] = out["voltage_V"] * out["current_A"]
    out["dv_dt"] = out.groupby("cell_id")["voltage_V"].diff() / out.groupby("cell_id")["time_s"].diff()
    out["v_roll_mean"] = out.groupby("cell_id")["voltage_V"].transform(lambda s: s.rolling(9, min_periods=1).mean())
    out["i_roll_mean"] = out.groupby("cell_id")["current_A"].transform(lambda s: s.rolling(9, min_periods=1).mean())
    out["t_rel_s"] = out.groupby("cell_id")["time_s"].transform(lambda s: s - s.min())
    return out.replace([np.inf, -np.inf], np.nan)


def train_soc(step: int = 3) -> dict:
    df = add_soc_features(extract_nasa_soc(stride=4)).iloc[::step].dropna(subset=SOC_FEATURES)
    from models.soc.estimators import SOCEKF, build_ocv_lookup, coulomb_count

    folds, preds = [], []
    for held in NASA_CELLS:
        train, test = df[df.cell_id != held], df[df.cell_id == held]
        med = train[SOC_FEATURES].median(numeric_only=True)
        x_train, y_train = train[SOC_FEATURES].fillna(med).to_numpy(), train["soc_ref"].to_numpy()
        x_test, y_test = test[SOC_FEATURES].fillna(med).to_numpy(), test["soc_ref"].to_numpy()

        ml_scores = {}
        for name, model in make_models().items():
            model.fit(x_train, y_train)
            pred = np.clip(model.predict(x_test), 0, 1)
            m = metrics(y_test, pred)
            ml_scores[name] = m["mae"]
            folds.append({"model": name, "held_out": held, **m})
            preds.append(pd.DataFrame({"model": name, "cell_id": held, "y_true": y_test, "y_pred": pred}))

        # Physics baselines on identical held-out samples (row-aligned).
        ocv_soc, ocv_v = build_ocv_lookup(train["soc_ref"], train["voltage_V"])
        resid = (np.interp(train["soc_ref"], ocv_soc, ocv_v) - train["voltage_V"]) / np.clip(train["current_A"], 0.2, None)
        r0 = float(np.clip(resid.dropna(), 0.0, 0.3).median()) if resid.notna().any() else 0.06

        # Physics baselines are reset at the start of every discharge cycle;
        # carrying one filter across cycles would be meaningless.
        ekf_series = pd.Series(index=test.index, dtype=float)
        cc_series = pd.Series(index=test.index, dtype=float)
        for _, cyc in test.groupby("cycle_id", sort=False):
            cyc = cyc.sort_index()
            ekf = SOCEKF(ocv_soc=ocv_soc, ocv_v=ocv_v,
                         capacity_Ah=float(cyc["capacity_Ah"].median()), r0=r0)
            ekf.soc = float(cyc["soc_ref"].iloc[0])
            ekf_series.loc[cyc.index] = ekf.run(
                cyc["current_A"].to_numpy(), cyc["voltage_V"].to_numpy(), cyc["time_s"].to_numpy()
            ).soc
            cc_series.loc[cyc.index] = coulomb_count(
                cyc["current_A"].to_numpy(), cyc["time_s"].to_numpy(), 2.0,
                initial_soc=float(cyc["soc_ref"].iloc[0]),
            )
        ekf_pred = ekf_series.to_numpy()
        cc_pred = cc_series.to_numpy()
        folds.append({"model": "EKF (physics)", "held_out": held, **metrics(y_test, ekf_pred)})
        folds.append({"model": "Coulomb counting (2.0 Ah nominal)", "held_out": held, **metrics(y_test, cc_pred)})
        best_ml = min(ml_scores.items(), key=lambda kv: kv[1])
        print(f"SOC {held}: best ML {best_ml[0]} MAE {best_ml[1]:.4f} | "
              f"EKF {folds[-2]['mae']:.4f} | CC {folds[-1]['mae']:.4f}")

    folds_df = pd.DataFrame(folds)
    folds_df.to_csv(RESULTS / "soc_folds.csv", index=False)
    pd.concat(preds).to_csv(RESULTS / "soc_predictions.csv", index=False)
    summary = {}
    for model in folds_df.model.unique():
        sub = folds_df[folds_df.model == model]
        summary[model] = {
            "mae_mean": float(sub.mae.mean()), "rmse_mean": float(sub.rmse.mean()),
            "r2_mean": float(sub.r2.mean()),
        }
    ml_only = {k: v for k, v in summary.items() if k in make_models()}
    best = min(ml_only.items(), key=lambda kv: kv[1]["mae_mean"])[0]
    med = df[SOC_FEATURES].median(numeric_only=True)
    final = make_models()[best]
    final.fit(df[SOC_FEATURES].fillna(med).to_numpy(), df["soc_ref"].to_numpy())
    joblib.dump({"model": final, "features": SOC_FEATURES, "impute": med.to_dict()}, SAVED / "soc_model.joblib")
    print(f"SOC best ML={best} MAE={summary[best]['mae_mean']:.4f}")
    return {"summary": summary, "best_model": best, "folds": folds_df.to_dict("records")}


def train_anomaly(df: pd.DataFrame) -> dict:
    """Isolation Forest on cycle-level sensor features (unsupervised)."""
    from sklearn.ensemble import IsolationForest

    feats = ["voltage_mean_V", "voltage_min_V", "voltage_max_V", "current_absmean_A",
             "temperature_mean_C", "duration_s"]
    present = [f for f in feats if f in df.columns]
    sub = df[present].dropna()
    if len(sub) < 50:
        return {"status": "insufficient rows"}
    iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=SEED, n_jobs=-1)
    labels = iso.fit_predict(sub.to_numpy())
    joblib.dump({"model": iso, "features": present}, SAVED / "anomaly_model.joblib")
    return {
        "rows": int(len(sub)), "flagged": int((labels == -1).sum()),
        "flagged_fraction": float((labels == -1).mean()),
        "features": present,
    }


def main() -> None:
    started = time.time()
    df = load_cycle_table()
    df.to_csv(ML / "cycle_features.csv", index=False)
    print(f"cycle table: {len(df)} rows, {df.groupby(['dataset','cell_id']).ngroups} cells")

    result = {
        "soh": train_soh(df),
        "rul": train_rul(df, 0.8),
        "soc": train_soc(),
        "anomaly": train_anomaly(df),
        "runtime_s": round(time.time() - started, 1),
        "seed": SEED,
        "validation": "leave-one-cell-out (battery-wise); no random row splits",
    }
    (RESULTS / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"done in {time.time() - started:.0f}s -> {RESULTS / 'metrics.json'}")


if __name__ == "__main__":
    main()
