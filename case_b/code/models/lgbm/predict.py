"""
Case B: LightGBM forecast for a whole hold-out year, as a comparison to LEAR.

Uses the same data, the same regressor blocks (CONFIG in ../lear/config.py)
and the same hold-out year as the LEAR scripts, but one model for all 24
hours, with the hour as a feature.

  1. Hyperparameters are tuned once with Optuna (25 trials) on the two years
     before the hold-out year: train on the first, validate on the second
     (2018/2019 for 2020, 2022/2023 for 2024).
  2. Every day d of the hold-out year, the model is refitted with those
     settings on the 365 days before d and forecasts the 24 hours of d.
     Price lags are the observed prices.

Run:     python predict.py      (from models/lgbm/; needs lightgbm and optuna)
Input:   data/prepared/DK1/lear_dataset.csv
Output:  results/lgbm/<label>_{hourly,daily,rmse_by_hour}.csv,
         <label>_summary.png, <label>_rmse_by_hour.png, <label>_best_params.json
         The forecast column is called forecast_lear so the files can be read
         by the same plotting code as the LEAR results.
"""

import sys
import json
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lear"))

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from core import (
    load_df_dict,
    CROSS_ZONES,
    DK1_CHANNELS,
    CROSS_ZONE_FLOWS,
    REF_DATE,
)
from config import HOLD_OUT_YEAR, CONFIG

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT    = Path(__file__).parent.parent.parent.parent
DATA    = ROOT / "data/prepared/DK1/lear_dataset.csv"
OUT_DIR = ROOT / "results/lgbm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Output label: lgbm_<year> + one suffix per active flag (as in rolling.py)
_suffixes = ["lgbm", str(HOLD_OUT_YEAR)]
if CONFIG.use_cross_zone_prices:
    _suffixes.append("xzone")
if CONFIG.use_cross_zone_ren:
    _suffixes.append("ren")
if CONFIG.use_dk1_transmission:
    _suffixes.append("trans")
if CONFIG.use_cross_zone_transmission:
    _suffixes.append("trans_xz")
if CONFIG.use_energy:
    _suffixes.append("energy")
if CONFIG.use_macro_daily:
    _suffixes.append("macro")
if CONFIG.use_holidays:
    _suffixes.append("holidays")
if CONFIG.use_interactions:
    _suffixes.append("interact")
LABEL = "_".join(_suffixes)

print(f"Case B LightGBM — hold-out year={HOLD_OUT_YEAR}", flush=True)
print(f"  Label: {LABEL}", flush=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("\nLoading lear_dataset.csv...", flush=True)
df_dict = load_df_dict(DATA)
all_dates_sorted = sorted(df_dict.keys())
print(f"  {len(all_dates_sorted)} days loaded", flush=True)

# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------
def _get_price(df_dict: dict, ds: str, h: int) -> float:
    """Observed DK1 price for date ds and hour h (NaN if not in the data)."""
    rec = df_dict.get(ds)
    return float(rec["prices"][h]) if rec is not None else np.nan


def build_features(
    dates: list,
    df_dict: dict,
    include_target: bool = True,
) -> pd.DataFrame:
    """Feature table with 24 rows per date (hour 0-23).

    The same blocks as the LEAR regressor vector, switched on and off by
    CONFIG, but each row only holds the values for its own hour.
    """
    rows = []
    for ds in dates:
        d   = pd.Timestamp(ds)
        rec = df_dict.get(ds)

        for h in range(24):
            row: dict = {}

            # --- Calendar ---
            row["hour"]       = h
            row["dow"]        = d.dayofweek
            row["month"]      = d.month
            row["doy"]        = d.dayofyear
            if CONFIG.use_holidays:
                row["is_holiday"] = float(rec["is_holiday"]) if rec is not None else 0.0

            # --- AR prices DK1: lags d-1, d-2, d-3, d-7 ---
            for lag in [1, 2, 3, 7]:
                lag_ds = (d - pd.Timedelta(days=lag)).strftime("%Y-%m-%d")
                row[f"price_lag{lag}d"] = _get_price(df_dict, lag_ds, h)

            # --- Own exog: load, onshore, offshore, solar x lags d, d-1, d-7 ---
            if CONFIG.use_exogenous:
                for lag in [0, 1, 7]:
                    lag_ds  = (d - pd.Timedelta(days=lag)).strftime("%Y-%m-%d")
                    lag_rec = df_dict.get(lag_ds)
                    row[f"load_lag{lag}d"]     = float(lag_rec["load"][h])    if lag_rec is not None else np.nan
                    row[f"onshore_lag{lag}d"]  = float(lag_rec["onshore"][h]) if lag_rec is not None else np.nan
                    row[f"offshore_lag{lag}d"] = float(lag_rec["offshore"][h])if lag_rec is not None else np.nan
                    row[f"solar_lag{lag}d"]    = float(lag_rec["solar"][h])   if lag_rec is not None else np.nan

            # --- Cross-zone prices: lags d-1, d-7 ---
            if CONFIG.use_cross_zone_prices:
                for lag in [1, 7]:
                    lag_ds  = (d - pd.Timedelta(days=lag)).strftime("%Y-%m-%d")
                    lag_rec = df_dict.get(lag_ds)
                    for z in CROSS_ZONES:
                        row[f"price_{z}_lag{lag}d"] = (
                            float(lag_rec["prices_xz"][z][h]) if lag_rec is not None else np.nan
                        )

            # --- Cross-zone ren: lags d, d-1, d-7 ---
            if CONFIG.use_cross_zone_ren:
                for lag in [0, 1, 7]:
                    lag_ds  = (d - pd.Timedelta(days=lag)).strftime("%Y-%m-%d")
                    lag_rec = df_dict.get(lag_ds)
                    for z in CROSS_ZONES:
                        row[f"ren_{z}_lag{lag}d"] = (
                            float(lag_rec["ren_xz"][z][h]) if lag_rec is not None else np.nan
                        )

            # --- DK1 channel flows: lags d-1, d-7 ---
            if CONFIG.use_dk1_transmission:
                for lag in [1, 7]:
                    lag_ds  = (d - pd.Timedelta(days=lag)).strftime("%Y-%m-%d")
                    lag_rec = df_dict.get(lag_ds)
                    for ch in DK1_CHANNELS:
                        row[f"flow_dk1_{ch}_lag{lag}d"] = (
                            float(lag_rec["flow_dk1_ch"][ch][h]) if lag_rec is not None else np.nan
                        )

            # --- Cross-zone channel flows: lags d-1, d-7 ---
            if CONFIG.use_cross_zone_transmission:
                for lag in [1, 7]:
                    lag_ds  = (d - pd.Timedelta(days=lag)).strftime("%Y-%m-%d")
                    lag_rec = df_dict.get(lag_ds)
                    for z, channels in CROSS_ZONE_FLOWS.items():
                        for ch in channels:
                            row[f"flow_{z}_{ch}_lag{lag}d"] = (
                                float(lag_rec["flow_xz"][(z, ch)][h]) if lag_rec is not None else np.nan
                            )

            # --- Energy scalars: lags d-1, d-7 ---
            if CONFIG.use_energy:
                for lag in [1, 7]:
                    lag_ds  = (d - pd.Timedelta(days=lag)).strftime("%Y-%m-%d")
                    lag_rec = df_dict.get(lag_ds)
                    en = lag_rec["energy"] if lag_rec is not None else np.full(4, np.nan)
                    row[f"coal_lag{lag}d"] = float(en[0])
                    row[f"eua_lag{lag}d"]  = float(en[1])
                    row[f"gas_lag{lag}d"]  = float(en[2])
                    row[f"oil_lag{lag}d"]  = float(en[3])

            # --- Macro daily scalars: lags d-1, d-7 ---
            if CONFIG.use_macro_daily:
                for lag in [1, 7]:
                    lag_ds  = (d - pd.Timedelta(days=lag)).strftime("%Y-%m-%d")
                    lag_rec = df_dict.get(lag_ds)
                    md = lag_rec["macro_daily"] if lag_rec is not None else np.full(4, np.nan)
                    row[f"stoxx_lag{lag}d"]  = float(md[0])
                    row[f"msci_lag{lag}d"]   = float(md[1])
                    row[f"usdeur_lag{lag}d"] = float(md[2])
                    row[f"vix_lag{lag}d"]    = float(md[3])

            # --- Interaction / nonlinear features ---
            if CONFIG.use_interactions:
                rec_d  = df_dict.get(ds)
                rec_d1 = df_dict.get((d - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))

                if rec_d is not None:
                    nl_dk1 = float(rec_d["load"][h] - rec_d["onshore"][h]
                                   - rec_d["offshore"][h] - rec_d["solar"][h])
                    nl_de  = float(rec_d["load_de"][h] - rec_d["wind_de"][h]
                                   - rec_d["solar_de"][h])
                    nl_dk2 = float(rec_d["load_dk2"][h] - rec_d["wind_dk2"][h]
                                   - rec_d["solar_dk2"][h])
                else:
                    nl_dk1 = nl_de = nl_dk2 = 0.0

                row["nl_dk1"]         = nl_dk1
                row["nl_dk1_sq"]      = nl_dk1 ** 2
                row["nl_dk1_x_de"]    = nl_dk1 * nl_de
                row["nl_dk1_x_dk2"]   = nl_dk1 * nl_dk2

                if rec_d1 is not None:
                    precip = float(rec_d1["precip_no2_r14"])
                    flow   = float(rec_d1["flow_dk1_ch"]["no2"][h])
                    row["precip_x_flow"] = precip * flow
                    row["temp_sq"]       = (float(rec_d1["temp_dk1"]) - 180.0) ** 2
                else:
                    row["precip_x_flow"] = 0.0
                    row["temp_sq"]       = 0.0

            if include_target and rec is not None:
                row["target"] = float(rec["prices"][h])
            else:
                row["target"] = np.nan

            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Data for hyperparameter tuning
# ---------------------------------------------------------------------------
if HOLD_OUT_YEAR == 2020:
    tune_train_start, tune_train_end = "2018-01-09", "2018-12-31"
    tune_val_start,   tune_val_end   = "2019-01-01", "2019-12-31"
elif HOLD_OUT_YEAR == 2024:
    tune_train_start, tune_train_end = "2022-01-09", "2022-12-31"
    tune_val_start,   tune_val_end   = "2023-01-01", "2023-12-31"
else:
    # Any other year: train on the two years before the validation year
    tune_train_start = f"{HOLD_OUT_YEAR - 3}-01-09"
    tune_train_end   = f"{HOLD_OUT_YEAR - 2}-12-31"
    tune_val_start   = f"{HOLD_OUT_YEAR - 1}-01-01"
    tune_val_end     = f"{HOLD_OUT_YEAR - 1}-12-31"

print(f"\nBuilding tuning feature matrices...", flush=True)
print(f"  Train: {tune_train_start} → {tune_train_end}", flush=True)
print(f"  Val  : {tune_val_start}   → {tune_val_end}", flush=True)

tune_train_dates = [d for d in all_dates_sorted
                    if tune_train_start <= d <= tune_train_end]
tune_val_dates   = [d for d in all_dates_sorted
                    if tune_val_start   <= d <= tune_val_end]

feat_train = build_features(tune_train_dates, df_dict, include_target=True)
feat_val   = build_features(tune_val_dates,   df_dict, include_target=True)

feat_train = feat_train.dropna(subset=["target"]).reset_index(drop=True)
feat_val   = feat_val.dropna(subset=["target"]).reset_index(drop=True)

FEATURE_COLS = [c for c in feat_train.columns if c != "target"]
print(f"  Tune train: {len(feat_train):,} rows | Tune val: {len(feat_val):,} rows "
      f"| {len(FEATURE_COLS)} features", flush=True)

X_tr = feat_train[FEATURE_COLS].values.astype(float)
y_tr = feat_train["target"].values.astype(float)
X_vl = feat_val[FEATURE_COLS].values.astype(float)
y_vl = feat_val["target"].values.astype(float)

# Drop rows with any missing feature
tr_ok = np.isfinite(X_tr).all(axis=1) & np.isfinite(y_tr)
vl_ok = np.isfinite(X_vl).all(axis=1) & np.isfinite(y_vl)
X_tr, y_tr = X_tr[tr_ok], y_tr[tr_ok]
X_vl, y_vl = X_vl[vl_ok], y_vl[vl_ok]
print(f"  After finite filter — train: {len(X_tr):,}  val: {len(X_vl):,}", flush=True)

# ---------------------------------------------------------------------------
# Hyperparameter search (validation RMSE)
# ---------------------------------------------------------------------------
N_OPTUNA_TRIALS = 25

def objective(trial: optuna.Trial) -> float:
    params = dict(
        num_leaves        = trial.suggest_int("num_leaves", 20, 200),
        max_depth         = trial.suggest_int("max_depth", 3, 10),
        min_child_samples = trial.suggest_int("min_child_samples", 10, 100),
        learning_rate     = trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        n_estimators      = trial.suggest_int("n_estimators", 200, 1000),
        subsample         = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.6, 1.0),
        random_state      = 42,
        verbose           = -1,
    )
    m = lgb.LGBMRegressor(**params)
    m.fit(X_tr, y_tr)
    preds = m.predict(X_vl)
    err   = preds - y_vl
    return float(np.sqrt(np.nanmean(err ** 2)))


print(f"\nRunning Optuna ({N_OPTUNA_TRIALS} trials)...", flush=True)
study = optuna.create_study(direction="minimize",
                            sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

best_params = {**study.best_params, "random_state": 42, "verbose": -1}
print(f"\nBest val RMSE: {study.best_value:.3f} EUR/MWh", flush=True)
print("Best params:", flush=True)
for k, v in best_params.items():
    if k not in ("random_state", "verbose"):
        print(f"  {k}: {v}", flush=True)

params_path = OUT_DIR / f"{LABEL}_best_params.json"
with open(params_path, "w") as f:
    json.dump(study.best_params, f, indent=2)
print(f"Saved: {params_path}", flush=True)

# ---------------------------------------------------------------------------
# Forecast each day of the hold-out year
# ---------------------------------------------------------------------------
forecast_dates = pd.date_range(f"{HOLD_OUT_YEAR}-01-01", f"{HOLD_OUT_YEAR}-12-31", freq="D")
all_results    = []
WINDOW_DAYS    = 365

print(f"\nForecasting {forecast_dates[0].date()} → {forecast_dates[-1].date()} "
      f"({len(forecast_dates)} days) with {WINDOW_DAYS}-day rolling window...\n", flush=True)

for day_i, forecast_date in enumerate(forecast_dates):
    ds = forecast_date.strftime("%Y-%m-%d")

    # Training window: the 365 days before the forecast day
    cutoff_end   = (forecast_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    cutoff_start = (forecast_date - pd.Timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")
    train_dates  = [d for d in all_dates_sorted
                    if cutoff_start <= d <= cutoff_end
                    and (pd.Timestamp(d) - REF_DATE).days >= 7]

    feat_day_train = build_features(train_dates, df_dict, include_target=True)
    feat_day_train = feat_day_train.dropna(subset=["target"]).reset_index(drop=True)

    X_win = feat_day_train[FEATURE_COLS].values.astype(float)
    y_win = feat_day_train["target"].values.astype(float)
    ok    = np.isfinite(X_win).all(axis=1) & np.isfinite(y_win)
    X_win, y_win = X_win[ok], y_win[ok]

    feat_fc = build_features([ds], df_dict, include_target=False)
    X_fc    = feat_fc[FEATURE_COLS].values.astype(float)  # 24 rows

    if len(X_win) < 24:
        ensemble = np.full(24, np.nan)
    else:
        model = lgb.LGBMRegressor(**best_params)
        model.fit(X_win, y_win)
        ensemble = model.predict(X_fc).astype(float)

    actual_day    = df_dict.get(ds)
    actual_prices = actual_day["prices"] if actual_day is not None else np.full(24, np.nan)

    for h in range(24):
        all_results.append({
            "date":          ds,
            "hour":          h,
            "forecast_lear": float(ensemble[h]),
            "actual":        float(actual_prices[h]),
        })

    # Progress line every 30 days
    if (day_i + 1) % 30 == 0 or day_i == 0:
        valid    = np.isfinite(ensemble) & np.isfinite(actual_prices)
        day_rmse = (np.sqrt(np.mean((ensemble[valid] - actual_prices[valid]) ** 2))
                    if valid.any() else np.nan)
        print(f"  [{day_i + 1:3d}/{len(forecast_dates)}]  {ds}  "
              f"RMSE={day_rmse:.2f}  fc_mean={np.nanmean(ensemble):.1f}  "
              f"act_mean={np.nanmean(actual_prices):.1f} EUR/MWh", flush=True)

print("\nAll days done.", flush=True)

# ---------------------------------------------------------------------------
# Error statistics
# ---------------------------------------------------------------------------
hourly_df = pd.DataFrame(all_results)
hourly_df["error"] = hourly_df["forecast_lear"] - hourly_df["actual"]

overall_rmse = np.sqrt(np.nanmean(hourly_df["error"] ** 2))
overall_mae  = np.nanmean(hourly_df["error"].abs())

print(f"\n{'=' * 60}")
print(f"Overall RMSE ({len(hourly_df)} hourly obs): {overall_rmse:.3f} EUR/MWh")
print(f"Overall MAE:                               {overall_mae:.3f} EUR/MWh")

daily_rows = []
for ds in sorted(hourly_df["date"].unique()):
    sub  = hourly_df[hourly_df["date"] == ds]
    errs = sub["error"].dropna()
    if len(errs) > 0:
        daily_rows.append({
            "date":          ds,
            "rmse":          np.sqrt(np.mean(errs ** 2)),
            "mae":           np.mean(np.abs(errs)),
            "mean_forecast": sub["forecast_lear"].mean(),
            "mean_actual":   sub["actual"].mean(),
        })

daily_df = pd.DataFrame(daily_rows)
daily_df["date"] = pd.to_datetime(daily_df["date"])

print(f"\nMedian daily RMSE: {daily_df['rmse'].median():.3f} EUR/MWh")
print(f"Mean   daily RMSE: {daily_df['rmse'].mean():.3f} EUR/MWh")

print("\nPer-month mean RMSE:")
monthly = daily_df.groupby(daily_df["date"].dt.to_period("M"))["rmse"].mean()
for period, rmse in monthly.items():
    print(f"  {period}: {rmse:.2f}")

# ---------------------------------------------------------------------------
# Save CSVs
# ---------------------------------------------------------------------------
hourly_path = OUT_DIR / f"{LABEL}_hourly.csv"
daily_path  = OUT_DIR / f"{LABEL}_daily.csv"
hourly_df.to_csv(hourly_path, index=False)
daily_df.to_csv(daily_path, index=False)
print(f"\nSaved: {hourly_path}")
print(f"Saved: {daily_path}")

# ---------------------------------------------------------------------------
# Plot: daily RMSE time series
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(daily_df["date"], daily_df["rmse"],
        color="#2166ac", linewidth=1.0, marker="o", markersize=2, label="Daily RMSE")
ax.axhline(daily_df["rmse"].mean(), color="#d6604d", linewidth=1.0, linestyle="--",
           label=f"Mean RMSE = {daily_df['rmse'].mean():.2f} EUR/MWh")
ax.axhline(daily_df["rmse"].median(), color="#4dac26", linewidth=1.0, linestyle=":",
           label=f"Median RMSE = {daily_df['rmse'].median():.2f} EUR/MWh")
ax.set_ylabel("RMSE (EUR/MWh)", fontsize=9)
ax.set_title(
    f"Case B LightGBM — Daily RMSE  |  Hold-out year {HOLD_OUT_YEAR}\n"
    f"{len(FEATURE_COLS)} features  |  {WINDOW_DAYS}-day rolling window  |  {LABEL}",
    fontsize=10,
)
ax.legend(fontsize=9, frameon=False)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
ax.tick_params(axis="x", which="major", labelsize=8)
ax.grid(axis="y", linewidth=0.4, alpha=0.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
plot_path = OUT_DIR / f"{LABEL}_summary.png"
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {plot_path}")

# Per-hour RMSE plot
hour_rmse = (
    hourly_df.groupby("hour")
    .apply(lambda g: np.sqrt(np.nanmean(g["error"] ** 2)), include_groups=False)
    .rename("rmse_h")
)
fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.bar(hour_rmse.index, hour_rmse.values, color="#2166ac", alpha=0.8)
ax2.axhline(overall_rmse, color="#d6604d", linewidth=1.2, linestyle="--",
            label=f"Overall RMSE = {overall_rmse:.2f}")
ax2.set_xlabel("Hour of day (UTC)", fontsize=9)
ax2.set_ylabel("RMSE (EUR/MWh)", fontsize=9)
ax2.set_title(f"Case B LightGBM — Per-hour RMSE  |  {HOLD_OUT_YEAR}", fontsize=10)
ax2.legend(fontsize=9, frameon=False)
ax2.grid(axis="y", linewidth=0.4, alpha=0.5)
ax2.spines[["top", "right"]].set_visible(False)
fig2.tight_layout()
plot2_path = OUT_DIR / f"{LABEL}_rmse_by_hour.png"
fig2.savefig(plot2_path, dpi=150, bbox_inches="tight")
plt.close(fig2)
hour_rmse.to_csv(OUT_DIR / f"{LABEL}_rmse_by_hour.csv")
print(f"Saved: {plot2_path}")

print("\nDone.", flush=True)
