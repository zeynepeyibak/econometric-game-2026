"""
LightGBM forecast for 2024-01-01 to 2024-01-07, as a comparison to LEAR.

Unlike LEAR (one model per hour), this is a single model for all 24 hours
with the hour as a feature. Trained on data up to 2023-12-31.

Features (one row per day x hour):
  calendar        hour, day of week, month, day of year, holiday, July and
                  Dec 18-31 dummies
  prices          same-hour price at d-1, d-2, d-3, d-7
  load/wind/solar same-hour values at d, d-1, d-7
  energy prices   coal, EUA, gas, oil at d-1 and d-7
  macro           EuroStoxx50, MSCI EU, USD/EUR, VIX at d-1 and d-7
  flows           same-hour net flow to each neighbour at d-1 and d-7

As in LEAR, day 1 uses the published load/wind/solar forecasts and days 2-7
use the seasonal stand-ins from lear/core.fit_structural_models, and each
day's forecast feeds the price lags of the following days.

Hyperparameters are tuned with Optuna (40 trials). The score is the mean
weekly RMSE over three validation half-years (H1 of 2021, 2022, 2023), each
time training on everything before.

Run:     python predict.py      (from models/lgbm/; needs lightgbm and optuna)
Input:   data/prepared/lear_dataset.csv
Output:  results/lear/<LABEL>_forecasts_validation_2024w1.csv
         The forecast column is called forecast_lear on purpose, so the LEAR
         comparison plots in plots/lear/ can read this file unchanged.
"""

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from pathlib import Path

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Shared helpers from models/lear/core.py
# ---------------------------------------------------------------------------
LEAR_DIR = Path(__file__).parent.parent / "lear"
sys.path.insert(0, str(LEAR_DIR))
from core import (
    load_df_dict,
    fit_structural_models,
    _DK_HOLIDAYS,
    FLOW_CHANNELS,
)

ROOT    = Path(__file__).parent.parent.parent.parent
DATA    = ROOT / "data/prepared/lear_dataset.csv"
OUT_DIR = ROOT / "results/lear"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRAIN_CUTOFF   = "2023-12-31"
FORECAST_DATES = pd.date_range("2024-01-01", "2024-01-07", freq="D")
LABEL          = "lgbm_structural_splitren_holidays_energy_macro_daily_trans_ch"

N_OPTUNA_TRIALS = 40

# Validation periods (start, end). Each fold trains on everything before start.
CV_FOLDS = [
    ("2021-01-01", "2021-06-30"),
    ("2022-01-01", "2022-06-30"),
    ("2023-01-01", "2023-06-30"),
]

# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------
def _get_price(df_dict: dict, ds: str, h: int, forecasted_prices: dict) -> float:
    # Our own forecast if we have one for that day, otherwise the observed price
    if ds in forecasted_prices:
        return float(forecasted_prices[ds][h])
    rec = df_dict.get(ds)
    return float(rec["prices"][h]) if rec is not None else np.nan


def _get_exog(df_dict: dict, ds: str, h: int, exog_override: dict) -> tuple:
    # Structural stand-in if there is one for that day, otherwise the published forecast
    if ds in exog_override:
        ov = exog_override[ds]
        return float(ov["load"][h]), float(ov["wind"][h]), float(ov["solar"][h])
    rec = df_dict.get(ds)
    if rec is None:
        return np.nan, np.nan, np.nan
    return float(rec["load"][h]), float(rec["wind"][h]), float(rec["solar"][h])


def build_features(
    dates: list,
    df_dict: dict,
    forecasted_prices: dict,
    exog_override: dict,
    include_target: bool = True,
) -> pd.DataFrame:
    """Feature table with 24 rows per date (hour 0-23, in that order).

    Lags that fall outside the data are left as NaN; LightGBM handles them.
    """
    rows = []
    for ds in dates:
        d   = pd.Timestamp(ds)
        rec = df_dict.get(ds)

        for h in range(24):
            row: dict = {}

            row["hour"]       = h
            row["dow"]        = d.dayofweek
            row["month"]      = d.month
            row["doy"]        = d.dayofyear
            row["is_holiday"] = float(d.date() in _DK_HOLIDAYS)
            row["is_summer"]  = float(d.month == 7)
            row["is_winter"]  = float(d.month == 12 and d.day >= 18)

            for lag in [1, 2, 3, 7]:
                lag_ds = (d - pd.Timedelta(days=lag)).strftime("%Y-%m-%d")
                row[f"price_lag{lag}d"] = _get_price(df_dict, lag_ds, h, forecasted_prices)

            for lag in [0, 1, 7]:
                lag_ds = (d - pd.Timedelta(days=lag)).strftime("%Y-%m-%d")
                lv, wv, sv = _get_exog(df_dict, lag_ds, h, exog_override)
                row[f"load_lag{lag}d"]  = lv
                row[f"wind_lag{lag}d"]  = wv
                row[f"solar_lag{lag}d"] = sv

            for lag in [1, 7]:
                lag_ds  = (d - pd.Timedelta(days=lag)).strftime("%Y-%m-%d")
                lag_rec = df_dict.get(lag_ds)
                if lag_rec is not None:
                    en = lag_rec["energy"]
                    row[f"coal_lag{lag}d"] = float(en[0])
                    row[f"eua_lag{lag}d"]  = float(en[1])
                    row[f"gas_lag{lag}d"]  = float(en[2])
                    row[f"oil_lag{lag}d"]  = float(en[3])
                else:
                    for nm in ["coal", "eua", "gas", "oil"]:
                        row[f"{nm}_lag{lag}d"] = np.nan

            for lag in [1, 7]:
                lag_ds  = (d - pd.Timedelta(days=lag)).strftime("%Y-%m-%d")
                lag_rec = df_dict.get(lag_ds)
                if lag_rec is not None:
                    md = lag_rec["macro_daily"]
                    row[f"stoxx_lag{lag}d"]  = float(md[0])
                    row[f"msci_lag{lag}d"]   = float(md[1])
                    row[f"usdeur_lag{lag}d"] = float(md[2])
                    row[f"vix_lag{lag}d"]    = float(md[3])
                else:
                    for nm in ["stoxx", "msci", "usdeur", "vix"]:
                        row[f"{nm}_lag{lag}d"] = np.nan

            for lag in [1, 7]:
                lag_ds  = (d - pd.Timedelta(days=lag)).strftime("%Y-%m-%d")
                lag_rec = df_dict.get(lag_ds)
                if lag_rec is not None:
                    for ch in FLOW_CHANNELS:
                        row[f"flow_{ch}_lag{lag}d"] = float(lag_rec["flow_ch"][ch][h])
                else:
                    for ch in FLOW_CHANNELS:
                        row[f"flow_{ch}_lag{lag}d"] = np.nan

            if include_target and rec is not None:
                row["target"] = float(rec["prices"][h])
            else:
                row["target"] = np.nan

            rows.append(row)

    return pd.DataFrame(rows)


def weekly_rmse(dates_in_fold: list, preds: np.ndarray, actuals: np.ndarray) -> float:
    """RMSE within each ISO week, averaged over the weeks of a fold.

    preds/actuals must be ordered by date and hour, 24 values per date.
    """
    weeks = [pd.Timestamp(d).isocalendar()[1] * 10000 + pd.Timestamp(d).year
             for d in dates_in_fold for _ in range(24)]
    weeks = np.array(weeks)
    rmses = []
    for w in np.unique(weeks):
        mask = weeks == w
        err  = preds[mask] - actuals[mask]
        if np.isfinite(err).any():
            rmses.append(np.sqrt(np.nanmean(err ** 2)))
    return float(np.mean(rmses)) if rmses else np.inf


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading lear_dataset.csv...", flush=True)
df_dict = load_df_dict(DATA)
all_dates_sorted = sorted(d for d in df_dict if d <= TRAIN_CUTOFF)
print(f"  {len(all_dates_sorted)} training days ≤ {TRAIN_CUTOFF}", flush=True)

# ---------------------------------------------------------------------------
# Stand-ins for the exogenous forecasts on days 2-7 of the forecast week
# ---------------------------------------------------------------------------
print("\nFitting structural models (Section 4.1)...", flush=True)
train_ds_struct = [d for d in all_dates_sorted if d >= "2018-01-01"]
exog_override = fit_structural_models(
    df_dict, train_ds_struct, FORECAST_DATES, use_split_renewables=True
)
print(f"  Structural overrides for: {list(exog_override.keys())}", flush=True)

# ---------------------------------------------------------------------------
# Training features (built once, reused by every CV fold and the final fit)
# ---------------------------------------------------------------------------
print("\nBuilding full feature matrix...", flush=True)
all_feat_dates = [d for d in all_dates_sorted if d >= "2018-01-09"]  # first day with all lags
feat_df = build_features(
    all_feat_dates, df_dict,
    forecasted_prices={}, exog_override={}, include_target=True,
)
# Attach each row's date before dropping rows, so days with missing prices
# cannot shift the dates of the rows after them
feat_df["_date"] = np.repeat(all_feat_dates, 24)
feat_df = feat_df.dropna(subset=["target"]).reset_index(drop=True)

FEATURE_COLS = [c for c in feat_df.columns if c not in ("target", "_date")]
print(f"  {len(feat_df):,} rows  |  {len(FEATURE_COLS)} features", flush=True)

# ---------------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------------
def objective(trial: optuna.Trial) -> float:
    params = dict(
        n_estimators      = trial.suggest_int("n_estimators", 100, 2000),
        learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        num_leaves        = trial.suggest_int("num_leaves", 15, 255),
        max_depth         = trial.suggest_int("max_depth", 3, 12),
        min_child_samples = trial.suggest_int("min_child_samples", 10, 100),
        subsample         = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha         = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        random_state      = 42,
        verbose           = -1,
    )

    fold_scores = []
    for val_start, val_end in CV_FOLDS:
        train_mask = feat_df["_date"] < val_start
        val_mask   = (feat_df["_date"] >= val_start) & (feat_df["_date"] <= val_end)

        X_tr = feat_df.loc[train_mask, FEATURE_COLS].values
        y_tr = feat_df.loc[train_mask, "target"].values
        X_vl = feat_df.loc[val_mask,   FEATURE_COLS].values
        y_vl = feat_df.loc[val_mask,   "target"].values

        if len(X_tr) < 100 or len(X_vl) < 24:
            return np.inf

        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr, y_tr)
        preds = m.predict(X_vl)

        val_dates = feat_df.loc[val_mask, "_date"].tolist()
        # weekly_rmse expects each date once (it repeats them 24 times itself)
        unique_val_dates = sorted(set(val_dates))
        fold_scores.append(weekly_rmse(unique_val_dates, preds, y_vl))

    return float(np.mean(fold_scores))


print(f"\nRunning Optuna ({N_OPTUNA_TRIALS} trials, 3 CV folds each)...", flush=True)
study = optuna.create_study(direction="minimize",
                            sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=N_OPTUNA_TRIALS,
               show_progress_bar=True)

best = study.best_params
print(f"\nBest trial: mean weekly RMSE = {study.best_value:.3f} EUR/MWh", flush=True)
print("Best params:", flush=True)
for k, v in best.items():
    print(f"  {k}: {v}", flush=True)

# ---------------------------------------------------------------------------
# Final model on all training data
# ---------------------------------------------------------------------------
print("\nRetraining on all data with best params...", flush=True)
final_params = {**best, "random_state": 42, "verbose": -1}
final_model = lgb.LGBMRegressor(**final_params)
X_all = feat_df[FEATURE_COLS].values
y_all = feat_df["target"].values
final_model.fit(X_all, y_all)

# ---------------------------------------------------------------------------
# Forecast the week day by day; each day's prices feed the next days' lags
# ---------------------------------------------------------------------------
print(
    f"\nForecasting {FORECAST_DATES[0].date()} → {FORECAST_DATES[-1].date()}...\n",
    flush=True,
)

forecasted_prices: dict = {}
results = []

for day_i, forecast_date in enumerate(FORECAST_DATES):
    ds = forecast_date.strftime("%Y-%m-%d")

    feat_day = build_features(
        [ds], df_dict,
        forecasted_prices=forecasted_prices,
        exog_override=exog_override,
        include_target=False,
    )
    X_day = feat_day[FEATURE_COLS].values
    preds = final_model.predict(X_day)

    actual_day    = df_dict.get(ds)
    actual_prices = actual_day["prices"].copy() if actual_day is not None else np.full(24, np.nan)

    forecasted_prices[ds] = np.where(np.isfinite(preds), preds, actual_prices)

    for h in range(24):
        results.append({
            "date":          ds,
            "hour":          h,
            "forecast_lear": float(preds[h]),
            "actual":        float(actual_prices[h]),
        })

    rmse_day = np.sqrt(np.nanmean((preds - actual_prices) ** 2))
    print(
        f"[{day_i+1}/7]  {ds}  mean={np.nanmean(preds):.1f}  RMSE={rmse_day:.2f} EUR/MWh",
        flush=True,
    )

# ---------------------------------------------------------------------------
# Evaluation and output
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df["error"] = results_df["forecast_lear"] - results_df["actual"]

rmse_week = np.sqrt(np.nanmean(results_df["error"] ** 2))
mae_week  = np.nanmean(results_df["error"].abs())

print("=" * 60, flush=True)
print(f"RMSE_week (168 h): {rmse_week:.3f} EUR/MWh", flush=True)
print(f"MAE_week  (168 h): {mae_week:.3f} EUR/MWh", flush=True)
print("\nPer-day:", flush=True)
for ds in sorted(results_df["date"].unique()):
    sub = results_df[results_df["date"] == ds]
    print(
        f"  {ds}: fc={sub['forecast_lear'].mean():.2f}  "
        f"actual={sub['actual'].mean():.2f}  "
        f"RMSE={np.sqrt(np.nanmean(sub['error']**2)):.2f}",
        flush=True,
    )

suffix   = f"validation_{FORECAST_DATES[0].strftime('%Y')}w{FORECAST_DATES[0].isocalendar()[1]}"
out_path = OUT_DIR / f"{LABEL}_forecasts_{suffix}.csv"
results_df[["date", "hour", "forecast_lear", "actual"]].to_csv(out_path, index=False)
print(f"\nSaved to {out_path}", flush=True)
print("Done.", flush=True)
