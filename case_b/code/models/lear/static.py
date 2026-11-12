"""
Case B: LEAR estimated once, then used for a whole hold-out year.

The 24 hourly LASSO models are fitted a single time on all data from
2018-01-01 to the end of the year before HOLD_OUT_YEAR (e.g. 2018-2019 for
2020), and every day of the hold-out year is forecast with those fixed
coefficients. Price lags are the observed prices. Outputs have the same
format as rolling.py, so the two can be compared directly.

Run:     python static.py      (from models/lear/)
Input:   data/prepared/DK1/lear_dataset.csv
Output:  results/lear/<label>_{hourly,daily,coefs,rmse_by_hour}.csv,
         <label>_summary.png, <label>_rmse_by_hour.png
         <label> = static_<year>_ + one suffix per active flag
"""

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import Lasso, LassoLarsIC
from core import ModelConfig, load_df_dict, build_feature_row, REF_DATE  # noqa: E402
from config import HOLD_OUT_YEAR, CONFIG

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT    = Path(__file__).parent.parent.parent.parent
DATA    = ROOT / "data/prepared/DK1/lear_dataset.csv"
OUT_DIR = ROOT / "results/lear"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Output label: static_<year> + one suffix per active flag
_suffixes = ["static", str(HOLD_OUT_YEAR)]
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

print(f"Case B LEAR — Static forecast  year={HOLD_OUT_YEAR}  n_features={CONFIG.n_features}",
      flush=True)
print(f"  Label: {LABEL}", flush=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("\nLoading lear_dataset.csv...", flush=True)
df_dict = load_df_dict(DATA)
all_dates_sorted = sorted(df_dict.keys())
print(f"  {len(all_dates_sorted)} days loaded", flush=True)

# ---------------------------------------------------------------------------
# Container for the fitted models
# ---------------------------------------------------------------------------
@dataclass
class StaticModel:
    """The 24 fitted hourly models plus the medians/MADs needed to apply the
    same asinh transform to new data."""
    lasso_models: list          # list[Lasso | None], one per hour
    X_medians:    np.ndarray    # shape (n_features,)
    X_mads:       np.ndarray    # shape (n_features,)
    y_meds:       np.ndarray    # shape (24,)
    y_mads:       np.ndarray    # shape (24,)
    n_train:      int


# ---------------------------------------------------------------------------
# Step 1: fit the 24 hourly models once
# ---------------------------------------------------------------------------
train_start = "2018-01-01"
train_end   = f"{HOLD_OUT_YEAR - 1}-12-31"
train_dates = [d for d in all_dates_sorted if train_start <= d <= train_end
               and (pd.Timestamp(d) - REF_DATE).days >= 7]

print(f"\nBuilding training features on {train_start} → {train_end}  "
      f"({len(train_dates)} days)...", flush=True)

X_rows, y_rows = [], {h: [] for h in range(24)}
for td in train_dates:
    xr = build_feature_row(td, df_dict, {}, df_dict[td]["dow"], CONFIG)
    if not np.isfinite(xr).all():
        continue
    X_rows.append(xr)
    for h in range(24):
        y_rows[h].append(df_dict[td]["prices"][h])

X_train_raw = np.array(X_rows, dtype=float)
print(f"  Training matrix: {X_train_raw.shape}", flush=True)

# asinh transform of the regressors, using training medians/MADs (as in core.asinh_transform)
X_medians = np.nanmedian(X_train_raw, axis=0)
X_mads    = np.nanmedian(np.abs(X_train_raw - X_medians), axis=0)
X_mads[X_mads < 1e-10] = 1.0
X_train_t = np.arcsinh((X_train_raw - X_medians) / X_mads)

# One LASSO per hour; same steps as core.fit_hour, but the fitted models are kept
print(f"  Fitting {CONFIG.n_features}-dim LASSO for 24 hours...", flush=True)
lasso_models = []
y_meds = np.zeros(24)
y_mads = np.zeros(24)

for h in range(24):
    y_train = np.array(y_rows[h], dtype=float)
    valid   = np.isfinite(y_train)

    if valid.sum() < 20:
        lasso_models.append(None)
        continue

    y_med = float(np.nanmedian(y_train[valid]))
    y_mad = float(np.nanmedian(np.abs(y_train[valid] - y_med)))
    if y_mad < 1e-10:
        y_mad = 1.0
    y_meds[h] = y_med
    y_mads[h] = y_mad
    y_t = np.arcsinh((y_train[valid] - y_med) / y_mad)

    Xt = X_train_t[valid]
    try:
        lars = LassoLarsIC(criterion="aic", fit_intercept=False)
        lars.fit(Xt, y_t)
    except Exception:
        noise_var = max(float(np.var(y_t)), 1e-4)
        lars = LassoLarsIC(criterion="aic", fit_intercept=False,
                           noise_variance=noise_var)
        lars.fit(Xt, y_t)

    alpha_opt = max(float(lars.alpha_), 1e-8)
    lasso = Lasso(alpha=alpha_opt, fit_intercept=False, max_iter=20000, tol=1e-5)
    lasso.fit(Xt, y_t)
    lasso_models.append(lasso)

    if (h + 1) % 6 == 0:
        print(f"    Hour {h:02d} done  alpha={alpha_opt:.4g}", flush=True)

static_model = StaticModel(
    lasso_models=lasso_models,
    X_medians=X_medians,
    X_mads=X_mads,
    y_meds=y_meds,
    y_mads=y_mads,
    n_train=len(X_rows),
)
print(f"\nFit complete. n_train={static_model.n_train}", flush=True)

# Save lambda and coefficients, one row per hour
coef_rows = []
for h, m in enumerate(lasso_models):
    coefs = m.coef_ if m is not None else np.zeros(CONFIG.n_features)
    alpha = float(m.alpha) if m is not None else np.nan
    row = {"hour": h, "alpha": alpha, "n_nonzero": int(np.count_nonzero(coefs))}
    for i, v in enumerate(coefs):
        row[f"coef_{i:04d}"] = float(v)
    coef_rows.append(row)
coef_path = OUT_DIR / f"{LABEL}_coefs.csv"
pd.DataFrame(coef_rows).to_csv(coef_path, index=False)
print(f"Saved: {coef_path}  (24 rows × {CONFIG.n_features} coefs + alpha/n_nonzero)",
      flush=True)


# ---------------------------------------------------------------------------
# Forecasting with the fixed coefficients
# ---------------------------------------------------------------------------
def predict_static(forecast_date: pd.Timestamp,
                   model: StaticModel,
                   df_dict: dict,
                   config: ModelConfig) -> np.ndarray:
    """24 price forecasts for one day from the stored models (NaN if any
    regressor is missing)."""
    dow_vec = np.zeros(7)
    dow_vec[forecast_date.dayofweek] = 1
    ds = forecast_date.strftime("%Y-%m-%d")

    x_raw = build_feature_row(ds, df_dict, {}, dow_vec, config)
    if not np.isfinite(x_raw).all():
        return np.full(24, np.nan)

    x_t = np.arcsinh((x_raw - model.X_medians) / model.X_mads)

    preds = np.full(24, np.nan)
    for h in range(24):
        m = model.lasso_models[h]
        if m is None:
            continue
        y_hat_t = float(m.predict(x_t.reshape(1, -1))[0])
        preds[h] = float(np.sinh(y_hat_t) * model.y_mads[h] + model.y_meds[h])

    return preds


# ---------------------------------------------------------------------------
# Step 2: forecast every day of the hold-out year
# ---------------------------------------------------------------------------
forecast_dates = pd.date_range(f"{HOLD_OUT_YEAR}-01-01", f"{HOLD_OUT_YEAR}-12-31", freq="D")
all_results    = []

print(f"\nForecasting {forecast_dates[0].date()} → {forecast_dates[-1].date()} "
      f"({len(forecast_dates)} days) with fixed coefficients...\n", flush=True)

for day_i, forecast_date in enumerate(forecast_dates):
    ds = forecast_date.strftime("%Y-%m-%d")

    ensemble   = predict_static(forecast_date, static_model, df_dict, CONFIG)
    actual_day = df_dict.get(ds)

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
        day_rmse = (np.sqrt(np.mean((ensemble[valid] - actual_prices[valid])**2))
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
            "rmse":          np.sqrt(np.mean(errs**2)),
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
    f"Case B LEAR Static — Daily RMSE  |  Hold-out year {HOLD_OUT_YEAR}\n"
    f"{CONFIG.n_features} features  |  Trained on {train_start} → {train_end}  |  {LABEL}",
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
ax2.set_title(f"Case B LEAR Static — Per-hour RMSE  |  {HOLD_OUT_YEAR}", fontsize=10)
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
