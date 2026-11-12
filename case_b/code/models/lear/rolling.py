"""
Case B: day-by-day LEAR forecast for a whole hold-out year.

For every day d of HOLD_OUT_YEAR the model is re-estimated on the
CONFIG.window_days (2 years) before d and used to forecast d. Price lags are
always the observed prices. The year and the regressor blocks are set in
config.py.

Run:     python rolling.py      (from models/lear/)
Input:   data/prepared/DK1/lear_dataset.csv
Output:  results/lear/<label>_hourly.csv        forecast, actual, error by hour
         results/lear/<label>_daily.csv         daily RMSE/MAE and means
         results/lear/<label>_coefs.csv         lambda and all LASSO coefficients,
                                                per day and hour
         results/lear/<label>_rmse_by_hour.{csv,png}, <label>_summary.png
         <label> = rolling_<year>_ + one suffix per active flag
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from core import load_df_dict, forecast_one_day
from config import HOLD_OUT_YEAR, CONFIG

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT    = Path(__file__).parent.parent.parent.parent
DATA    = ROOT / "data/prepared/DK1/lear_dataset.csv"
OUT_DIR = ROOT / "results/lear"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Output label: rolling_<year> + one suffix per active flag
_suffixes = ["rolling", str(HOLD_OUT_YEAR)]
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

print(f"Case B LEAR — Rolling forecast  year={HOLD_OUT_YEAR}  n_features={CONFIG.n_features}",
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
# Forecast each day of the year
# ---------------------------------------------------------------------------
forecast_dates = pd.date_range(f"{HOLD_OUT_YEAR}-01-01", f"{HOLD_OUT_YEAR}-12-31", freq="D")
all_results    = []
all_coef_rows  = []   # one row per day and hour: lambda, # non-zero, coefficients

print(f"\nForecasting {forecast_dates[0].date()} → {forecast_dates[-1].date()} "
      f"({len(forecast_dates)} days)...\n", flush=True)

for day_i, forecast_date in enumerate(forecast_dates):
    ds = forecast_date.strftime("%Y-%m-%d")

    # Training window: the window_days days before the forecast day
    cutoff_end   = (forecast_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    cutoff_start = (forecast_date - pd.Timedelta(days=CONFIG.window_days)).strftime("%Y-%m-%d")
    train_dates  = [d for d in all_dates_sorted if cutoff_start <= d <= cutoff_end]

    day_res  = forecast_one_day(forecast_date, CONFIG, train_dates, df_dict, {})
    ensemble = day_res["ensemble"]
    coefs    = day_res["coefs"]    # shape (24, n_features)
    alphas   = day_res["alphas"]   # shape (24,)

    actual_day    = df_dict.get(ds)
    actual_prices = actual_day["prices"] if actual_day is not None else np.full(24, np.nan)
    for h in range(24):
        all_results.append({
            "date":          ds,
            "hour":          h,
            "forecast_lear": float(ensemble[h]),
            "actual":        float(actual_prices[h]),
        })
        row = {"date": ds, "hour": h, "alpha": float(alphas[h]),
               "n_nonzero": int(np.count_nonzero(coefs[h]))}
        for i, v in enumerate(coefs[h]):
            row[f"coef_{i:04d}"] = float(v)
        all_coef_rows.append(row)

    # Progress line every 30 days
    if (day_i + 1) % 30 == 0 or day_i == 0:
        day_fc   = ensemble
        day_act  = actual_prices
        valid    = np.isfinite(day_fc) & np.isfinite(day_act)
        day_rmse = np.sqrt(np.mean((day_fc[valid] - day_act[valid])**2)) if valid.any() else np.nan
        print(f"  [{day_i + 1:3d}/{len(forecast_dates)}]  {ds}  "
              f"RMSE={day_rmse:.2f}  fc_mean={np.nanmean(day_fc):.1f}  "
              f"act_mean={np.nanmean(day_act):.1f} EUR/MWh", flush=True)

print("\nAll days done.", flush=True)

# ---------------------------------------------------------------------------
# Save coefficients
# ---------------------------------------------------------------------------
coef_path = OUT_DIR / f"{LABEL}_coefs.csv"
pd.DataFrame(all_coef_rows).to_csv(coef_path, index=False)
print(f"Saved: {coef_path}  "
      f"({len(all_coef_rows)} rows × {CONFIG.n_features} coefs + alpha/n_nonzero)",
      flush=True)

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

# Daily RMSE/MAE
daily_rows = []
for ds in sorted(hourly_df["date"].unique()):
    sub    = hourly_df[hourly_df["date"] == ds]
    errs   = sub["error"].dropna()
    if len(errs) > 0:
        daily_rows.append({
            "date":         ds,
            "rmse":         np.sqrt(np.mean(errs**2)),
            "mae":          np.mean(np.abs(errs)),
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
    f"Case B LEAR Rolling — Daily RMSE  |  Hold-out year {HOLD_OUT_YEAR}\n"
    f"{CONFIG.n_features} features  |  2-year rolling window  |  {LABEL}",
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
ax2.set_title(f"Case B LEAR Rolling — Per-hour RMSE  |  {HOLD_OUT_YEAR}", fontsize=10)
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
