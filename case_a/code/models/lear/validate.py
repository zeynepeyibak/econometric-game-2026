"""
Walk-forward validation of LEAR over 2022-2023, one week at a time.

For each of the ~104 non-overlapping weeks:
  - the model only sees data up to the day before the week starts;
  - the seven days are forecast in order, and each day's forecast is used as
    the lagged price for the following days (as in a real weekly forecast);
  - if CONFIG.use_structural is on, exogenous inputs for days 2-7 come from
    the seasonal models in core.fit_structural_models.

The validation starts in 2022 because the 4-year window needs four full
years of history and the data begin on 2018-01-01.

Switch variants by editing CONFIG below; the output label is built from the
flags that are on.

Run:     python validate.py          (from models/lear/)
Input:   data/prepared/lear_dataset.csv
Output:  results/lear/<label>_rolling_validation_{hourly,weekly,rmse_by_hour}.csv
         results/lear/<label>_rolling_validation_{summary,rmse_by_hour}.png
"""

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from core import (
    ModelConfig,
    load_df_dict,
    fit_structural_models,
    forecast_one_day,
)

ROOT = Path(__file__).parent.parent.parent.parent
DATA = ROOT / "data/prepared/lear_dataset.csv"
OUT_DIR = ROOT / "results/lear"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_LABEL = "lear_structural"

CONFIG = ModelConfig(
    use_exogenous  = True,
    use_structural = False,
    use_holidays   = False,
)

# Output label = BASE_LABEL + one suffix per active flag
_flag_suffixes = []
if CONFIG.use_holidays:
    _flag_suffixes.append("holidays")
if CONFIG.use_split_renewables:
    _flag_suffixes.append("splitren")
if CONFIG.use_energy:
    _flag_suffixes.append("energy")
if CONFIG.use_macro_daily:
    _flag_suffixes.append("macro_daily")
if CONFIG.use_transmission:
    _flag_suffixes.append("trans_ch" if CONFIG.use_transmission_channels else "trans_agg")

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
print("Loading lear_dataset.csv...", flush=True)
df_dict = load_df_dict(DATA)
all_dates_sorted = sorted(df_dict.keys())
print(f"  {len(all_dates_sorted)} days loaded", flush=True)


# ---------------------------------------------------------------------------
# Rolling validation loop
# ---------------------------------------------------------------------------
week_starts = pd.date_range("2022-01-01", "2023-12-25", freq="7D")
print(
    f"\nRolling validation: {len(week_starts)} weeks, "
    f"{week_starts[0].date()} → {(week_starts[-1] + pd.Timedelta(days=6)).date()}",
    flush=True,
)

label = BASE_LABEL + ("_" + "_".join(_flag_suffixes) if _flag_suffixes else "")
all_hourly = []
weekly_rows = []

for w_i, week_start in enumerate(week_starts):
    ws_str = week_start.strftime("%Y-%m-%d")
    we_str = (week_start + pd.Timedelta(days=6)).strftime("%Y-%m-%d")
    print(f"[{w_i + 1:3d}/{len(week_starts)}]  {ws_str} → {we_str}", flush=True)

    train_cutoff_str = (week_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    forecast_dates = pd.date_range(week_start, periods=7, freq="D")
    train_dates_all = [d for d in all_dates_sorted if d <= train_cutoff_str]

    # Stand-ins for the exogenous forecasts on days 2-7
    exog_override = {}
    if CONFIG.use_structural and CONFIG.use_exogenous:
        train_ds = [d for d in train_dates_all if d >= "2018-01-01"]
        exog_override = fit_structural_models(df_dict, train_ds, forecast_dates,
                                              use_split_renewables=CONFIG.use_split_renewables)

    forecasted_prices = {}
    week_results = []

    for forecast_date in forecast_dates:
        ds = forecast_date.strftime("%Y-%m-%d")

        day_res = forecast_one_day(
            forecast_date,
            CONFIG,
            train_dates_all,
            df_dict,
            forecasted_prices,
            exog_override,
        )
        ensemble = day_res["ensemble"]

        # Store the forecast so later days in the week use it as a lag.
        # If some hours failed (NaN), those hours fall back to the observed
        # price - note this lets actual prices into the lags for that week.
        actual_day = df_dict.get(ds)
        if np.isfinite(ensemble).all():
            forecasted_prices[ds] = ensemble.copy()
        elif actual_day is not None:
            forecasted_prices[ds] = np.where(
                np.isfinite(ensemble), ensemble, actual_day["prices"]
            )
        else:
            forecasted_prices[ds] = ensemble.copy()

        actual_prices = df_dict.get(ds, {"prices": np.full(24, np.nan)})["prices"]
        for h in range(24):
            row = {
                "date": ds,
                "hour": h,
                "forecast_lear": ensemble[h],
                "actual": actual_prices[h],
            }
            for wn in CONFIG.calibration_windows:
                row[f"fc_{wn}"] = day_res[wn][h]
            week_results.append(row)

    all_hourly.extend(week_results)

    errors = [
        r["forecast_lear"] - r["actual"]
        for r in week_results
        if np.isfinite(r["forecast_lear"]) and np.isfinite(r["actual"])
    ]
    if errors:
        errors = np.array(errors)
        rmse = np.sqrt(np.mean(errors**2))
        mae = np.mean(np.abs(errors))
        mean_fc = np.nanmean([r["forecast_lear"] for r in week_results])
        mean_ac = np.nanmean([r["actual"] for r in week_results])
        print(
            f"          RMSE={rmse:.2f}  MAE={mae:.2f}  "
            f"fc_mean={mean_fc:.1f}  ac_mean={mean_ac:.1f} EUR/MWh",
            flush=True,
        )
        weekly_rows.append(
            {
                "week_start": ws_str,
                "rmse": rmse,
                "mae": mae,
                "mean_forecast": mean_fc,
                "mean_actual": mean_ac,
                "n_valid": len(errors),
            }
        )
    else:
        print("          No valid forecasts this week", flush=True)

print("\nAll weeks done.", flush=True)


# ---------------------------------------------------------------------------
# Overall error statistics and CSV output
# ---------------------------------------------------------------------------
hourly_df = pd.DataFrame(all_hourly)
hourly_df["error"] = hourly_df["forecast_lear"] - hourly_df["actual"]
weekly_df = pd.DataFrame(weekly_rows)
weekly_df["week_start"] = pd.to_datetime(weekly_df["week_start"])

overall_rmse = np.sqrt(np.nanmean(hourly_df["error"] ** 2))
overall_mae = np.nanmean(hourly_df["error"].abs())

print(f"\n{'=' * 60}")
print(f"Overall RMSE ({len(hourly_df)} hourly obs): {overall_rmse:.3f} EUR/MWh")
print(f"Overall MAE:                                {overall_mae:.3f} EUR/MWh")
print(f"Median weekly RMSE: {weekly_df['rmse'].median():.3f} EUR/MWh")
print(f"Mean   weekly RMSE: {weekly_df['rmse'].mean():.3f} EUR/MWh")

hourly_df.to_csv(OUT_DIR / f"{label}_rolling_validation_hourly.csv", index=False)
weekly_df.to_csv(OUT_DIR / f"{label}_rolling_validation_weekly.csv", index=False)
print(f"\nSaved CSVs to {OUT_DIR}/", flush=True)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
print("Plotting...", flush=True)

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(
    weekly_df["week_start"],
    weekly_df["rmse"],
    color="#2166ac",
    linewidth=1.2,
    marker="o",
    markersize=3,
    label="Weekly RMSE",
)
ax.axhline(
    weekly_df["rmse"].mean(),
    color="#d6604d",
    linewidth=1.0,
    linestyle="--",
    label=f"Mean RMSE = {weekly_df['rmse'].mean():.2f} EUR/MWh",
)
ax.axhline(
    weekly_df["rmse"].median(),
    color="#4dac26",
    linewidth=1.0,
    linestyle=":",
    label=f"Median RMSE = {weekly_df['rmse'].median():.2f} EUR/MWh",
)
ax.set_ylabel("RMSE (EUR/MWh)", fontsize=9)
ax.set_title(
    f"LEAR ({label}) Rolling Walk-Forward Validation — Weekly RMSE (2022–2023)",
    fontsize=11,
)
ax.legend(fontsize=9, frameon=False)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
ax.tick_params(axis="x", which="major", labelsize=8)
ax.grid(axis="y", linewidth=0.4, alpha=0.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(
    OUT_DIR / f"{label}_rolling_validation_summary.png", dpi=150, bbox_inches="tight"
)
plt.close(fig)

hour_rmse = (
    hourly_df.groupby("hour")
    .apply(lambda g: np.sqrt(np.nanmean(g["error"] ** 2)), include_groups=False)
    .rename("rmse_h")
)

fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.bar(hour_rmse.index, hour_rmse.values, color="#2166ac", alpha=0.8)
ax2.axhline(
    overall_rmse,
    color="#d6604d",
    linewidth=1.2,
    linestyle="--",
    label=f"Overall RMSE = {overall_rmse:.2f}",
)
ax2.set_xlabel("Hour of day (UTC)", fontsize=9)
ax2.set_ylabel("RMSE (EUR/MWh)", fontsize=9)
ax2.set_title(
    f"LEAR ({label}) Per-hour RMSE — Rolling Validation 2022–2023", fontsize=11
)
ax2.legend(fontsize=9, frameon=False)
ax2.grid(axis="y", linewidth=0.4, alpha=0.5)
ax2.spines[["top", "right"]].set_visible(False)
fig2.tight_layout()
fig2.savefig(
    OUT_DIR / f"{label}_rolling_validation_rmse_by_hour.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close(fig2)

hour_rmse.to_csv(OUT_DIR / f"{label}_rolling_validation_rmse_by_hour.csv")

print(f"Saved plots to {OUT_DIR}/", flush=True)
print("\nDone.", flush=True)
