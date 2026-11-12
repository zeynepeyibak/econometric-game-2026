"""
LEAR forecast for a single week (by default the first week of 2024).

Day 1 is forecast with CONFIG_FULL, which can use the published day-ahead
load and renewable forecasts. For days 2-7 those forecasts do not exist yet,
so CONFIG_AR is used instead. Two set-ups we used for days 2-7:
  - AR only:     ModelConfig(use_exogenous=False, use_structural=False)
                 with BASE_LABEL = "lear_baseline"
  - structural:  ModelConfig(use_exogenous=True, use_structural=True)
                 with BASE_LABEL = "lear_structural"; exogenous inputs are
                 then predicted by core.fit_structural_models.
Each day's forecast becomes the lagged price for the following days.

If the target week is in the data, the script also reports RMSE/MAE against
the realised prices.

Run:     python predict.py [--plot-windows]      (from models/lear/)
         --plot-windows adds the four single-window forecasts to the figure
Input:   data/prepared/lear_dataset.csv
Output:  results/lear/<label>_forecasts_validation_<year>w<week>.csv
         results/lear/<label>_rmse_per_hour_validation_<year>w<week>.csv
         results/lear/<label>_forecast_validation_<year>w<week>[_windows].png
"""

import sys
import argparse
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

parser = argparse.ArgumentParser(description="LEAR single-week forecast")
parser.add_argument(
    "--plot-windows",
    action="store_true",
    help="Also plot individual calibration-window forecasts alongside the ensemble",
)
args = parser.parse_args()

ROOT = Path(__file__).parent.parent.parent.parent
DATA = ROOT / "data/prepared/lear_dataset.csv"
OUT_DIR = ROOT / "results/lear"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRAIN_CUTOFF = "2023-12-31"
FORECAST_DATES = pd.date_range("2024-01-01", "2024-01-07", freq="D")
BASE_LABEL = "lear_structural"  # file name prefix; flag suffixes are added below

# Day 1: published ENTSO-E day-ahead load/renewable forecasts are available
CONFIG_FULL = ModelConfig(
    use_exogenous=True,
    use_structural=True,
    use_split_renewables=True,
    use_holidays=True,
    use_transmission=True,
    use_transmission_channels=True,
    use_energy=True,
    use_macro_daily=True,
    train_cutoff=TRAIN_CUTOFF,
)

# Days 2-7: either AR only (use_exogenous=False, use_structural=False) or
# structural stand-ins for the exogenous inputs (both True), see header
CONFIG_AR = ModelConfig(
    use_exogenous=True,
    use_structural=True,
    use_split_renewables=True,
    use_holidays=True,
    train_cutoff=TRAIN_CUTOFF,
)

# Output label = BASE_LABEL + one suffix per active flag in CONFIG_FULL
_flag_suffixes = []
if CONFIG_FULL.use_holidays:
    _flag_suffixes.append("holidays")
if CONFIG_FULL.use_split_renewables:
    _flag_suffixes.append("splitren")
if CONFIG_FULL.use_energy:
    _flag_suffixes.append("energy")
if CONFIG_FULL.use_macro_daily:
    _flag_suffixes.append("macro_daily")
if CONFIG_FULL.use_transmission:
    _flag_suffixes.append(
        "trans_ch" if CONFIG_FULL.use_transmission_channels else "trans_agg"
    )
LABEL = BASE_LABEL + ("_" + "_".join(_flag_suffixes) if _flag_suffixes else "")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading lear_dataset.csv...", flush=True)
df_dict = load_df_dict(DATA)
all_dates_sorted = sorted(d for d in df_dict if d <= TRAIN_CUTOFF)
print(f"  {len(all_dates_sorted)} training days ≤ {TRAIN_CUTOFF}", flush=True)

# ---------------------------------------------------------------------------
# Stand-ins for the exogenous forecasts on days 2-7 (fitted once, before the loop)
# ---------------------------------------------------------------------------
exog_override = {}
if CONFIG_AR.use_structural and CONFIG_AR.use_exogenous:
    print(f"\nFitting structural models on 2018-01-01 → {TRAIN_CUTOFF}...", flush=True)
    train_ds = [d for d in all_dates_sorted if d >= "2018-01-01"]
    exog_override = fit_structural_models(
        df_dict,
        train_ds,
        FORECAST_DATES,
        use_split_renewables=CONFIG_AR.use_split_renewables,
    )
    print(f"  Overrides for: {list(exog_override.keys())}", flush=True)

# ---------------------------------------------------------------------------
# Forecast the week day by day
# ---------------------------------------------------------------------------
print(
    f"\nForecasting {FORECAST_DATES[0].date()} → {FORECAST_DATES[-1].date()}...\n",
    flush=True,
)

forecasted_prices = {}
results = []

for day_i, forecast_date in enumerate(FORECAST_DATES):
    ds = forecast_date.strftime("%Y-%m-%d")
    config = CONFIG_FULL if day_i == 0 else CONFIG_AR
    print(f"[{day_i + 1}/7]  {ds}  ({config.n_features} params)", flush=True)

    day_res = forecast_one_day(
        forecast_date,
        config,
        all_dates_sorted,
        df_dict,
        forecasted_prices,
        exog_override,
    )
    ensemble = day_res["ensemble"]

    # Keep the forecast as the lagged price for later days. Hours that
    # failed (NaN) fall back to the observed price if the day is in the data.
    actual_day = df_dict.get(ds)
    if np.isfinite(ensemble).all():
        forecasted_prices[ds] = ensemble.copy()
    elif actual_day is not None:
        forecasted_prices[ds] = np.where(
            np.isfinite(ensemble), ensemble, actual_day["prices"]
        )
    else:
        forecasted_prices[ds] = ensemble.copy()

    for h in range(24):
        row = {"date": ds, "hour": h, "forecast_lear": ensemble[h]}
        for wn in config.calibration_windows:
            row[f"fc_{wn}"] = day_res[wn][h]
        results.append(row)

    print(
        f"  Ensemble: mean={np.nanmean(ensemble):.2f}  "
        f"min={np.nanmin(ensemble):.1f}  max={np.nanmax(ensemble):.1f} EUR/MWh\n",
        flush=True,
    )

# ---------------------------------------------------------------------------
# Compare with realised prices (NaN if the week is outside the data)
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df["actual"] = results_df.apply(
    lambda r: float(
        df_dict.get(r["date"], {"prices": np.full(24, np.nan)})["prices"][r["hour"]]
    ),
    axis=1,
)
results_df["error"] = results_df["forecast_lear"] - results_df["actual"]

rmse_week = np.sqrt(np.nanmean(results_df["error"] ** 2))
mae_week = np.nanmean(results_df["error"].abs())

print("=" * 60)
print(f"RMSE_week (168 h): {rmse_week:.3f} EUR/MWh")
print(f"MAE_week  (168 h): {mae_week:.3f} EUR/MWh")
print("\nPer-day:")
for ds in sorted(results_df["date"].unique()):
    sub = results_df[results_df["date"] == ds]
    print(
        f"  {ds}: fc={sub['forecast_lear'].mean():.2f}  "
        f"actual={sub['actual'].mean():.2f}  "
        f"RMSE={np.sqrt(np.nanmean(sub['error'] ** 2)):.2f}"
    )

hour_rmse = (
    results_df.groupby("hour")
    .apply(lambda g: np.sqrt(np.nanmean(g["error"] ** 2)), include_groups=False)
    .rename("rmse_h")
)

suffix = f"validation_{FORECAST_DATES[0].strftime('%Y')}w{FORECAST_DATES[0].isocalendar()[1]}"
results_df.to_csv(OUT_DIR / f"{LABEL}_forecasts_{suffix}.csv", index=False)
hour_rmse.to_csv(OUT_DIR / f"{LABEL}_rmse_per_hour_{suffix}.csv")

# ---------------------------------------------------------------------------
# Plot: forecast vs actual (top), error by hour (bottom)
# ---------------------------------------------------------------------------
if CONFIG_AR.use_structural:
    days_2_7_desc = "Days 2–7: structural calendar model (exog approximated)"
elif CONFIG_AR.use_exogenous:
    days_2_7_desc = "Days 2–7: full model (exogenous, no structural approximation)"
else:
    days_2_7_desc = "Days 2–7: AR-only (103 params, no exogenous)"

WIN_COLORS = {"8w": "#f4a582", "12w": "#92c5de", "3y": "#a6d96a", "4y": "#d9ef8b"}

times, fc_vals, ac_vals = [], [], []
win_vals: dict = {wn: [] for wn in WIN_COLORS}

for ds in sorted(results_df["date"].unique()):
    for h in range(24):
        sub = results_df[(results_df["date"] == ds) & (results_df["hour"] == h)]
        if len(sub):
            times.append(pd.Timestamp(ds) + pd.Timedelta(hours=h))
            fc_vals.append(float(sub["forecast_lear"].iloc[0]))
            ac_vals.append(float(sub["actual"].iloc[0]))
            for wn in win_vals:
                win_vals[wn].append(float(sub[f"fc_{wn}"].iloc[0]))

times = pd.DatetimeIndex(times)
fc_vals = np.array(fc_vals)
ac_vals = np.array(ac_vals)
for wn in win_vals:
    win_vals[wn] = np.array(win_vals[wn])

fig, axes = plt.subplots(
    2, 1, figsize=(15, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)
ax1, ax2 = axes

if args.plot_windows:
    for wn, color in WIN_COLORS.items():
        ax1.plot(
            times,
            win_vals[wn],
            color=color,
            linewidth=1.1,
            alpha=0.95,
            linestyle="--",
            label=f"Window {wn}",
        )

ax1.plot(times, ac_vals, color="#2166ac", linewidth=1.4, label="Actual (realized)")
ax1.plot(
    times,
    fc_vals,
    color="#d6604d",
    linewidth=1.6 if args.plot_windows else 1.4,
    linestyle="--",
    label=f"{LABEL} ensemble (RMSE={rmse_week:.2f} €/MWh)",
)
ax1.set_ylabel("Day-ahead Price (EUR/MWh)", fontsize=9)
ax1.set_title(
    f"DK1 {LABEL} — Forecast {FORECAST_DATES[0].date()} → {FORECAST_DATES[-1].date()}\n"
    f"Day 1: full model ({CONFIG_FULL.n_features} params)  |  {days_2_7_desc}\n"
    f"Trained on data ≤ {TRAIN_CUTOFF}",
    fontsize=10,
)
ax1.legend(fontsize=9, frameon=False, ncol=2 if args.plot_windows else 1)
ax1.grid(axis="y", linewidth=0.4, alpha=0.5)
ax1.spines[["top", "right"]].set_visible(False)

ax2.bar(
    times,
    fc_vals - ac_vals,
    width=0.04,
    color="#7b3294",
    alpha=0.7,
    label="Error (forecast − actual)",
)
ax2.axhline(0, color="k", linewidth=0.6)
ax2.set_ylabel("Error (EUR/MWh)", fontsize=9)
ax2.legend(fontsize=8, frameon=False)
ax2.grid(axis="y", linewidth=0.4, alpha=0.5)
ax2.spines[["top", "right"]].set_visible(False)
ax2.xaxis.set_major_locator(mdates.DayLocator())
ax2.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b"))
ax2.tick_params(axis="x", which="major", labelsize=8)
ax2.set_xlim(times.min(), times.max())

fig.tight_layout()
plot_suffix = suffix + ("_windows" if args.plot_windows else "")
fig.savefig(
    OUT_DIR / f"{LABEL}_forecast_{plot_suffix}.png", dpi=150, bbox_inches="tight"
)
plt.close(fig)
print(f"\nSaved to {OUT_DIR}/", flush=True)
print("Done.", flush=True)
