"""
Diagnostic figures for the 2022-2023 LEAR walk-forward validation
(output of models/lear/validate.py):
  1. weekly RMSE over time, with the market regimes shaded
  2. actual vs forecast price, 7-day moving average, and the bias below it
  3. error density by year
  4. RMSE and MAE by hour of the day
  5. error by day of the week

Input:   results/lear/<LABEL>_rolling_validation_{weekly,hourly}.csv
         Set LABEL below to the run you want to plot (the file-name prefix
         written by models/lear/validate.py).
Output:  figures/lear_validation/1_weekly_rmse.png ... 5_error_by_dow.png
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT    = Path(__file__).parent.parent.parent.parent   # .../case_a
RES_DIR = ROOT / "results/lear"
LABEL   = "lear_structural"   # file-name prefix of the validate.py run to plot
OUT_DIR = ROOT / "figures/lear_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading results...")
weekly = pd.read_csv(RES_DIR / f"{LABEL}_rolling_validation_weekly.csv",
                     parse_dates=["week_start"])
hourly = pd.read_csv(RES_DIR / f"{LABEL}_rolling_validation_hourly.csv",
                     parse_dates=["date"])

hourly["error"]     = hourly["forecast_lear"] - hourly["actual"]
hourly["abs_error"] = hourly["error"].abs()
hourly["datetime"]  = hourly["date"] + pd.to_timedelta(hourly["hour"], unit="h")
hourly["dow"]       = hourly["date"].dt.day_name()
hourly["year"]      = hourly["date"].dt.year

n_weeks = len(weekly)
n_obs   = hourly["error"].notna().sum()
overall_rmse = np.sqrt(np.nanmean(hourly["error"] ** 2))
overall_mae  = np.nanmean(hourly["abs_error"])
print(f"  {n_weeks} weeks, {n_obs} valid hourly obs")
print(f"  Overall RMSE={overall_rmse:.2f}  MAE={overall_mae:.2f} EUR/MWh")

# Periods shaded in the time-series figures: (label, start, end, colour)
REGIMES = [
    ("Pre-crisis",     "2022-01-01", "2022-05-31", "#d0e8f0"),
    ("Energy crisis",  "2022-06-01", "2022-12-31", "#fde8d8"),
    ("Normalisation",  "2023-01-01", "2023-12-31", "#d8f0d8"),
]

COLORS = ["#2166ac", "#d6604d", "#4dac26", "#7b3294", "#1a1a1a"]
DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def add_regime_shading(ax):
    for label, start, end, color in REGIMES:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=color, alpha=0.35, zorder=0)
    patches = [mpatches.Patch(color=c, alpha=0.5, label=l)
               for l, _, _, c in REGIMES]
    return patches


def style(ax):
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def set_monthly_xaxis(ax, interval=2):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.tick_params(axis="x", which="major", labelsize=8)


# ---------------------------------------------------------------------------
# Figure 1: Weekly RMSE over time with regime shading
# ---------------------------------------------------------------------------
print("Figure 1: Weekly RMSE time series...")
fig, ax = plt.subplots(figsize=(14, 5))

regime_patches = add_regime_shading(ax)
ax.plot(weekly["week_start"], weekly["rmse"],
        color=COLORS[0], linewidth=1.2, marker="o", markersize=3, zorder=3,
        label="Weekly RMSE")
ax.axhline(weekly["rmse"].mean(), color=COLORS[1], linewidth=1.0, linestyle="--", zorder=2,
           label=f"Mean = {weekly['rmse'].mean():.1f} EUR/MWh")
ax.axhline(weekly["rmse"].median(), color=COLORS[2], linewidth=1.0, linestyle=":", zorder=2,
           label=f"Median = {weekly['rmse'].median():.1f} EUR/MWh")

ax.set_ylabel("RMSE (EUR/MWh)", fontsize=9)
ax.set_title(f"LEAR Model — Weekly RMSE, Rolling Validation 2022–2023  "
             f"(n={n_weeks} weeks)", fontsize=11)
set_monthly_xaxis(ax)
ax.set_xlim(weekly["week_start"].min(), weekly["week_start"].max())

# One legend with both the lines and the regime colours
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles + regime_patches, labels + [r for r, *_ in REGIMES],
          fontsize=8, frameon=False, loc="upper right")
style(ax)
fig.tight_layout()
fig.savefig(OUT_DIR / "1_weekly_rmse.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved 1_weekly_rmse.png")


# ---------------------------------------------------------------------------
# Figure 2: Actual vs. forecast - 7-day MA
# ---------------------------------------------------------------------------
print("Figure 2: Actual vs forecast 7-day MA...")

ts = (hourly.sort_values("datetime")
      .set_index("datetime")[["actual", "forecast_lear"]]
      .resample("D").mean())
ts_ma = ts.rolling(7, min_periods=1).mean()

fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True,
                          gridspec_kw={"height_ratios": [3, 1]})

ax1 = axes[0]
add_regime_shading(ax1)
ax1.plot(ts_ma.index, ts_ma["actual"],         color=COLORS[0], linewidth=1.2, label="Actual")
ax1.plot(ts_ma.index, ts_ma["forecast_lear"],  color=COLORS[1], linewidth=1.2,
         linestyle="--", label="LEAR forecast")
ax1.set_ylabel("Day-ahead Price (EUR/MWh)", fontsize=9)
ax1.set_title("DK1 Actual vs. LEAR Forecast — 7-day MA, 2022–2023", fontsize=11)
ax1.legend(fontsize=9, frameon=False)
style(ax1)

ax2 = axes[1]
bias_ma = ts_ma["forecast_lear"] - ts_ma["actual"]
ax2.axhline(0, color="k", linewidth=0.6)
ax2.fill_between(bias_ma.index, bias_ma, 0,
                 where=bias_ma >= 0, color=COLORS[1], alpha=0.5, label="Over-forecast")
ax2.fill_between(bias_ma.index, bias_ma, 0,
                 where=bias_ma < 0,  color=COLORS[0], alpha=0.5, label="Under-forecast")
ax2.set_ylabel("Bias (EUR/MWh)", fontsize=9)
ax2.legend(fontsize=8, frameon=False)
style(ax2)

set_monthly_xaxis(ax2)
ax2.set_xlim(ts_ma.index.min(), ts_ma.index.max())

fig.tight_layout()
fig.savefig(OUT_DIR / "2_actual_vs_forecast_ma.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved 2_actual_vs_forecast_ma.png")


# ---------------------------------------------------------------------------
# Figure 3: Error distribution - KDE by year
# ---------------------------------------------------------------------------
print("Figure 3: Error distribution by year...")
import matplotlib as mpl

fig, ax = plt.subplots(figsize=(9, 5))

year_colors = {2022: COLORS[0], 2023: COLORS[1]}
for yr, color in year_colors.items():
    errs = hourly.loc[hourly["year"] == yr, "error"].dropna()
    if len(errs) < 10:
        continue
    errs.plot.kde(ax=ax, color=color, linewidth=1.4,
                  label=f"{yr}  (RMSE={np.sqrt(np.mean(errs**2)):.1f}, "
                        f"bias={errs.mean():.1f} EUR/MWh)")
    ax.axvline(errs.mean(), color=color, linewidth=0.8, linestyle="--")

ax.axvline(0, color="k", linewidth=0.6)
ax.set_xlabel("Forecast error (EUR/MWh)", fontsize=9)
ax.set_ylabel("Density", fontsize=9)
ax.set_title("LEAR Forecast Error Distribution by Year", fontsize=11)
ax.legend(fontsize=9, frameon=False)
style(ax)
# Cut the long tails: show +-6 MADs of the error
err_mad = np.nanmedian(np.abs(hourly["error"] - hourly["error"].median()))
ax.set_xlim(-6 * err_mad, 6 * err_mad)
fig.tight_layout()
fig.savefig(OUT_DIR / "3_error_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved 3_error_distribution.png")


# ---------------------------------------------------------------------------
# Figure 4: Per-hour RMSE
# ---------------------------------------------------------------------------
print("Figure 4: Per-hour RMSE...")

hour_stats = hourly.groupby("hour").apply(
    lambda g: pd.Series({
        "rmse": np.sqrt(np.nanmean(g["error"] ** 2)),
        "mae":  np.nanmean(g["abs_error"]),
        "bias": np.nanmean(g["error"]),
    }), include_groups=False
)

fig, ax = plt.subplots(figsize=(12, 4))
bars = ax.bar(hour_stats.index, hour_stats["rmse"], color=COLORS[0], alpha=0.8, label="RMSE")
ax.plot(hour_stats.index, hour_stats["mae"], color=COLORS[1], linewidth=1.4,
        marker="o", markersize=4, label="MAE")
ax.axhline(overall_rmse, color=COLORS[2], linewidth=1.0, linestyle="--",
           label=f"Overall RMSE = {overall_rmse:.1f}")
ax.set_xlabel("Hour of day (UTC)", fontsize=9)
ax.set_ylabel("EUR/MWh", fontsize=9)
ax.set_title("LEAR Per-hour RMSE and MAE — Rolling Validation 2022–2023", fontsize=11)
ax.set_xticks(range(24))
ax.legend(fontsize=9, frameon=False)
style(ax)
fig.tight_layout()
fig.savefig(OUT_DIR / "4_rmse_by_hour.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved 4_rmse_by_hour.png")


# ---------------------------------------------------------------------------
# Figure 5: Error box plots by day-of-week
# ---------------------------------------------------------------------------
print("Figure 5: Error by day-of-week...")

dow_data = [hourly.loc[hourly["dow"] == d, "error"].dropna().values
            for d in DOW_ORDER]

fig, ax = plt.subplots(figsize=(10, 5))
bp = ax.boxplot(dow_data,
                patch_artist=True, notch=False, whis=1.5,
                medianprops=dict(color="black", linewidth=1.5),
                flierprops=dict(marker=".", markersize=2, alpha=0.3))
# Set the tick labels separately: boxplot's labels= argument was renamed in
# newer matplotlib versions, this works with all of them
ax.set_xticks(range(1, len(DOW_ORDER) + 1))
ax.set_xticklabels([d[:3] for d in DOW_ORDER])

palette = [COLORS[0]] * 5 + [COLORS[1]] * 2  # weekdays blue, weekend red
for patch, color in zip(bp["boxes"], palette):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.axhline(0, color="k", linewidth=0.6)
ax.set_xlabel("Day of week", fontsize=9)
ax.set_ylabel("Forecast error (EUR/MWh)", fontsize=9)
ax.set_title("LEAR Forecast Error by Day of Week — Rolling Validation 2022–2023", fontsize=11)

weekday_patch  = mpatches.Patch(color=COLORS[0], alpha=0.6, label="Weekday")
weekend_patch  = mpatches.Patch(color=COLORS[1], alpha=0.6, label="Weekend")
ax.legend(handles=[weekday_patch, weekend_patch], fontsize=9, frameon=False)
style(ax)
fig.tight_layout()
fig.savefig(OUT_DIR / "5_error_by_dow.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved 5_error_by_dow.png")

print(f"\nAll figures saved to {OUT_DIR}/")
print("Done.")
