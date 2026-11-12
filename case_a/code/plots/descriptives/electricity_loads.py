"""
DK1 actual vs day-ahead forecast load in the first ISO week of 2023
(Mon 2 - Sun 8 January): a density plot and a time-series plot.

Input:   data/interim/ElectricityPrices/DK1/Loads.csv
Output:  figures/descriptives/ElectricityPrices/loads_2023w1_{distribution,timeseries}.png
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT    = Path(__file__).parent.parent.parent.parent   # .../case_a
DATA    = ROOT / "data/interim/ElectricityPrices/DK1/Loads.csv"
OUT_DIR = ROOT / "figures/descriptives/ElectricityPrices"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COL_ACTUAL   = "Actual Total Load (MW)"
COL_FORECAST = "Day-ahead Total Load Forecast (MW)"

# ---------------------------------------------------------------------------
# Load and parse
# ---------------------------------------------------------------------------
print("Loading Loads.csv...")
df = pd.read_csv(DATA, low_memory=False)

# Start of each MTU interval (local time). Older files have "(CET)"/"(CEST)"
# inside the string, which has to be removed before parsing.
start = (df["MTU (CET/CEST)"].str.split(" - ").str[0]
           .str.replace(r"\s*\(CES?T\)", "", regex=True)
           .str.strip())
df["time"] = pd.to_datetime(start, format="mixed", dayfirst=True)

# Missing values are written as "n/e" in the file, so force the load columns
# to numbers (otherwise newer pandas reads them as text)
for col in (COL_ACTUAL, COL_FORECAST):
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Filter to ISO week 1 of 2023 (Mon 2 Jan - Sun 8 Jan)
w1 = df[(df["time"] >= "2023-01-02") & (df["time"] < "2023-01-09")].copy()
print(f"  Week 1 rows: {len(w1)}  ({w1['time'].min()} → {w1['time'].max()})")

# ---------------------------------------------------------------------------
# Plot 1: Distribution of actual vs forecast load
# ---------------------------------------------------------------------------
print("Plot 1: Distribution...")

plot_df = pd.melt(
    w1[["time", COL_ACTUAL, COL_FORECAST]],
    id_vars="time",
    value_vars=[COL_ACTUAL, COL_FORECAST],
    var_name="Series",
    value_name="Load (MW)",
)
plot_df["Series"] = plot_df["Series"].map({
    COL_ACTUAL:   "Actual",
    COL_FORECAST: "Day-ahead Forecast",
})

fig, ax = plt.subplots(figsize=(8, 5))
sns.kdeplot(
    data=plot_df, x="Load (MW)", hue="Series",
    fill=True, alpha=0.35, linewidth=1.5,
    palette={"Actual": "#2166ac", "Day-ahead Forecast": "#d6604d"},
    ax=ax,
)
ax.set_title("DK1 Load Distribution — 2023 Week 1 (Jan 2–8)", fontsize=11)
ax.set_xlabel("Load (MW)", fontsize=9)
ax.set_ylabel("Density", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = OUT_DIR / "loads_2023w1_distribution.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out}")

# ---------------------------------------------------------------------------
# Plot 2: Time series - actual vs forecast
# ---------------------------------------------------------------------------
print("Plot 2: Time series...")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(w1["time"], w1[COL_ACTUAL],   color="#2166ac", linewidth=1.5, label="Actual")
ax.plot(w1["time"], w1[COL_FORECAST], color="#d6604d", linewidth=1.5,
        linestyle="--", label="Day-ahead Forecast")

ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b"))
ax.tick_params(axis="x", which="major", labelsize=8)

ax.set_xlim(w1["time"].min(), w1["time"].max())
ax.set_ylabel("Load (MW)", fontsize=9)
ax.set_title("DK1 Actual vs Day-ahead Forecast Load — 2023 Week 1 (Jan 2–8)", fontsize=11)
ax.legend(fontsize=9, frameon=False)
ax.grid(axis="y", linewidth=0.4, alpha=0.5)
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
out = OUT_DIR / "loads_2023w1_timeseries.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out}")

print("Done.")
