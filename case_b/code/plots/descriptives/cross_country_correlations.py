"""
How closely the DK1 price moves with prices in DK2, NO2, SE4 and DE, 2018-2024:
  1. rolling 120-day correlation of hourly prices with DK1
  2. one correlation matrix of all five zones per year
  3. yearly correlation with DK1, one line per zone

Input:   data/prepared/DK1/dataset.csv
Output:  figures/descriptives/cross_country_correlations/*.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).parent.parent.parent.parent
DATA     = ROOT / "data/prepared/DK1/dataset.csv"
OUT_DIR  = ROOT / "figures/descriptives/cross_country_correlations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ZONES   = ["price_dk2", "price_no2", "price_se4", "price_de"]
LABELS  = {"price_dk2": "DK2", "price_no2": "NO2", "price_se4": "SE4", "price_de": "DE"}
COLORS  = {"price_dk2": "#2166ac", "price_no2": "#d6604d", "price_se4": "#4dac26", "price_de": "#762a83"}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data …")
df = pd.read_csv(DATA, low_memory=False,
                 usecols=["time", "price", "price_dk2", "price_no2", "price_se4", "price_de"])
df["time"] = pd.to_datetime(df["time"])
df = df.set_index("time").sort_index()

# ---------------------------------------------------------------------------
# Figure 1: rolling correlation of each neighbour with DK1
# ---------------------------------------------------------------------------
print("Figure 1: rolling correlations …")

WINDOW = 120 * 24   # 120 days of hourly observations

fig, ax = plt.subplots(figsize=(9, 4))

for col in ZONES:
    valid = df[["price", col]].dropna()
    roll  = valid["price"].rolling(WINDOW).corr(valid[col])
    daily = roll.resample("D").mean()   # one point per day for plotting
    ax.plot(daily.index, daily.values,
            color=COLORS[col], linewidth=1.6, label=LABELS[col])

# Shade the energy crisis, Sep 2021 - Jun 2023
ax.axvspan(pd.Timestamp("2021-09-01"), pd.Timestamp("2023-06-01"),
           color="gold", alpha=0.13, zorder=0)
ax.text(pd.Timestamp("2021-09-05"), 0.03, "Energy crisis",
        fontsize=8, color="#8a6500", va="bottom")

ax.set_ylim(0, 1.05)
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
ax.set_xlim(df.index.min(), df.index.max())
ax.set_ylabel("Pearson correlation\n(120-day rolling)", fontsize=10)
ax.set_title("Cross-country electricity price correlations with DK1 (2018–2024)",
             fontsize=11, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linewidth=0.4, alpha=0.5)

ax.legend(title="vs DK1", fontsize=9, title_fontsize=9,
          loc="upper right", framealpha=0.9, borderpad=0.6)

fig.tight_layout()
out1 = OUT_DIR / "rolling_correlations.png"
fig.savefig(out1, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  saved: {out1}")

# ---------------------------------------------------------------------------
# Figure 2: correlation matrix of the five zones, one panel per year
# ---------------------------------------------------------------------------
print("Figure 2: annual heatmaps …")

all_zones  = ["price", "price_dk2", "price_no2", "price_se4", "price_de"]
zone_names = ["DK1", "DK2", "NO2", "SE4", "DE"]
years      = sorted(df.index.year.unique())

ncols = len(years)
fig, axes = plt.subplots(1, ncols, figsize=(ncols * 2.1, 2.6))

# Colour scale centred at 0.7 so differences among high correlations are visible
norm = TwoSlopeNorm(vmin=0.0, vcenter=0.7, vmax=1.0)

for ax, year in zip(axes, years):
    sub  = df[df.index.year == year][all_zones].dropna()
    corr = sub.corr().values

    im = ax.imshow(corr, cmap="RdYlGn", norm=norm, aspect="auto")

    ax.set_xticks(range(len(zone_names)))
    ax.set_yticks(range(len(zone_names)))
    ax.set_xticklabels(zone_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(zone_names, fontsize=7)
    ax.set_title(str(year), fontsize=9, fontweight="bold", pad=3)

    # Write the value in each cell
    for r in range(len(zone_names)):
        for c in range(len(zone_names)):
            val = corr[r, c]
            color = "white" if val < 0.5 else "black"
            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                    fontsize=5.5, color=color)

# One colour bar for all panels
cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.85, pad=0.01, aspect=20)
cbar.set_label("Pearson correlation", fontsize=8)
cbar.ax.tick_params(labelsize=7)

fig.suptitle("Annual cross-country electricity price correlation matrices",
             fontsize=10, fontweight="bold")
fig.tight_layout(rect=[0, 0, 0.92, 1])

out2 = OUT_DIR / "annual_heatmaps.png"
fig.savefig(out2, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  saved: {out2}")

# ---------------------------------------------------------------------------
# Figure 3: yearly correlation with DK1, one line per zone
# ---------------------------------------------------------------------------
print("Figure 3: annual trend lines …")

fig, ax = plt.subplots(figsize=(6, 3.5))

df["year"] = df.index.year
annual_corr = {col: [] for col in ZONES}

for year in years:
    sub = df[df["year"] == year][["price"] + ZONES].dropna()
    for col in ZONES:
        annual_corr[col].append(sub["price"].corr(sub[col]))

for col in ZONES:
    ax.plot(years, annual_corr[col], marker="o", markersize=5,
            linewidth=1.8, color=COLORS[col], label=LABELS[col])

ax.axvspan(2021, 2022.95, color="gold", alpha=0.15, zorder=0, label="Energy crisis")

ax.set_xticks(years)
ax.set_xlim(years[0] - 0.3, years[-1] + 0.3)
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
ax.set_ylabel("Annual Pearson correlation", fontsize=10)
ax.set_xlabel("Year", fontsize=10)
ax.set_title("Yearly price correlation with DK1 by market (2018–2024)",
             fontsize=11, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linewidth=0.4, alpha=0.5)
ax.legend(title="Market\n(vs DK1)", fontsize=9, title_fontsize=9,
          loc="lower right", framealpha=0.9)

fig.tight_layout()
out3 = OUT_DIR / "annual_trend_lines.png"
fig.savefig(out3, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  saved: {out3}")
print("Done.")
