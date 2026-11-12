"""
DK1 day-ahead price, hour by hour, in the first ISO week of 2023 (2-8 January, UTC).

Input:   data/interim/ElectricityPrices/DK1/Prices.csv
Output:  figures/descriptives/ElectricityPrices/prices_2023w1_timeseries.png
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT    = Path(__file__).parent.parent.parent.parent   # .../case_a
DATA    = ROOT / "data/interim/ElectricityPrices/DK1/Prices.csv"
OUT_DIR = ROOT / "figures/descriptives/ElectricityPrices"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading Prices.csv...")
df = pd.read_csv(DATA, low_memory=False, on_bad_lines="skip")

start = df["MTU (UTC)"].str.split(" - ").str[0].str.strip()
df["time"] = pd.to_datetime(start, format="mixed", dayfirst=True)

w1 = df[(df["time"] >= "2023-01-02") & (df["time"] < "2023-01-09")].copy()
print(f"  Week 1 rows: {len(w1)}  ({w1['time'].min()} → {w1['time'].max()})")

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(w1["time"], w1["Day-ahead Price (EUR/MWh)"],
        color="#2166ac", linewidth=1.5)
ax.fill_between(w1["time"], w1["Day-ahead Price (EUR/MWh)"],
                alpha=0.12, color="#2166ac")

ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b"))
ax.tick_params(axis="x", which="major", labelsize=8)

ax.set_xlim(w1["time"].min(), w1["time"].max())
ax.set_ylabel("Day-ahead Price (EUR/MWh)", fontsize=9)
ax.set_title("DK1 Day-ahead Electricity Price — 2023 Week 1 (Jan 2–8, UTC)", fontsize=11)
ax.grid(axis="y", linewidth=0.4, alpha=0.5)
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
out = OUT_DIR / "prices_2023w1_timeseries.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
