"""
Descriptive figures, all smoothed with a 7-day moving average:
  1. DK1 day-ahead price, 2018-2023
  2. Day-ahead renewable generation forecast by type, 2018-2023
  3. Total physical flows into and out of DK1, 2023
  4. Coal, EUA, gas and oil prices against the DK1 price, 2018-2023

Input:   data/interim/
Output:  figures/descriptives/ElectricityPrices/*.png
         figures/descriptives/EnergySeries/energy_prices_2018_2023_7dma.png
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parent.parent.parent.parent   # .../case_a
DATA = ROOT / "data/interim"
OUT_EP = ROOT / "figures/descriptives/ElectricityPrices"
OUT_ES = ROOT / "figures/descriptives/EnergySeries"
OUT_EP.mkdir(parents=True, exist_ok=True)
OUT_ES.mkdir(parents=True, exist_ok=True)

COLORS = ["#2166ac", "#d6604d", "#4dac26", "#7b3294", "#1a1a1a"]


def parse_mtu(df, col):
    # Start of the MTU interval as a datetime (no time-zone conversion)
    start = (df[col].str.split(" - ").str[0]
             .str.replace(r"\s*\(CES?T\)", "", regex=True).str.strip())
    return pd.to_datetime(start, format="mixed", dayfirst=True)


def set_monthly_xaxis(ax, interval=3):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.tick_params(axis="x", which="major", labelsize=7)


def style(ax):
    # Common look: light horizontal grid, no top/right frame
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)


# ---------------------------------------------------------------------------
# Plot 1 - Day-ahead electricity price 7-day MA (2018-2023)
# ---------------------------------------------------------------------------
print("Plot 1: Electricity prices 7-day MA...")
df = pd.read_csv(DATA / "ElectricityPrices/DK1/Prices.csv",
                 low_memory=False, on_bad_lines="skip")
df["time"] = parse_mtu(df, "MTU (UTC)")
df = df[(df["time"] >= "2018-01-01") & (df["time"] < "2024-01-01")]
df = df.set_index("time")["Day-ahead Price (EUR/MWh)"]
ma = df.resample("D").mean().rolling(7, min_periods=1).mean()

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(ma.index, ma.values, color=COLORS[0], linewidth=1.2)
set_monthly_xaxis(ax, interval=3)
ax.set_xlim(ma.index.min(), ma.index.max())
ax.set_ylabel("Day-ahead Price (EUR/MWh)", fontsize=9)
ax.set_title("DK1 Day-ahead Electricity Price — 7-day MA (2018–2023)", fontsize=11)
style(ax)
fig.tight_layout()
fig.savefig(OUT_EP / "prices_2018_2023_7dma.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved prices_2018_2023_7dma.png")

# ---------------------------------------------------------------------------
# Plot 2 - Renewables Day-ahead generation 7-day MA by Type (2018-2023)
# ---------------------------------------------------------------------------
print("Plot 2: Renewables 7-day MA by type...")
df = pd.read_csv(DATA / "ElectricityPrices/DK1/Renewables.csv", low_memory=False)
df["time"] = parse_mtu(df, "MTU (CET/CEST)")
df = df[(df["time"] >= "2018-01-01") & (df["time"] < "2024-01-01")]
df["Day-ahead (MW)"] = pd.to_numeric(df["Day-ahead (MW)"], errors="coerce")

fig, ax = plt.subplots(figsize=(14, 5))
type_colors = {"OFFSHORE": COLORS[0], "ONSHORE": COLORS[1], "SOLAR": COLORS[2]}
for type_name, color in type_colors.items():
    sub = df[df["Type"] == type_name].set_index("time")["Day-ahead (MW)"]
    ma = sub.resample("D").mean().rolling(7, min_periods=1).mean()
    ax.plot(ma.index, ma.values, color=color, linewidth=1.2, label=type_name)

set_monthly_xaxis(ax, interval=3)
ax.set_xlim(df["time"].min(), df["time"].max())
ax.set_ylabel("Day-ahead Generation (MW)", fontsize=9)
ax.set_title("DK1 Renewables Day-ahead Generation — 7-day MA by Type (2018–2023)", fontsize=11)
ax.legend(fontsize=9, frameon=False)
style(ax)
fig.tight_layout()
fig.savefig(OUT_EP / "renewables_2018_2023_7dma.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved renewables_2018_2023_7dma.png")

# ---------------------------------------------------------------------------
# Plot 3 - Transmission flows into/out of DK1 7-day MA (2023)
# ---------------------------------------------------------------------------
print("Plot 3: Transmission 7-day MA (2023)...")
df = pd.read_csv(DATA / "ElectricityPrices/DK1/Transmission.csv", low_memory=False)
df["time"] = parse_mtu(df, "MTU")
df["Physical Flow (MW)"] = pd.to_numeric(df["Physical Flow (MW)"], errors="coerce")
df = df[(df["time"] >= "2023-01-01") & (df["time"] < "2024-01-01")]

into_dk1 = (df[df["In Area"] == "BZN|DK1"]
            .groupby("time")["Physical Flow (MW)"].sum()
            .resample("D").sum()
            .rolling(7, min_periods=1).mean())

out_dk1 = (df[df["Out Area"] == "BZN|DK1"]
           .groupby("time")["Physical Flow (MW)"].sum()
           .resample("D").sum()
           .rolling(7, min_periods=1).mean())

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(into_dk1.index, into_dk1.values, color=COLORS[0], linewidth=1.2, label="Flow into DK1")
ax.plot(out_dk1.index, out_dk1.values, color=COLORS[1], linewidth=1.2,
        linestyle="--", label="Flow out of DK1")
set_monthly_xaxis(ax, interval=1)
ax.set_xlim(into_dk1.index.min(), into_dk1.index.max())
ax.set_ylabel("Physical Flow (MW·days)", fontsize=9)
ax.set_title("DK1 Cross-border Physical Flows — 7-day MA (2023)", fontsize=11)
ax.legend(fontsize=9, frameon=False)
style(ax)
fig.tight_layout()
fig.savefig(OUT_EP / "transmission_2023_7dma.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved transmission_2023_7dma.png")

# ---------------------------------------------------------------------------
# Plot 4 - Energy commodities + electricity price 7-day MA (2018-2023)
# ---------------------------------------------------------------------------
print("Plot 4: Energy series + electricity price 7-day MA...")

# Commodities
es = pd.read_csv(DATA / "EnergySeries.csv", parse_dates=["Date"])
es = es[(es["Date"] >= "2018-01-01") & (es["Date"] < "2024-01-01")].set_index("Date")
commodities = {
    "Coal ($/t)":         ("Coal_Price",        "#7b3294"),
    "EUA (€/tCO₂)":      ("EUA_Price",         "#4dac26"),
    "Natural Gas ($/MMBtu)": ("Natural_Gas_Price", "#d6604d"),
    "Oil ($/bbl)":        ("Oil_Price",         "#b35806"),
}

# Electricity price (daily mean)
pr = pd.read_csv(DATA / "ElectricityPrices/DK1/Prices.csv",
                 low_memory=False, on_bad_lines="skip")
pr["time"] = parse_mtu(pr, "MTU (UTC)")
pr = pr[(pr["time"] >= "2018-01-01") & (pr["time"] < "2024-01-01")]
elec_ma = (pr.set_index("time")["Day-ahead Price (EUR/MWh)"]
           .resample("D").mean()
           .rolling(7, min_periods=1).mean())

fig, ax1 = plt.subplots(figsize=(15, 5))
ax2 = ax1.twinx()

for label, (col, color) in commodities.items():
    ma = es[col].rolling(7, min_periods=1).mean()
    ax1.plot(ma.index, ma.values, linewidth=1.2, label=label, color=color)

ax2.plot(elec_ma.index, elec_ma.values, color=COLORS[0], linewidth=1.4,
         linestyle="--", label="Day-ahead Price (EUR/MWh)")
ax2.set_ylabel("Day-ahead Electricity Price (EUR/MWh)", fontsize=9, color=COLORS[0])
ax2.tick_params(axis="y", labelcolor=COLORS[0])
ax2.spines[["top"]].set_visible(False)

ax1.set_ylabel("Commodity Price", fontsize=9)
ax1.set_title("Energy Commodity Prices + DK1 Day-ahead Electricity — 7-day MA (2018–2023)", fontsize=11)
set_monthly_xaxis(ax1, interval=3)
ax1.set_xlim(es.index.min(), es.index.max())
style(ax1)

# One legend for both y-axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, frameon=False,
           loc="upper left")

fig.tight_layout()
fig.savefig(OUT_ES / "energy_prices_2018_2023_7dma.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved energy_prices_2018_2023_7dma.png")

print("\nDone.")
