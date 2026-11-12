"""
Daily weather by station, one figure per variable: wind speed (fg),
precipitation (rr), sunshine (ss) and mean temperature (tg). Each variable is
plotted for the full sample and for 2023 only, one line per station.

Input:   data/interim/Climate Variables/DK1.csv
Output:  figures/descriptives/climate_variables/<var>_{full,2023}.png
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT    = Path(__file__).parent.parent.parent.parent   # .../case_a
DATA    = ROOT / "data/interim/Climate Variables/DK1.csv"
OUT_DIR = ROOT / "figures/descriptives/climate_variables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# variable code -> (description, unit, raw values are tenths of a unit?)
VARS = {
    "fg": ("Mean Wind Speed",      "m/s",        False),
    "rr": ("Precipitation",        "mm",         False),
    "ss": ("Sunshine Duration",    "hours",      False),
    "tg": ("Mean Temperature",     "°C",         True),
}

print("Loading data...")
df = pd.read_csv(DATA, parse_dates=["DATE"], low_memory=False)

def make_plot(wide, var, desc, unit, suffix):
    # wide: one column per station, indexed by date
    n_stations = wide.shape[1]
    colors = plt.colormaps["tab20b"](np.linspace(0, 1, n_stations))

    fig, ax = plt.subplots(figsize=(16, 5))
    for i, col in enumerate(wide.columns):
        ax.plot(wide.index, wide[col], color=colors[i], alpha=0.5,
                linewidth=0.6, rasterized=True)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.tick_params(axis="x", which="major", labelsize=7)

    ax.set_xlim(wide.index.min(), wide.index.max())
    ax.set_ylabel(f"{var} ({unit})", fontsize=9)
    ax.set_title(f"{var.upper()} — {desc}  ({n_stations} stations, daily)", fontsize=11)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_path = OUT_DIR / f"{var}_{suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


for var, (desc, unit, scale_tenths) in VARS.items():
    print(f"Plotting {var}...")

    wide = df.pivot(index="DATE", columns="staid", values=var)
    if scale_tenths:
        wide = wide / 10.0

    # Full sample
    make_plot(wide, var, desc, unit, suffix="full")

    # 2023 only
    wide_2023 = wide.loc["2023-01-01":"2023-12-31"]
    make_plot(wide_2023, var, desc, unit, suffix="2023")

print("Done.")
