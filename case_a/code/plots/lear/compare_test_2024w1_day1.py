"""
Same comparison as compare_test_2024w1.py, but only for the first day of the
test week (2024-01-01), the one day on which the models can use the published
load and renewable forecasts. RMSE is over those 24 hours.

Output:  figures/test_2024w1/compare_2024w1_day1.png
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
OUT_DIR = ROOT / "figures" / "test_2024w1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DAY1 = "2024-01-01"

# ---------------------------------------------------------------------------
# Models to plot. Uncomment entries to add them. Each display name matches the
# label of the file it reads.
# ---------------------------------------------------------------------------
MODELS = [
    {
        "name": "LEAR Baseline (AR-only days 2–7)",
        "color": "#d6604d",
        "csv_path": ROOT / "results/lear/lear_baseline_forecasts_validation_2024w1.csv",
    },
    # {
    #     "name": "LEAR Baseline + Holidays",
    #     "color": "#f4a582",
    #     "csv_path": ROOT
    #     / "results/lear/lear_baseline_holidays_forecasts_validation_2024w1.csv",
    # },
    # {
    #     "name": "LEAR Structural + Holidays",
    #     "color": "#b2abd2",
    #     "csv_path": ROOT
    #     / "results/lear/lear_structural_holidays_forecasts_validation_2024w1.csv",
    # },
    {
        "name": "LEAR Baseline + Holidays + Split Ren",
        "color": "#5e3c99",
        "csv_path": ROOT
        / "results/lear/lear_baseline_holidays_splitren_forecasts_validation_2024w1.csv",
    },
    # {
    #     "name": "LEAR Baseline + Holidays + Split Ren + Channel Transmission",
    #     "color": "#2c7bb6",
    #     "csv_path": ROOT
    #     / "results/lear/lear_baseline_holidays_splitren_trans_ch_forecasts_validation_2024w1.csv",
    # },
    # {
    #     "name": "LEAR Baseline + Holidays + Split Ren + Energy",
    #     "color": "#7fbf7f",
    #     "csv_path": ROOT
    #     / "results/lear/lear_baseline_holidays_splitren_energy_forecasts_validation_2024w1.csv",
    # },
    # {
    #     "name": "LEAR Baseline + Holidays + Split Ren + Channel Transmission + Energy",
    #     "color": "#7fbf7f",
    #     "csv_path": ROOT
    #     / "results/lear/lear_baseline_holidays_splitren_energy_trans_ch_forecasts_validation_2024w1.csv",
    # },
    # {
    #     "name": "LEAR Baseline + Holidays + Split Ren + Energy + Macro Daily",
    #     "color": "#e08214",
    #     "csv_path": ROOT
    #     / "results/lear/lear_baseline_holidays_splitren_energy_macro_daily_forecasts_validation_2024w1.csv",
    # },
    {
        "name": "LEAR Structural + Holidays + Split Ren + Energy + Macro Daily + Channel Transmission",
        "color": "#e08214",
        "csv_path": ROOT
        / "results/lear/lear_structural_holidays_splitren_energy_macro_daily_trans_ch_forecasts_validation_2024w1.csv",
    },
    {
        "name": "LightGBM Structural + Holidays + Split Ren + Energy + Macro Daily + Transmission",
        "color": "#4d9221",
        "csv_path": ROOT
        / "results/lear/lgbm_structural_splitren_holidays_energy_macro_daily_trans_ch_forecasts_validation_2024w1.csv",
    },
]

STRUCT_CSV = ROOT / "results/struct/data_pred_first_week.csv"

# Forecasts of the separate structural price model (Section 3) are columns of
# one CSV: struct_variant "baseline" reads column price_pred_baseline.
STRUCT_MODELS = [
    {
        "name": "Structural Baseline",
        "color": "#8c510a",
        "struct_variant": "baseline",
    },
]

# ---------------------------------------------------------------------------
# Load data (day 1 only)
# ---------------------------------------------------------------------------
model_data = []
actuals = None

for m in MODELS:
    path = Path(m["csv_path"])
    if not path.exists():
        print(f"  WARNING: {path} not found — skipping {m['name']}", flush=True)
        continue
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"], unit="h")
    df = df[df["date"] == DAY1].sort_values("datetime").reset_index(drop=True)

    if actuals is None:
        actuals = df[["datetime", "actual"]].copy()

    rmse = np.sqrt(np.nanmean((df["forecast_lear"] - df["actual"]) ** 2))
    model_data.append(
        {
            "name": m["name"],
            "color": m["color"],
            "times": df["datetime"].values,
            "fc": df["forecast_lear"].values,
            "error": df["forecast_lear"].values - df["actual"].values,
            "rmse": rmse,
        }
    )
    print(f"  Loaded: {m['name']}  RMSE={rmse:.2f} EUR/MWh", flush=True)

if STRUCT_MODELS and STRUCT_CSV.exists():
    sdf = pd.read_csv(STRUCT_CSV)
    sdf["datetime"] = pd.to_datetime(sdf["date"]) + pd.to_timedelta(
        sdf["hour"], unit="h"
    )
    sdf = sdf[sdf["date"] == DAY1].sort_values("datetime").reset_index(drop=True)

    for m in STRUCT_MODELS:
        col = f"price_pred_{m['struct_variant']}"
        sub = sdf[["datetime", col, "price"]].copy()
        sub = sub.rename(columns={"price": "actual_s", col: "fc"})

        if actuals is None:
            actuals = sub[["datetime", "actual_s"]].rename(
                columns={"actual_s": "actual"}
            )

        rmse = np.sqrt(np.nanmean((sub["fc"] - sub["actual_s"]) ** 2))
        model_data.append(
            {
                "name": m["name"],
                "color": m["color"],
                "times": np.asarray(sub["datetime"]),
                "fc": np.asarray(sub["fc"], dtype=float),
                "error": np.asarray(sub["fc"], dtype=float)
                - np.asarray(sub["actual_s"], dtype=float),
                "rmse": rmse,
            }
        )
        print(f"  Loaded: {m['name']}  RMSE={rmse:.2f} EUR/MWh", flush=True)

if not model_data:
    raise RuntimeError("No model CSVs found — check csv_path entries in MODELS config.")

times = actuals["datetime"].values
ac_vals = actuals["actual"].values

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
n_models = len(model_data)
fig, axes = plt.subplots(
    2,
    1,
    figsize=(12, 7),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1]},
)
ax1, ax2 = axes

ax1.plot(
    times, ac_vals, color="#2166ac", linewidth=1.6, label="Actual (realized)", zorder=10
)
for m in model_data:
    ax1.plot(
        m["times"],
        m["fc"],
        color=m["color"],
        linewidth=1.3,
        linestyle="--",
        label=f"{m['name']}  (RMSE={m['rmse']:.2f} €/MWh)",
    )

ax1.set_ylabel("Day-ahead Price (EUR/MWh)", fontsize=9)
ax1.set_title(
    f"DK1 — Model Comparison, Day 1 only ({DAY1})",
    fontsize=11,
)
ax1.legend(fontsize=9, frameon=False)
ax1.grid(axis="y", linewidth=0.4, alpha=0.5)
ax1.spines[["top", "right"]].set_visible(False)

for m in model_data:
    ax2.plot(
        m["times"],
        m["error"],
        color=m["color"],
        linewidth=0.9,
        alpha=0.85,
        label=m["name"],
    )
ax2.axhline(0, color="k", linewidth=0.6)
ax2.set_ylabel("Error (EUR/MWh)", fontsize=9)
if n_models > 1:
    ax2.legend(fontsize=8, frameon=False)
ax2.grid(axis="y", linewidth=0.4, alpha=0.5)
ax2.spines[["top", "right"]].set_visible(False)
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=3))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax2.tick_params(axis="x", which="major", labelsize=8)
ax2.set_xlim(times.min(), times.max())

fig.tight_layout()
out_path = OUT_DIR / "compare_2024w1_day1.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved to {out_path}", flush=True)
print("Done.", flush=True)
