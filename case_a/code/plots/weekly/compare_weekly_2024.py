"""
Weekly average DK1 price in 2024 against three forecasts: the weekly LASSO
model, the Fourier-only OLS benchmark (both from models/weekly/predict.py)
and the structural price model. Errors are shown in the lower panel.

Input:   results/weekly/weekly_forecast_2024.csv
         results/struct/data_pred_weekly.csv
Output:  figures/weekly/compare_weekly_2024.png
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
CSV_PATH = ROOT / "results/weekly/weekly_forecast_2024.csv"
STRUCT_CSV = ROOT / "results/struct/data_pred_weekly.csv"
OUT_DIR = ROOT / "figures/weekly"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"{CSV_PATH} not found — run models/weekly/predict.py first."
    )

df = pd.read_csv(CSV_PATH)
weeks = df["week"].values
actual = df["actual"].values
fc_lasso = df["forecast_lasso"].values
fc_bench = df["forecast_benchmark"].values

# Structural model forecasts are hourly; average them within each week
sdf = pd.read_csv(STRUCT_CSV)
struct_weekly = sdf.groupby("week")["price_pred_baseline"].mean().reindex(weeks).values

rmse = lambda a, b: np.sqrt(np.nanmean((a - b) ** 2))
rmse_lasso = rmse(fc_lasso, actual)
rmse_bench = rmse(fc_bench, actual)
rmse_struct = rmse(struct_weekly, actual)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1]})
ax1, ax2 = axes

ax1.plot(weeks, actual, color="#2166ac", linewidth=1.8,
         label="Actual (realized)", zorder=10)
ax1.plot(weeks, fc_lasso, color="#d6604d", linewidth=1.4, linestyle="--",
         label=f"LASSO Weekly Model  (RMSE={rmse_lasso:.2f} €/MWh)")
ax1.plot(weeks, fc_bench, color="#4dac26", linewidth=1.4, linestyle=":",
         label=f"Benchmark (Fourier OLS)  (RMSE={rmse_bench:.2f} €/MWh)")
ax1.plot(weeks, struct_weekly, color="#8c510a", linewidth=1.4, linestyle="-.",
         label=f"Structural Baseline  (RMSE={rmse_struct:.2f} €/MWh)")

ax1.set_ylabel("Weekly Avg Day-ahead Price (EUR/MWh)", fontsize=9)
ax1.set_title("DK1 — Weekly Average Price Forecasts vs Actuals, 2024", fontsize=11)
ax1.legend(fontsize=9, frameon=False)
ax1.grid(axis="y", linewidth=0.4, alpha=0.5)
ax1.spines[["top", "right"]].set_visible(False)

# Error panel
ax2.plot(weeks, fc_lasso - actual, color="#d6604d", linewidth=0.9,
         alpha=0.85, label="LASSO error")
ax2.plot(weeks, fc_bench - actual, color="#4dac26", linewidth=0.9,
         alpha=0.85, linestyle=":", label="Benchmark error")
ax2.plot(weeks, struct_weekly - actual, color="#8c510a", linewidth=0.9,
         alpha=0.85, linestyle="-.", label="Structural error")
ax2.axhline(0, color="k", linewidth=0.6)
ax2.set_xlabel("ISO Week of 2024", fontsize=9)
ax2.set_ylabel("Error (EUR/MWh)", fontsize=9)
ax2.legend(fontsize=8, frameon=False)
ax2.grid(axis="y", linewidth=0.4, alpha=0.5)
ax2.spines[["top", "right"]].set_visible(False)

# Dashed lines at the first week of Q2, Q3 and Q4
for q_start in [14, 27, 40]:
    ax1.axvline(q_start, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)
    ax2.axvline(q_start, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)

ax2.set_xlim(weeks.min(), weeks.max())
ax2.set_xticks(range(1, 53, 4))

fig.tight_layout()
out_path = OUT_DIR / "compare_weekly_2024.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out_path}", flush=True)
print("Done.", flush=True)
