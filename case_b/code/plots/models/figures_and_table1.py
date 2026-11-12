"""
Figures and Table 1 for the Case B write-up, from the saved rolling-forecast
results (models/lear/rolling.py and models/lgbm/predict.py).

  Fig 1-2  size and selection frequency of the LASSO coefficients on the
           renewable regressors over the year, 2020 vs 2024
  Fig 3    variance of the forecast errors over the year
  Fig 4    selection frequency of the neighbouring-zone prices
  Table 1  RMSE of baseline LEAR, extended LEAR and LightGBM (CSV and LaTeX)
  compare_* plots: forecasts vs actual prices for selected weeks, days and
           the full year, 2020 and 2024 side by side

Each figure comes in a "simple" version (one line per year) and a "detailed"
one (split by variable type or zone). Result files that are missing are
reported and skipped.

Run:     python figures_and_table1.py      (from plots/models/)
Output:  results/figures/
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "models/lear"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

from core import ModelConfig, CROSS_ZONES, DK1_CHANNELS, CROSS_ZONE_FLOWS

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT             = Path(__file__).parent.parent.parent.parent
RESULTS_DIR      = ROOT / "results/lear"
LGBM_RESULTS_DIR = ROOT / "results/lgbm"
OUT_DIR          = ROOT / "results/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Settings: which result files to read. Labels are the file name prefixes
# written by rolling.py / lgbm/predict.py.
# ---------------------------------------------------------------------------

# Suffix of the full specification. These runs were made before rolling.py
# added "ren" to the label, so it is missing here although cross-zone
# renewables were included (the coefs files have 2497 coefficient columns,
# which matches COEFS_CONFIG with cross-zone ren on).
_FULL = "xzone_trans_trans_xz_energy_macro_holidays_interact"

# Coefficient files for Figs 1, 2 and 4 (full specification)
COEFS_LABEL = {
    2020: f"rolling_2020_{_FULL}",
    2024: f"rolling_2024_{_FULL}",
}

# Must match the runs above: it is used to work out which coef_XXXX column is which regressor
COEFS_CONFIG = ModelConfig(
    use_exogenous               = True,
    use_cross_zone_prices       = True,
    use_cross_zone_ren          = True,
    use_dk1_transmission        = True,
    use_cross_zone_transmission = True,
    use_energy                  = True,
    use_macro_daily             = True,
    use_holidays                = True,
    use_interactions            = True,
    train_cutoff                = "2019-12-31",
)

# Hourly forecast files for Fig 3
HOURLY_LABEL = {
    2020: COEFS_LABEL[2020],
    2024: COEFS_LABEL[2024],
}

# Table 1 rows: (name, label 2020, label 2024). "lgbm:" means the file is in
# results/lgbm/ instead of results/lear/.
TABLE1_VARIANTS = [
    ("Baseline LEAR",
     "rolling_2020_4var_holidays",   # the 2020 baseline was saved under an older label
     "rolling_2024_holidays"),

    (f"Full extended LEAR",
     f"rolling_2020_{_FULL}",
     f"rolling_2024_{_FULL}"),

    ("LightGBM (full spec)",
     f"lgbm:lgbm_2020_{_FULL}",
     f"lgbm:lgbm_2024_{_FULL}"),
]

# Smoothing window (days) for the time series in Figs 1-4
SMOOTH = 30

# ---------------------------------------------------------------------------
# Comparison plots
# ---------------------------------------------------------------------------
# (name, label, colour). {year} is filled in with 2020 or 2024; "lgbm:" again
# means results/lgbm/. A label of None means: use _BASELINE for that year.
_BASELINE = {2020: "rolling_2020_4var_holidays", 2024: "rolling_2024_holidays"}

COMPARE_MODELS = [
    ("Baseline LEAR",     None,                                       "#d6604d"),
    ("Full extended LEAR", f"rolling_{{year}}_{_FULL}",               "#1b7837"),
    ("LightGBM",           f"lgbm:lgbm_{{year}}_{_FULL}",            "#762a83"),
]

# Weeks to plot: (start MM-DD, number of days, title), used for both years
COMPARE_PERIODS_WEEK = [
    ("01-06", 7, "January"),
    ("04-01", 7, "April"),
    ("07-01", 7, "July"),
    ("10-07", 7, "October"),
]

# Single days to plot: (MM-DD, title)
COMPARE_PERIODS_DAY = [
    ("01-06", "January"),
    ("07-01", "July"),
]



# ---------------------------------------------------------------------------
# Regressor names, in the same order as core.build_feature_row
# ---------------------------------------------------------------------------
def build_feature_names(config: ModelConfig) -> list:
    """Names of the regressors, in the order core.build_feature_row builds them.

    If core.py changes, this has to change with it; the assert in __main__
    checks at least that the counts agree.
    """
    names = []

    for lag in [1, 2, 3, 7]:
        for h in range(24):
            names.append(f"ar_p_lag{lag}d_h{h:02d}")

    if config.use_exogenous:
        for lag in [0, 1, 7]:
            for var in ["load", "onshore", "offshore", "solar"]:
                for h in range(24):
                    names.append(f"{var}_lag{lag}d_h{h:02d}")

    for day in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
        names.append(f"dow_{day}")

    if config.use_cross_zone_prices:
        for lag in [1, 7]:
            for z in CROSS_ZONES:
                for h in range(24):
                    names.append(f"price_{z}_lag{lag}d_h{h:02d}")

    if config.use_cross_zone_ren:
        for lag in [0, 1, 7]:
            for z in CROSS_ZONES:
                for h in range(24):
                    names.append(f"ren_{z}_lag{lag}d_h{h:02d}")

    if config.use_dk1_transmission:
        for lag in [1, 7]:
            for ch in DK1_CHANNELS:
                for h in range(24):
                    names.append(f"flow_dk1_{ch}_lag{lag}d_h{h:02d}")

    if config.use_cross_zone_transmission:
        for lag in [1, 7]:
            for z, channels in CROSS_ZONE_FLOWS.items():
                for ch in channels:
                    for h in range(24):
                        names.append(f"flow_{z}_{ch}_lag{lag}d_h{h:02d}")

    if config.use_energy:
        for lag in [1, 7]:
            for var in ["coal", "eua", "gas", "oil"]:
                names.append(f"{var}_lag{lag}d")

    if config.use_macro_daily:
        for lag in [1, 7]:
            for var in ["stoxx", "msci", "usdeur", "vix"]:
                names.append(f"{var}_lag{lag}d")

    if config.use_holidays:
        names.append("is_holiday")

    if config.use_interactions:
        for h in range(24):
            names.append(f"nl_dk1_h{h:02d}")
        for h in range(24):
            names.append(f"nl_dk1_sq_h{h:02d}")
        for h in range(24):
            names.append(f"nl_dk1_x_de_h{h:02d}")
        for h in range(24):
            names.append(f"nl_dk1_x_dk2_h{h:02d}")
        for h in range(24):
            names.append(f"precip_x_flow_h{h:02d}")
        names.append("temp_sq")

    return names


def ren_col_groups(feature_names: list) -> dict:
    """{"onshore"/"offshore"/"solar": names of their coefficient columns}."""
    return {
        var: [f"coef_{i:04d}" for i, n in enumerate(feature_names)
              if n.startswith(f"{var}_lag")]
        for var in ["onshore", "offshore", "solar"]
    }


def xzone_price_col_groups(feature_names: list) -> dict:
    """{zone: names of the coefficient columns of that zone's price lags}."""
    return {
        z: [f"coef_{i:04d}" for i, n in enumerate(feature_names)
            if n.startswith(f"price_{z}_lag")]
        for z in CROSS_ZONES
    }


# ---------------------------------------------------------------------------
# Loading results (missing files print a warning and return None)
# ---------------------------------------------------------------------------
def _load_coefs(label: str, usecols=None) -> pd.DataFrame | None:
    path = RESULTS_DIR / f"{label}_coefs.csv"
    if not path.exists():
        print(f"  [WARN] not found: {path.name}", flush=True)
        return None
    kw = {"usecols": usecols} if usecols is not None else {}
    df = pd.read_csv(path, **kw)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _load_hourly(label: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / f"{label}_hourly.csv"
    if not path.exists():
        print(f"  [WARN] not found: {path.name}", flush=True)
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["error"] = df["forecast_lear"] - df["actual"]
    return df


# ---------------------------------------------------------------------------
# Daily summaries of the coefficients, indexed by day of year
# ---------------------------------------------------------------------------
def _daily_mean_abs_coef(coefs_df: pd.DataFrame, col_list: list) -> pd.Series:
    """Mean absolute coefficient per day (over hours and the given columns),
    smoothed over SMOOTH days."""
    present = [c for c in col_list if c in coefs_df.columns]
    if not present:
        return pd.Series(dtype=float)
    abs_vals = coefs_df[["date"] + present].copy()
    abs_vals[present] = abs_vals[present].abs()
    daily = abs_vals.groupby("date")[present].mean().mean(axis=1)
    daily.index = pd.to_datetime(daily.index).dayofyear
    return daily.rolling(SMOOTH, min_periods=1, center=True).mean()


def _daily_selection_freq(coefs_df: pd.DataFrame, col_list: list) -> pd.Series:
    """Share of non-zero coefficients per day (over hours and the given
    columns), smoothed over SMOOTH days."""
    present = [c for c in col_list if c in coefs_df.columns]
    if not present:
        return pd.Series(dtype=float)
    nonzero = (coefs_df[present] != 0).astype(float)
    nonzero["date"] = coefs_df["date"].values
    daily = nonzero.groupby("date")[present].mean().mean(axis=1)
    daily.index = pd.to_datetime(daily.index).dayofyear
    return daily.rolling(SMOOTH, min_periods=1, center=True).mean()


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
_MONTH_DOY   = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
_MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]


def _format_doy_ax(ax, ylabel: str, title: str):
    # Day-of-year x-axis labelled by month
    ax.set_xticks(_MONTH_DOY)
    ax.set_xticklabels(_MONTH_NAMES, fontsize=8)
    ax.set_xlim(1, 365)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}", flush=True)


# ---------------------------------------------------------------------------
# Figures 1 and 2: renewable coefficients (size and how often selected).
# Made together because they read the same files.
# ---------------------------------------------------------------------------
def make_figs_1_2(feat_names: list):
    print("\nFigures 1 & 2 — renewable coefficients...", flush=True)

    groups   = ren_col_groups(feat_names)
    all_cols = [c for cols in groups.values() for c in cols]
    usecols  = ["date", "hour"] + all_cols

    raw = {yr: _load_coefs(lbl, usecols=usecols)
           for yr, lbl in COEFS_LABEL.items()}

    COL  = {2020: "#2166ac", 2024: "#d6604d"}
    STYLE = {"onshore": "-", "offshore": "--", "solar": ":"}
    NAME  = {"onshore": "Onshore wind", "offshore": "Offshore wind", "solar": "Solar"}

    for agg_fn, ylabel, fname_base, title_word in [
        (_daily_mean_abs_coef, "Mean |coefficient| (asinh scale)",
         "fig1_coef_magnitude",   "magnitude"),
        (_daily_selection_freq,  "Selection frequency",
         "fig2_ren_selection",    "selection frequency"),
    ]:
        # Simple: one line per year, averaged over the three types
        fig, ax = plt.subplots(figsize=(12, 4))
        for yr, df in raw.items():
            if df is None:
                continue
            s = agg_fn(df, all_cols)
            ax.plot(s.index, s.values, color=COL[yr], linewidth=1.3, label=str(yr))
        _format_doy_ax(ax, ylabel,
            f"Renewable variables — {title_word}  ({SMOOTH}-day rolling average)")
        ax.legend(fontsize=9, frameon=False)
        _save(fig, OUT_DIR / f"{fname_base}_simple.png")

        # Detailed: one line per type and year
        fig, ax = plt.subplots(figsize=(12, 4))
        for yr, df in raw.items():
            if df is None:
                continue
            for var, cols in groups.items():
                s = agg_fn(df, cols)
                ax.plot(s.index, s.values,
                        color=COL[yr], linestyle=STYLE[var], linewidth=1.1, alpha=0.9,
                        label=f"{NAME[var]} {yr}")
        _format_doy_ax(ax, ylabel,
            f"Renewable variables by type — {title_word}  ({SMOOTH}-day rolling average)")
        ax.legend(fontsize=8, frameon=False, ncol=2)
        _save(fig, OUT_DIR / f"{fname_base}_detailed.png")

    print("  Done.", flush=True)


# ---------------------------------------------------------------------------
# Figure 3: forecast error variance over the year
# ---------------------------------------------------------------------------
def make_fig3():
    print("\nFigure 3 — residual variance...", flush=True)

    COL = {2020: "#2166ac", 2024: "#d6604d"}

    stats = {}
    for yr, lbl in HOURLY_LABEL.items():
        df = _load_hourly(lbl)
        if df is None:
            continue
        df["sq_error"] = df["error"] ** 2
        g = df.groupby(df["date"].dt.dayofyear)["sq_error"]
        stats[yr] = pd.DataFrame({
            "mean": g.mean(),
            "q25":  g.quantile(0.25),
            "q75":  g.quantile(0.75),
        })

    ylabel = "Variance of forecast errors (EUR/MWh)\u00b2"

    # Simple: smoothed mean squared error
    fig, ax = plt.subplots(figsize=(12, 4))
    for yr, ds in stats.items():
        s = ds["mean"].rolling(SMOOTH, min_periods=1, center=True).mean()
        ax.plot(s.index, s.values, color=COL[yr], linewidth=1.3, label=str(yr))
    _format_doy_ax(ax, ylabel,
        f"Residual variance — {SMOOTH}-day rolling mean of daily MSE")
    ax.legend(fontsize=9, frameon=False)
    _save(fig, OUT_DIR / "fig3_residual_variance_simple.png")

    # Detailed: plus the interquartile range of the squared errors
    fig, ax = plt.subplots(figsize=(12, 4))
    for yr, ds in stats.items():
        mean_s = ds["mean"].rolling(SMOOTH, min_periods=1, center=True).mean()
        q25_s  = ds["q25"].rolling(SMOOTH, min_periods=1, center=True).mean()
        q75_s  = ds["q75"].rolling(SMOOTH, min_periods=1, center=True).mean()
        ax.plot(mean_s.index, mean_s.values, color=COL[yr], linewidth=1.3, label=str(yr))
        ax.fill_between(ds.index, q25_s.values, q75_s.values,
                        color=COL[yr], alpha=0.15)
    _format_doy_ax(ax, ylabel,
        f"Residual variance — {SMOOTH}-day rolling mean ± IQR of daily squared errors")
    ax.legend(fontsize=9, frameon=False)
    _save(fig, OUT_DIR / "fig3_residual_variance_detailed.png")

    print("  Done.", flush=True)


# ---------------------------------------------------------------------------
# Figure 4: how often the neighbouring-zone prices are selected
# ---------------------------------------------------------------------------
def make_fig4(feat_names: list):
    print("\nFigure 4 — cross-zone price selection...", flush=True)

    groups   = xzone_price_col_groups(feat_names)
    all_cols = [c for cols in groups.values() for c in cols]

    if not all_cols:
        print("  use_cross_zone_prices=False in COEFS_CONFIG — skipping.", flush=True)
        return

    usecols = ["date", "hour"] + all_cols
    raw = {yr: _load_coefs(lbl, usecols=usecols)
           for yr, lbl in COEFS_LABEL.items()}

    COL   = {2020: "#2166ac", 2024: "#d6604d"}
    STYLE = {"dk2": "-", "no2": "--", "se4": "-.", "de": ":"}
    NAME  = {"dk2": "DK2", "no2": "NO2", "se4": "SE4", "de": "DE"}

    ylabel = "Selection frequency"

    # Simple: averaged over the four zones
    fig, ax = plt.subplots(figsize=(12, 4))
    for yr, df in raw.items():
        if df is None:
            continue
        s = _daily_selection_freq(df, all_cols)
        ax.plot(s.index, s.values, color=COL[yr], linewidth=1.3, label=str(yr))
    _format_doy_ax(ax, ylabel,
        f"Cross-zone price variables — selection frequency  ({SMOOTH}-day rolling average)")
    ax.legend(fontsize=9, frameon=False)
    _save(fig, OUT_DIR / "fig4_xzone_price_selection_simple.png")

    # Detailed: by zone
    fig, ax = plt.subplots(figsize=(12, 4))
    for yr, df in raw.items():
        if df is None:
            continue
        for z, cols in groups.items():
            s = _daily_selection_freq(df, cols)
            ax.plot(s.index, s.values,
                    color=COL[yr], linestyle=STYLE[z], linewidth=1.1, alpha=0.9,
                    label=f"{NAME[z]} {yr}")
    _format_doy_ax(ax, ylabel,
        f"Cross-zone price variables by zone — selection frequency  ({SMOOTH}-day rolling average)")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    _save(fig, OUT_DIR / "fig4_xzone_price_selection_detailed.png")

    print("  Done.", flush=True)


# ---------------------------------------------------------------------------
# Table 1: RMSE by model and year
# ---------------------------------------------------------------------------
def make_table1():
    print("\nTable 1 — RMSE comparison...", flush=True)

    rows = []
    for display_name, lbl_2020, lbl_2024 in TABLE1_VARIANTS:
        rmse = {}
        for yr, lbl in [(2020, lbl_2020), (2024, lbl_2024)]:
            if lbl.startswith("lgbm:"):
                df = _load_hourly_lgbm(lbl[5:])
            else:
                df = _load_hourly(lbl)
            if df is not None:
                rmse[yr] = float(np.sqrt(np.nanmean(df["error"] ** 2)))
            else:
                rmse[yr] = np.nan
        rows.append({"Model": display_name,
                     "RMSE 2020": rmse[2020],
                     "RMSE 2024": rmse[2024]})

    tbl = pd.DataFrame(rows)

    # CSV
    csv_path = OUT_DIR / "table1_rmse.csv"
    tbl.to_csv(csv_path, index=False, float_format="%.3f")
    print(f"  Saved: {csv_path.name}", flush=True)

    # LaTeX (needs \usepackage{booktabs})
    tex_path = OUT_DIR / "table1_rmse.tex"
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{RMSE comparison across LEAR model variants (EUR/MWh).}\n")
        f.write("\\label{tab:lear_rmse}\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write("Model & 2020 & 2024 \\\\\n")
        f.write("\\midrule\n")
        for _, r in tbl.iterrows():
            v20 = f"{r['RMSE 2020']:.3f}" if not np.isnan(r["RMSE 2020"]) else "---"
            v24 = f"{r['RMSE 2024']:.3f}" if not np.isnan(r["RMSE 2024"]) else "---"
            f.write(f"{r['Model']} & {v20} & {v24} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"  Saved: {tex_path.name}", flush=True)

    # Console preview
    print()
    print(tbl.to_string(index=False, float_format=lambda x: f"{x:.3f}" if not np.isnan(x) else "---"))
    print("  Done.", flush=True)


# ---------------------------------------------------------------------------
# Comparison plots
# ---------------------------------------------------------------------------
def _load_hourly_lgbm(label: str) -> pd.DataFrame | None:
    """_load_hourly for results/lgbm/ (no warning if the file is missing)."""
    path = LGBM_RESULTS_DIR / f"{label}_hourly.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["date"]  = pd.to_datetime(df["date"])
    df["error"] = df["forecast_lear"] - df["actual"]
    return df


def _render_comparison_column(ax_price, ax_err, model_data, start_dt, end_dt,
                               col_title, kind, show_ylabel):
    """Fill one year's column of a comparison figure: prices on top, errors below."""
    # Actual price, taken from the first model file that covers the period
    actual_drawn = False
    for m in model_data:
        df = m["df"]
        mask = (df["datetime"] >= start_dt) & (df["datetime"] <= end_dt)
        sub  = df[mask].sort_values("datetime")
        if sub.empty:
            continue
        if not actual_drawn:
            ax_price.plot(sub["datetime"].values, sub["actual"].values,
                          color="#2166ac", linewidth=1.6,
                          label="Actual (realized)", zorder=10)
            actual_drawn = True
        break

    # Forecast and error of each model
    for m in model_data:
        df = m["df"]
        mask = (df["datetime"] >= start_dt) & (df["datetime"] <= end_dt)
        sub  = df[mask].sort_values("datetime")
        if sub.empty:
            continue

        times  = sub["datetime"].values
        fc     = sub["forecast_lear"].values
        err    = sub["error"].values
        rmse   = float(np.sqrt(np.nanmean(err ** 2)))

        ax_price.plot(times, fc,
                      color=m["color"], linewidth=1.3, linestyle="--",
                      label=f"{m['name']}  (RMSE={rmse:.2f} \u20ac/MWh)")

        ax_err.plot(times, err,
                    color=m["color"], linewidth=0.9, alpha=0.85,
                    label=m["name"])

    ax_err.axhline(0, color="k", linewidth=0.6)

    if show_ylabel:
        ax_price.set_ylabel("Day-ahead Price (EUR/MWh)", fontsize=9)
        ax_err.set_ylabel("Error (EUR/MWh)", fontsize=9)
    ax_price.legend(fontsize=8, frameon=False)

    ax_price.set_title(col_title, fontsize=10, fontweight="bold")
    ax_price.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax_price.spines[["top", "right"]].set_visible(False)
    ax_err.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax_err.spines[["top", "right"]].set_visible(False)

    # Hour ticks for single days, day ticks for weeks
    if kind == "day":
        ax_err.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax_err.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    else:
        ax_err.xaxis.set_major_locator(mdates.DayLocator())
        ax_err.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
        ax_err.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b"))

    ax_err.tick_params(axis="x", which="major", labelsize=8)

    if start_dt is not None and end_dt is not None:
        ax_price.set_xlim(start_dt, end_dt)
        ax_err.set_xlim(start_dt, end_dt)


def _plot_side_by_side(all_data: dict, mm_dd: str, n_days: int,
                       title_suffix: str, kind: str):
    """Same period in 2020 (left) and 2024 (right); kind is "week" or "day"."""
    figsize = (22, 8) if kind == "week" else (18, 7)
    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(2, 2, height_ratios=[3, 1], hspace=0.08, wspace=0.12)

    ax_price = {2020: fig.add_subplot(gs[0, 0]),
                2024: fig.add_subplot(gs[0, 1])}
    ax_err   = {2020: fig.add_subplot(gs[1, 0], sharex=ax_price[2020]),
                2024: fig.add_subplot(gs[1, 1], sharex=ax_price[2024])}

    for col_i, year in enumerate([2020, 2024]):
        models = all_data.get(year)
        if not models:
            ax_price[year].set_visible(False)
            ax_err[year].set_visible(False)
            continue

        start_dt = pd.Timestamp(f"{year}-{mm_dd}")
        end_dt   = start_dt + pd.Timedelta(days=n_days) - pd.Timedelta(hours=1)

        _render_comparison_column(
            ax_price[year], ax_err[year],
            models, start_dt, end_dt,
            col_title=f"Hold-out {year}",
            kind=kind,
            show_ylabel=(col_i == 0),
        )
        # y labels only on the left column
        if col_i == 1:
            ax_price[year].tick_params(labelleft=False)
            ax_err[year].tick_params(labelleft=False)

        ax_price[year].tick_params(labelbottom=False)

    fig.suptitle(f"DK1 \u2014 Model Comparison: {title_suffix}", fontsize=11, y=1.01)
    fname = f"compare_{kind}_{mm_dd.replace('-','')}.png"
    _save(fig, OUT_DIR / fname)


def _plot_side_by_side_daily(all_data: dict):
    """Daily mean forecast vs actual over the whole year, 2020 left, 2024 right."""
    fig = plt.figure(figsize=(22, 7))
    gs  = gridspec.GridSpec(2, 2, height_ratios=[3, 1], hspace=0.08, wspace=0.12)

    ax_price = {2020: fig.add_subplot(gs[0, 0]),
                2024: fig.add_subplot(gs[0, 1])}
    ax_err   = {2020: fig.add_subplot(gs[1, 0]),
                2024: fig.add_subplot(gs[1, 1])}

    for col_i, year in enumerate([2020, 2024]):
        models = all_data.get(year)
        if not models:
            ax_price[year].set_visible(False)
            ax_err[year].set_visible(False)
            continue

        axp, axe = ax_price[year], ax_err[year]

        # Actual price, taken from the first model file
        actual_drawn = False
        for m in models:
            df = m["df"].copy()
            daily = (df.groupby("date")
                     .agg(actual=("actual", "mean"))
                     .reset_index())
            if daily.empty:
                continue
            axp.plot(daily["date"], daily["actual"],
                     color="#2166ac", linewidth=1.5,
                     label="Actual (realized)", zorder=10)
            actual_drawn = True
            break

        # Forecast and error of each model
        for m in models:
            df = m["df"].copy()
            daily = (df.groupby("date")
                     .agg(forecast=("forecast_lear", "mean"),
                          actual=("actual", "mean"),
                          error=("error", "mean"))
                     .reset_index())
            if daily.empty:
                continue

            rmse = float(np.sqrt(np.nanmean(df["error"] ** 2)))
            axp.plot(daily["date"], daily["forecast"],
                     color=m["color"], linewidth=1.1, linestyle="--",
                     label=f"{m['name']}  (RMSE={rmse:.2f} \u20ac/MWh)")
            axe.plot(daily["date"], daily["error"],
                     color=m["color"], linewidth=0.9, alpha=0.85)

        axe.axhline(0, color="k", linewidth=0.6)

        for ax in (axp, axe):
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            ax.tick_params(axis="x", which="major", labelsize=8)
            ax.grid(axis="y", linewidth=0.4, alpha=0.5)
            ax.spines[["top", "right"]].set_visible(False)

        axp.set_title(f"Hold-out {year} (daily means)", fontsize=10, fontweight="bold")
        axp.tick_params(labelbottom=False)

        axp.legend(fontsize=8, frameon=False)
        if col_i == 0:
            axp.set_ylabel("Day-ahead Price (EUR/MWh)", fontsize=9)
            axe.set_ylabel("Error (EUR/MWh)", fontsize=9)
        else:
            axp.tick_params(labelleft=False)
            axe.tick_params(labelleft=False)

    fig.suptitle("DK1 \u2014 Model Comparison: Full Hold-out Year", fontsize=11, y=1.01)
    _save(fig, OUT_DIR / "compare_full_daily.png")


def make_compare_plots():
    """All 2020 vs 2024 comparison figures (weeks, days, full year)."""
    print("\nComparison plots — loading model data...", flush=True)

    all_data: dict = {}
    for year in [2020, 2024]:
        loaded = []
        for display_name, tmpl, color in COMPARE_MODELS:
            if tmpl is None:
                label = _BASELINE.get(year, f"rolling_{year}_holidays")
            else:
                label = tmpl.replace("{year}", str(year))
            if label.startswith("lgbm:"):
                df = _load_hourly_lgbm(label[5:])
            else:
                df = _load_hourly(label)
            if df is None:
                continue
            df["datetime"] = (pd.to_datetime(df["date"])
                              + pd.to_timedelta(df["hour"], unit="h"))
            loaded.append({"name": display_name, "color": color, "df": df})
        if loaded:
            all_data[year] = loaded
            print(f"  {year}: {len(loaded)} model(s) loaded", flush=True)
        else:
            print(f"  {year}: no data found", flush=True)

    if not all_data:
        print("  No data for any year — skipping comparison plots.", flush=True)
        return

    # Week figures
    for mm_dd, n_days, title in COMPARE_PERIODS_WEEK:
        _plot_side_by_side(all_data, mm_dd, n_days, title, kind="week")

    # Day figures
    for mm_dd, title in COMPARE_PERIODS_DAY:
        _plot_side_by_side(all_data, mm_dd, 1, title, kind="day")

    # Full-year daily overview
    _plot_side_by_side_daily(all_data)

    print("  Done.", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("LEAR analysis — generating figures and Table 1", flush=True)
    print(f"Results source : {RESULTS_DIR}", flush=True)
    print(f"Output         : {OUT_DIR}", flush=True)
    print("=" * 60, flush=True)

    feat_names = build_feature_names(COEFS_CONFIG)
    n_ren   = sum(1 for n in feat_names if any(n.startswith(v) for v in ["onshore","offshore","solar"]))
    n_xzone = sum(1 for n in feat_names if any(n.startswith(f"price_{z}_lag") for z in CROSS_ZONES))
    print(f"\nFeature count from COEFS_CONFIG : {len(feat_names)}", flush=True)
    print(f"  Renewable feature columns     : {n_ren}", flush=True)
    print(f"  Cross-zone price columns      : {n_xzone}", flush=True)

    # Stop if the name list and core.py disagree on the number of regressors
    assert len(feat_names) == COEFS_CONFIG.n_features, (
        f"Feature count mismatch: build_feature_names={len(feat_names)} "
        f"vs ModelConfig.n_features={COEFS_CONFIG.n_features}"
    )

    make_figs_1_2(feat_names)
    make_fig3()
    make_fig4(feat_names)
    make_table1()
    make_compare_plots()

    print("\nAll done.", flush=True)
