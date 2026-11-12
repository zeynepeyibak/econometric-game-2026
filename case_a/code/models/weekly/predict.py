"""
Weekly average price forecast for all 52 weeks of 2024 (Section 3.8).

The weekly mean DK1 price is regressed, with LASSO, on
  - K pairs of annual Fourier terms in the week number (K = 1, 2 or 3,
    chosen by AIC),
  - seven fundamentals divided by their yearly mean (eq. 10): gas, CO2,
    wind, temperature, CPI, GDP, EuroStoxx50,
  - the normalised price in the same week of 2023 and of 2022 (eq. 9).
The LASSO penalty is chosen by expanding-window CV (one week ahead, at
least 52 weeks of training). Estimation sample: 2018-2023, 312 weeks.

For 2024 the fundamentals are not known in advance, so their 2023 values
for the same week are used.

Benchmark: OLS on the Fourier terms only (Wagner et al. 2022).

Run:     python predict.py      (from models/weekly/)
Input:   data/prepared/lear_dataset.csv  (prices, wind, fuel, stocks)
         data/prepared/dataset.csv       (temperature, CPI, GDP)
Output:  results/weekly/weekly_forecast_2024.csv
         results/weekly/weekly_forecast_summary.txt
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent.parent.parent
LEAR_CSV = ROOT / "data/prepared/lear_dataset.csv"
DS_CSV = ROOT / "data/prepared/dataset.csv"
OUT_DIR = ROOT / "results/weekly"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Step 1 - Load daily data from lear_dataset.csv
# ---------------------------------------------------------------------------
print("Loading lear_dataset.csv ...", flush=True)
lear = pd.read_csv(LEAR_CSV, parse_dates=["date"])

price_cols = [f"price_h{h:02d}" for h in range(24)]
wind_cols = [f"wind_forecast_h{h:02d}" for h in range(24)]

lear["daily_price"] = lear[price_cols].mean(axis=1)
lear["daily_wind"] = lear[wind_cols].mean(axis=1)

# A few missing values at the start of the sample
lear["natural_gas_price"] = lear["natural_gas_price"].ffill()
lear["eua_price"] = lear["eua_price"].ffill()
lear["eurostoxx50_price"] = lear["eurostoxx50_price"].ffill()

daily_lear = lear[["date", "daily_price", "daily_wind",
                    "natural_gas_price", "eua_price", "eurostoxx50_price"]].copy()

# ---------------------------------------------------------------------------
# Step 2 - Load temperature, CPI, GDP from dataset.csv (hourly -> daily)
# ---------------------------------------------------------------------------
print("Loading dataset.csv ...", flush=True)
ds = pd.read_csv(DS_CSV, parse_dates=["time"])
ds["date"] = ds["time"].dt.normalize()

tg_cols = [c for c in ds.columns if c.startswith("tg_")]
ds["daily_temp"] = ds[tg_cols].mean(axis=1)

# CPI and GDP are monthly/quarterly, so any hour of the day gives the same value
daily_ds = (
    ds.groupby("date")
    .agg(
        daily_temp=("daily_temp", "mean"),
        cpi=("Monthly_DNKCPIHICMINMEI", "first"),
        gdp=("Quarterly_CLVMNACNSAB1GQDK", "first"),
    )
    .reset_index()
)
# Carry the latest release forward
daily_ds["cpi"] = daily_ds["cpi"].ffill()
daily_ds["gdp"] = daily_ds["gdp"].ffill()

# ---------------------------------------------------------------------------
# Step 3 - Merge and assign ISO weeks (week 53 is merged into week 52)
# ---------------------------------------------------------------------------
daily = daily_lear.merge(daily_ds, on="date", how="left")

# Use the ISO year, not the calendar year: e.g. 31 Dec 2018 is in week 1 of
# 2019 and 1 Jan 2023 in week 52 of 2022.
daily["year"] = daily["date"].apply(lambda d: d.isocalendar()[0])
daily["iso_week"] = daily["date"].apply(lambda d: d.isocalendar()[1])
daily["week_num"] = daily["iso_week"].clip(upper=52)

# ---------------------------------------------------------------------------
# Step 4 - Aggregate to weekly
# ---------------------------------------------------------------------------
weekly = (
    daily.groupby(["year", "week_num"])
    .agg(
        price=("daily_price", "mean"),
        gas=("natural_gas_price", "mean"),
        co2=("eua_price", "mean"),
        wind=("daily_wind", "mean"),
        temp=("daily_temp", "mean"),
        cpi=("cpi", "mean"),
        gdp=("gdp", "mean"),
        stoxx=("eurostoxx50_price", "mean"),
    )
    .reset_index()
    .sort_values(["year", "week_num"])
    .reset_index(drop=True)
)

print(f"  Total weekly rows: {len(weekly)}", flush=True)

# ---------------------------------------------------------------------------
# Step 5 - Normalize each variable by yearly mean (eq. 10)
# ---------------------------------------------------------------------------
fund_vars = ["gas", "co2", "wind", "temp", "cpi", "gdp", "stoxx"]

for var in fund_vars + ["price"]:
    yearly_mean = weekly.groupby("year")[var].transform("mean")
    weekly[f"{var}_norm"] = weekly[var] / yearly_mean

# ---------------------------------------------------------------------------
# Step 6 - Price in the same week of 2023 and 2022 (normalised)
# Note: these are attached to every year, so in 2018-2022 the regressor is a
# later year's price.
# ---------------------------------------------------------------------------
price_2023 = weekly.loc[weekly["year"] == 2023].set_index("week_num")["price_norm"]
price_2022 = weekly.loc[weekly["year"] == 2022].set_index("week_num")["price_norm"]

weekly["ar_2023"] = weekly["week_num"].map(price_2023)
weekly["ar_2022"] = weekly["week_num"].map(price_2022)

# ---------------------------------------------------------------------------
# Step 7 - Split estimation sample (2018-2023) and 2024
# ---------------------------------------------------------------------------
est = weekly[weekly["year"].between(2018, 2023)].copy().reset_index(drop=True)
fwd = weekly[weekly["year"] == 2024].copy().reset_index(drop=True)

print(f"  Estimation sample: N={len(est)} weeks (expected 312)", flush=True)

y_est = est["price"].values  # target is the price level in EUR/MWh, not normalised

# ---------------------------------------------------------------------------
# Step 8 - Number of Fourier pairs K, by AIC of an OLS on Fourier terms only
# ---------------------------------------------------------------------------
def fourier_matrix(weeks, K):
    """sin and cos of 2*pi*k*week/52 for k = 1..K, as an (n, 2K) matrix."""
    cols = []
    for k in range(1, K + 1):
        cols.append(np.sin(2 * np.pi * k * weeks / 52))
        cols.append(np.cos(2 * np.pi * k * weeks / 52))
    return np.column_stack(cols)

w_est = est["week_num"].values
n_est = len(est)

best_K, best_aic = None, np.inf
for K in [1, 2, 3]:
    F = fourier_matrix(w_est, K)
    X_ols = np.column_stack([np.ones(n_est), F])
    coef, _, _, _ = np.linalg.lstsq(X_ols, y_est, rcond=None)
    resid = y_est - X_ols @ coef
    rss = np.sum(resid ** 2)
    p_ols = 2 * K + 1  # intercept + 2K Fourier
    aic = n_est * np.log(rss / n_est) + 2 * p_ols
    print(f"  K={K}: AIC={aic:.2f}", flush=True)
    if aic < best_aic:
        best_aic, best_K = aic, K

print(f"  Selected K={best_K} (AIC={best_aic:.2f})", flush=True)

# ---------------------------------------------------------------------------
# Step 9 - LASSO design matrix: Fourier (2K) | fundamentals (7) | past prices (2)
# The intercept is fitted by sklearn and not penalised.
# ---------------------------------------------------------------------------
F_est = fourier_matrix(w_est, best_K)
fund_norm_cols = [f"{v}_norm" for v in fund_vars]
X_fund = est[fund_norm_cols].values
X_ar = est[["ar_2023", "ar_2022"]].values

X_raw = np.hstack([F_est, X_fund, X_ar])

# Standardise so the penalty treats all regressors alike
scaler = StandardScaler()
X_est = scaler.fit_transform(X_raw)

# ---------------------------------------------------------------------------
# Step 10 - Choose lambda: train on weeks 1..t-1, predict week t, for every t
# after the first 52 weeks; sum the squared errors per lambda.
# (The scaler was fitted on the full sample, so the folds are not fully clean.)
# ---------------------------------------------------------------------------
lambda_grid = np.logspace(-4, 3, 140)
min_train = 52
cv_mse = np.zeros(len(lambda_grid))
n_folds = n_est - min_train

print(f"  Running expanding-window CV ({n_folds} folds) ...", flush=True)
for t in range(min_train, n_est):
    X_tr, y_tr = X_est[:t], y_est[:t]
    X_val, y_val = X_est[t : t + 1], y_est[t : t + 1]
    for j, lam in enumerate(lambda_grid):
        mdl = Lasso(alpha=lam, fit_intercept=True, max_iter=20000, tol=1e-4)
        mdl.fit(X_tr, y_tr)
        cv_mse[j] += (mdl.predict(X_val)[0] - y_val[0]) ** 2

best_lambda = lambda_grid[np.argmin(cv_mse)]
print(f"  Best λ={best_lambda:.6f}  (CV MSE={cv_mse.min()/n_folds:.2f})", flush=True)

# Refit on the whole estimation sample at the chosen lambda
final_model = Lasso(alpha=best_lambda, fit_intercept=True, max_iter=20000)
final_model.fit(X_est, y_est)

coef_names = (
    [f"sin_{k}" if i % 2 == 0 else f"cos_{k}"
     for k in range(1, best_K + 1) for i in range(2)]
    + fund_norm_cols
    + ["ar_2023", "ar_2022"]
)
print(f"  Intercept: {final_model.intercept_:+.4f}", flush=True)
print("  Non-zero coefficients:", flush=True)
for name, coef in zip(coef_names, final_model.coef_):
    if abs(coef) > 1e-9:
        print(f"    {name:30s} {coef:+.4f}", flush=True)

# ---------------------------------------------------------------------------
# Step 11 - Benchmark: Fourier-only OLS
# ---------------------------------------------------------------------------
X_bench_est = F_est
benchmark = LinearRegression(fit_intercept=True).fit(X_bench_est, y_est)

# ---------------------------------------------------------------------------
# Step 12 - 2024 regressors: fundamentals from the same week of 2023 (already
# normalised by the 2023 mean); past prices from 2023 and 2022
# ---------------------------------------------------------------------------
w_2024 = np.arange(1, 53)  # weeks 1-52

# 2023 normalized fundamentals, indexed by week
est_2023 = est[est["year"] == 2023].set_index("week_num")

def get_2023_norm(var, weeks):
    """Normalised 2023 values of var for the given weeks (NaN if missing)."""
    col = f"{var}_norm"
    return np.array([est_2023.loc[w, col] if w in est_2023.index else np.nan
                     for w in weeks])

F_2024 = fourier_matrix(w_2024, best_K)
X_fund_2024 = np.column_stack([get_2023_norm(v, w_2024) for v in fund_vars])
ar_2023_2024 = get_2023_norm("price", w_2024)
ar_2022_2024 = np.array([price_2022.get(w, np.nan) for w in w_2024])

X_raw_2024 = np.hstack([F_2024, X_fund_2024,
                         ar_2023_2024.reshape(-1, 1), ar_2022_2024.reshape(-1, 1)])
X_2024 = scaler.transform(X_raw_2024)
X_bench_2024 = F_2024

fc_lasso = final_model.predict(X_2024)
fc_bench = benchmark.predict(X_bench_2024)

# ---------------------------------------------------------------------------
# Step 13 - Realised 2024 weekly prices, if the data cover 2024
# ---------------------------------------------------------------------------
if len(fwd) > 0:
    actual_2024 = fwd.set_index("week_num")["price"].reindex(w_2024).values
else:
    actual_2024 = np.full(52, np.nan)

# ---------------------------------------------------------------------------
# Step 14 - Save results
# ---------------------------------------------------------------------------
out_df = pd.DataFrame({
    "week": w_2024,
    "forecast_lasso": fc_lasso,
    "forecast_benchmark": fc_bench,
    "actual": actual_2024,
})
out_path = OUT_DIR / "weekly_forecast_2024.csv"
out_df.to_csv(out_path, index=False)
print(f"\nSaved forecasts → {out_path}", flush=True)

# Summary metrics
rmse = lambda a, b: np.sqrt(np.nanmean((a - b) ** 2))
mae = lambda a, b: np.nanmean(np.abs(a - b))

rmse_lasso = rmse(fc_lasso, actual_2024)
mae_lasso = mae(fc_lasso, actual_2024)
rmse_bench = rmse(fc_bench, actual_2024)
mae_bench = mae(fc_bench, actual_2024)

summary = (
    f"Weekly Price Forecasting — 2024 Out-of-Sample Results\n"
    f"{'='*55}\n"
    f"Estimation sample : N={n_est} (weeks 1-52, 2018-2023)\n"
    f"Selected K        : {best_K} (AIC)\n"
    f"Best λ (LASSO)    : {best_lambda:.6f}\n\n"
    f"{'Model':<35} {'RMSE':>8} {'MAE':>8}\n"
    f"{'-'*55}\n"
    f"{'LASSO (full model)':<35} {rmse_lasso:>8.2f} {mae_lasso:>8.2f}\n"
    f"{'Benchmark (Fourier OLS)':<35} {rmse_bench:>8.2f} {mae_bench:>8.2f}\n"
)
print("\n" + summary, flush=True)
summary_path = OUT_DIR / "weekly_forecast_summary.txt"
summary_path.write_text(summary)
print(f"Saved summary  → {summary_path}", flush=True)
print("Done.", flush=True)
