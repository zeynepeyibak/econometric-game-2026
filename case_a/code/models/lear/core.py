"""
LEAR model: shared code used by validate.py, predict.py and models/lgbm/predict.py.

LEAR = LASSO-estimated autoregressive model (Lago et al. 2021, Uniejewski et
al. 2016). For each hour of the day a separate LASSO is fitted on lagged
prices, exogenous forecasts and day-of-week dummies. This is done on four
calibration windows (8 weeks, 12 weeks, 3 years, 4 years) and the four
forecasts are averaged.

Contents
  ModelConfig            which regressor blocks to include
  load_df_dict           lear_dataset.csv -> {date: arrays} lookup
  build_feature_row      regressor vector for one day
  fit_structural_models  seasonal OLS models that stand in for exogenous
                         forecasts not yet published (days 2-7 of the week)
  asinh_transform, fit_hour, forecast_one_window, forecast_one_day
                         estimation and forecasting

Equation and section numbers in comments refer to the case paper.
"""

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import holidays as holidays_pkg
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLarsIC, Lasso, LinearRegression

# Pre-built Danish holiday set covering the full data range
_DK_HOLIDAYS = holidays_pkg.Denmark(years=range(2015, 2030))

# ---------------------------------------------------------------------------
# Column name constants
# ---------------------------------------------------------------------------
PRICE_COLS = [f"price_h{h:02d}" for h in range(24)]
LOAD_COLS  = [f"load_forecast_h{h:02d}" for h in range(24)]
REN_COLS   = [f"ren_forecast_h{h:02d}" for h in range(24)]
WIND_COLS  = [f"wind_forecast_h{h:02d}" for h in range(24)]
SOLAR_COLS = [f"solar_forecast_h{h:02d}" for h in range(24)]
DOW_COLS   = [f"dow_{d}" for d in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]]
ENERGY_PRICE_COLS = ["coal_price", "eua_price", "natural_gas_price", "oil_price"]
MACRO_DAILY_COLS  = ["eurostoxx50_price", "msci_eu_price", "usd_eur_exchange", "vix"]
FLOW_IN_COLS  = [f"flow_in_h{h:02d}"  for h in range(24)]
FLOW_OUT_COLS = [f"flow_out_h{h:02d}" for h in range(24)]
FLOW_CHANNELS = ["germany", "dk2", "gb", "nl", "no2", "se3"]
FLOW_CHAN_COLS = {ch: [f"flow_{ch}_h{h:02d}" for h in range(24)] for ch in FLOW_CHANNELS}

DEFAULT_WINDOWS = {"8w": 56, "12w": 84, "3y": 1095, "4y": 1460}
REF_DATE        = pd.Timestamp("2018-01-01")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """Which regressors a LEAR variant uses.

    use_exogenous         add load and renewable forecasts (lags 0, 1, 7).
                          True gives the full 247-regressor model of eq. 16,
                          False the 103-regressor AR-only baseline.
    use_structural        for days 2-7 of the forecast week, replace the
                          exogenous forecasts (not yet published) with
                          predictions from the seasonal models of Section 3.1.
                          Without it those days have no exogenous inputs.
    use_holidays          add a Danish public holiday dummy (+1 regressor).
    use_split_renewables  use wind and solar separately instead of total
                          renewables (+72 regressors, 247 -> 319). With
                          use_structural, wind and solar get their own
                          seasonal models (Section 4.1).
    use_energy            coal, EUA, gas and oil prices at lags 1 and 7.
    use_macro_daily       EuroStoxx50, MSCI EU, USD/EUR and VIX at lags 1 and 7.
    use_transmission      cross-border flows at lags 1 and 7, either total
                          in/out (2 series) or, with use_transmission_channels,
                          net flow per neighbour (6 series).
    calibration_windows   {name: length in days}, default 8w/12w/3y/4y.
    train_cutoff          last training day, inclusive.
    label                 short name used in output file names.

    The energy, macro and transmission blocks are only added when
    use_exogenous is True.
    """
    use_exogenous: bool = True
    use_structural: bool = False
    use_holidays: bool = False
    use_split_renewables: bool = False
    use_energy: bool = False
    use_macro_daily: bool = False
    use_transmission: bool = False
    use_transmission_channels: bool = False
    calibration_windows: Dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_WINDOWS)
    )
    train_cutoff: str = "2023-12-31"
    label: str = "lear"

    @property
    def n_features(self) -> int:
        # Length of the regressor vector built by build_feature_row
        if self.use_exogenous:
            exog_per_lag = 72 if self.use_split_renewables else 48  # load+wind+solar vs load+ren
            base = 96 + 3 * exog_per_lag + 7
        else:
            base = 103
        if self.use_energy and self.use_exogenous:
            base += 4 * 2  # 4 prices x lags 1 and 7
        if self.use_macro_daily and self.use_exogenous:
            base += 4 * 2  # 4 series x lags 1 and 7
        if self.use_transmission and self.use_exogenous:
            n_trans_vars = 6 if self.use_transmission_channels else 2
            base += n_trans_vars * 2 * 24  # lags 1 and 7 only
        return base + (1 if self.use_holidays else 0)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_df_dict(data_path: Path) -> dict:
    """Read lear_dataset.csv into a dict keyed by "YYYY-MM-DD".

    Each value holds the day's arrays: prices, load, ren, wind, solar (24
    each), dow (7), energy (4), macro_daily (4), flow_in, flow_out (24 each)
    and flow_ch ({neighbour: 24}). Looking days up by string is much faster
    than filtering the DataFrame inside the estimation loops.
    """
    df = pd.read_csv(data_path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df_dict = {}
    for _, row in df.iterrows():
        ds = row["date"].strftime("%Y-%m-%d")
        df_dict[ds] = {
            "prices":   row[PRICE_COLS].values.astype(float),
            "load":     row[LOAD_COLS].values.astype(float),
            "ren":      row[REN_COLS].values.astype(float),
            "wind":     row[WIND_COLS].values.astype(float),
            "solar":    row[SOLAR_COLS].values.astype(float),
            "dow":      row[DOW_COLS].values.astype(float),
            "energy":      row[ENERGY_PRICE_COLS].values.astype(float),
            "macro_daily": row[MACRO_DAILY_COLS].values.astype(float),
            "flow_in":  row[FLOW_IN_COLS].values.astype(float),
            "flow_out": row[FLOW_OUT_COLS].values.astype(float),
            "flow_ch":  {ch: row[FLOW_CHAN_COLS[ch]].values.astype(float)
                         for ch in FLOW_CHANNELS},
        }
    return df_dict


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------
def get_price_vec(df_dict: dict, date_str: str, forecasted_prices: dict) -> np.ndarray:
    """Prices for one day. Uses our own earlier forecast if there is one
    (days inside the forecast week), otherwise the observed prices."""
    if date_str in forecasted_prices:
        return forecasted_prices[date_str].copy()
    rec = df_dict.get(date_str)
    return rec["prices"].copy() if rec is not None else np.full(24, np.nan)


def get_exog_vecs(df_dict: dict, date_str: str,
                  exog_override: dict) -> tuple:
    """Load and renewable forecasts for one day, taken from exog_override
    (structural-model predictions) when the day is in it."""
    if date_str in exog_override:
        return exog_override[date_str]["load"], exog_override[date_str]["ren"]
    rec = df_dict.get(date_str)
    if rec is None:
        return np.full(24, np.nan), np.full(24, np.nan)
    return rec["load"].copy(), rec["ren"].copy()


def get_exog_vecs_split(df_dict: dict, date_str: str,
                        exog_override: dict) -> tuple:
    """Same as get_exog_vecs, with wind and solar kept apart."""
    if date_str in exog_override:
        ov = exog_override[date_str]
        return ov["load"], ov["wind"], ov["solar"]
    rec = df_dict.get(date_str)
    if rec is None:
        return np.full(24, np.nan), np.full(24, np.nan), np.full(24, np.nan)
    return rec["load"].copy(), rec["wind"].copy(), rec["solar"].copy()


def get_transmission_vecs(df_dict: dict, date_str: str,
                          use_channels: bool) -> List[np.ndarray]:
    """Flow arrays for one day: [flow_in, flow_out], or with use_channels
    the net flow to each of the six neighbours."""
    rec = df_dict.get(date_str)
    if rec is None:
        n = 6 if use_channels else 2
        return [np.zeros(24)] * n  # treat missing as zero-flow
    if use_channels:
        return [rec["flow_ch"][ch].copy() for ch in FLOW_CHANNELS]
    return [rec["flow_in"].copy(), rec["flow_out"].copy()]


def build_feature_row(date_str: str, df_dict: dict,
                      forecasted_prices: dict, exog_override: dict,
                      dow_vec: np.ndarray, use_exogenous: bool = True,
                      use_holidays: bool = False,
                      use_split_renewables: bool = False,
                      use_energy: bool = False,
                      use_macro_daily: bool = False,
                      use_transmission: bool = False,
                      use_transmission_channels: bool = False) -> np.ndarray:
    """Regressor vector used to forecast the 24 prices of date_str.

    Order of the blocks (base case, 247 entries):
      0-95     prices of d-1, d-2, d-3, d-7
      96-239   load and renewables for d, d-1, d-7
      240-246  day-of-week dummies
    then, if switched on: energy prices, macro series, flows (lags 1 and 7),
    and the holiday dummy last. With use_split_renewables the exogenous block
    is load/wind/solar (216 entries, total 319). Without exogenous inputs the
    vector is just prices + DOW (103 entries).
    """
    d = pd.Timestamp(date_str)
    parts = []

    # AR blocks: lags 1, 2, 3, 7
    for lag in [1, 2, 3, 7]:
        lag_ds = (d - pd.Timedelta(days=lag)).strftime("%Y-%m-%d")
        parts.append(get_price_vec(df_dict, lag_ds, forecasted_prices))

    # Exogenous blocks: lags 0, 1, 7
    if use_exogenous:
        for lag in [0, 1, 7]:
            lag_ds = (d - pd.Timedelta(days=lag)).strftime("%Y-%m-%d")
            if use_split_renewables:
                lv, wv, sv = get_exog_vecs_split(df_dict, lag_ds, exog_override)
                parts.extend([lv, wv, sv])   # 72 dims per lag
            else:
                lv, rv = get_exog_vecs(df_dict, lag_ds, exog_override)
                parts.extend([lv, rv])        # 48 dims per lag

    parts.append(dow_vec)

    # Fuel/carbon prices: lags 1 and 7 only, since day-d prices are not known
    # when the forecast is made on d-1
    if use_energy and use_exogenous:
        for lag in [1, 7]:
            lag_ds = (d - pd.Timedelta(days=lag)).strftime("%Y-%m-%d")
            rec = df_dict.get(lag_ds)
            parts.append(rec["energy"].copy() if rec is not None else np.zeros(4))

    # Macro-finance daily block: lags 1 and 7 only
    if use_macro_daily and use_exogenous:
        for lag in [1, 7]:
            lag_ds = (d - pd.Timedelta(days=lag)).strftime("%Y-%m-%d")
            rec = df_dict.get(lag_ds)
            parts.append(rec["macro_daily"].copy() if rec is not None else np.zeros(4))

    # Flows: lags 1 and 7 only, for the same reason
    if use_transmission and use_exogenous:
        for lag in [1, 7]:
            lag_ds = (d - pd.Timedelta(days=lag)).strftime("%Y-%m-%d")
            parts.extend(get_transmission_vecs(df_dict, lag_ds, use_transmission_channels))

    # Danish public holiday dummy (1 = holiday, 0 = regular day)
    if use_holidays:
        parts.append(np.array([float(d.date() in _DK_HOLIDAYS)]))

    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Structural auxiliary models (Section 3.1) - used when use_structural=True
# ---------------------------------------------------------------------------
def fit_structural_models(df_dict: dict, train_dates: List[str],
                          forecast_dates: pd.DatetimeIndex,
                          use_split_renewables: bool = False) -> dict:
    """Seasonal OLS models for load and renewables, one per hour.

    The published day-ahead forecasts only exist for day 1 of the forecast
    week, so for days 2-7 we predict them from calendar terms instead. Returns
    {date: {"load": .., "ren": ..}} (or load/wind/solar) for forecast_dates[1:].

    All models are in logs; t = days since 2018-01-01, doy = day of year.
      load:  trend, 4-year cycle in t, 3 annual harmonics in doy, DOW,
             July dummy, Dec 18-31 dummy
      ren:   trend + 1 annual harmonic in t
      wind:  trend + 6 annual harmonics                   (Section 4.1)
      solar: quadratic trend + 3 harmonics + 2 harmonics interacted with t,
             zeros set to 0.01 before taking logs           (Section 4.1)
    """
    n     = len(train_dates)
    t_arr = np.array([(pd.Timestamp(d) - REF_DATE).days for d in train_dates], dtype=float)
    nf    = len(forecast_dates)
    t_f   = np.array([(d - REF_DATE).days for d in forecast_dates], dtype=float)

    # ---- Load design matrices (Section 4.1) ----
    # Business cycle: 4-year period in linear time t
    biz_sin_arr = np.sin(2 * np.pi * t_arr / (365.25 * 4))
    biz_cos_arr = np.cos(2 * np.pi * t_arr / (365.25 * 4))
    # Seasonal harmonics k=1,2,3 in day-of-year d
    doy_arr = np.array([pd.Timestamp(d).dayofyear for d in train_dates], dtype=float)
    sea_sin_arr = np.column_stack([np.sin(2*np.pi*k*doy_arr/365.25) for k in range(1, 4)])
    sea_cos_arr = np.column_stack([np.cos(2*np.pi*k*doy_arr/365.25) for k in range(1, 4)])
    # DOW, SummerBreak (July), WinterBreak (Dec 18-31)
    dow_arr      = np.array([df_dict[d]["dow"] for d in train_dates], dtype=float)
    summer_arr   = np.array([float(pd.Timestamp(d).month == 7) for d in train_dates])
    winter_arr   = np.array([float(pd.Timestamp(d).month == 12 and pd.Timestamp(d).day >= 18)
                              for d in train_dates])
    X_load = np.column_stack([np.ones(n), t_arr,
                               biz_sin_arr, biz_cos_arr,
                               sea_sin_arr, sea_cos_arr,
                               dow_arr, summer_arr, winter_arr])

    biz_sin_f = np.sin(2 * np.pi * t_f / (365.25 * 4))
    biz_cos_f = np.cos(2 * np.pi * t_f / (365.25 * 4))
    doy_f     = np.array([fd.dayofyear for fd in forecast_dates], dtype=float)
    sea_sin_f = np.column_stack([np.sin(2*np.pi*k*doy_f/365.25) for k in range(1, 4)])
    sea_cos_f = np.column_stack([np.cos(2*np.pi*k*doy_f/365.25) for k in range(1, 4)])
    dow_f     = np.zeros((nf, 7))
    for i, fd in enumerate(forecast_dates):
        dow_f[i, fd.dayofweek] = 1
    summer_f = np.array([float(fd.month == 7) for fd in forecast_dates])
    winter_f = np.array([float(fd.month == 12 and fd.day >= 18) for fd in forecast_dates])
    X_load_pred = np.column_stack([np.ones(nf), t_f,
                                    biz_sin_f, biz_cos_f,
                                    sea_sin_f, sea_cos_f,
                                    dow_f, summer_f, winter_f])

    # ---- Fit load ----
    load_mat    = np.array([df_dict[d]["load"] for d in train_dates], dtype=float)
    load_struct = np.zeros((nf, 24))
    lr = LinearRegression(fit_intercept=False)
    for h in range(24):
        y_l = np.log(np.maximum(load_mat[:, h], 1.0))
        ok  = np.isfinite(y_l)
        lr.fit(X_load[ok], y_l[ok])
        load_struct[:, h] = np.exp(np.clip(lr.predict(X_load_pred), -5, 15))

    if not use_split_renewables:
        # ---- Aggregate ren model: intercept + trend + 1 annual harmonic in t ----
        ren_sin_arr = np.sin(2 * np.pi * t_arr / 365.25)
        ren_cos_arr = np.cos(2 * np.pi * t_arr / 365.25)
        ren_sin_f   = np.sin(2 * np.pi * t_f   / 365.25)
        ren_cos_f   = np.cos(2 * np.pi * t_f   / 365.25)
        X_ren      = np.column_stack([np.ones(n),  t_arr, ren_sin_arr, ren_cos_arr])
        X_ren_pred = np.column_stack([np.ones(nf), t_f,   ren_sin_f,   ren_cos_f])
        ren_mat    = np.array([df_dict[d]["ren"] for d in train_dates], dtype=float)
        ren_struct = np.zeros((nf, 24))
        for h in range(24):
            y_r = np.log(np.maximum(ren_mat[:, h], 1.0))
            ok  = np.isfinite(y_r)
            lr.fit(X_ren[ok], y_r[ok])
            ren_struct[:, h] = np.exp(np.clip(lr.predict(X_ren_pred), -5, 15))

        override = {}
        for i, fd in enumerate(forecast_dates):
            if i >= 1:
                override[fd.strftime("%Y-%m-%d")] = {
                    "load": load_struct[i],
                    "ren":  ren_struct[i],
                }
        return override

    # ---- Wind model (Section 4.1): 6 harmonics, linear trend ----
    sin_k_arr  = np.column_stack([np.sin(2*np.pi*k*t_arr/365.25) for k in range(1, 7)])
    cos_k_arr  = np.column_stack([np.cos(2*np.pi*k*t_arr/365.25) for k in range(1, 7)])
    X_wind     = np.column_stack([np.ones(n), t_arr, sin_k_arr, cos_k_arr])
    sin_k_f    = np.column_stack([np.sin(2*np.pi*k*t_f/365.25) for k in range(1, 7)])
    cos_k_f    = np.column_stack([np.cos(2*np.pi*k*t_f/365.25) for k in range(1, 7)])
    X_wind_pred = np.column_stack([np.ones(nf), t_f, sin_k_f, cos_k_f])

    wind_mat    = np.array([df_dict[d]["wind"] for d in train_dates], dtype=float)
    wind_struct = np.zeros((nf, 24))
    for h in range(24):
        y_w = np.log(np.maximum(wind_mat[:, h], 1.0))
        ok  = np.isfinite(y_w)
        lr.fit(X_wind[ok], y_w[ok])
        wind_struct[:, h] = np.exp(np.clip(lr.predict(X_wind_pred), -5, 15))

    # ---- Solar model (Section 4.1): quadratic trend, 3 harmonics + 2 trend interactions ----
    sin3_arr   = np.column_stack([np.sin(2*np.pi*k*t_arr/365.25) for k in range(1, 4)])
    cos3_arr   = np.column_stack([np.cos(2*np.pi*k*t_arr/365.25) for k in range(1, 4)])
    sint_arr   = np.column_stack([np.sin(2*np.pi*k*t_arr/365.25)*t_arr for k in range(1, 3)])
    cost_arr   = np.column_stack([np.cos(2*np.pi*k*t_arr/365.25)*t_arr for k in range(1, 3)])
    X_solar    = np.column_stack([np.ones(n), t_arr, t_arr**2, sin3_arr, cos3_arr, sint_arr, cost_arr])

    sin3_f     = np.column_stack([np.sin(2*np.pi*k*t_f/365.25) for k in range(1, 4)])
    cos3_f     = np.column_stack([np.cos(2*np.pi*k*t_f/365.25) for k in range(1, 4)])
    sint_f     = np.column_stack([np.sin(2*np.pi*k*t_f/365.25)*t_f for k in range(1, 3)])
    cost_f     = np.column_stack([np.cos(2*np.pi*k*t_f/365.25)*t_f for k in range(1, 3)])
    X_solar_pred = np.column_stack([np.ones(nf), t_f, t_f**2, sin3_f, cos3_f, sint_f, cost_f])

    solar_mat    = np.array([df_dict[d]["solar"] for d in train_dates], dtype=float)
    solar_struct = np.zeros((nf, 24))
    for h in range(24):
        y_s = np.log(np.maximum(solar_mat[:, h], 0.01))  # night hours are 0
        ok  = np.isfinite(y_s)
        lr.fit(X_solar[ok], y_s[ok])
        solar_struct[:, h] = np.exp(np.clip(lr.predict(X_solar_pred), -5, 15))

    # Day 1 uses actual published forecasts; override only days 2 onwards
    override = {}
    for i, fd in enumerate(forecast_dates):
        if i >= 1:
            override[fd.strftime("%Y-%m-%d")] = {
                "load":  load_struct[i],
                "wind":  wind_struct[i],
                "solar": solar_struct[i],
            }
    return override


# ---------------------------------------------------------------------------
# Variance-stabilising transformation (eq. 18)
# ---------------------------------------------------------------------------
def asinh_transform(X_train: np.ndarray,
                    x_pred: np.ndarray) -> tuple:
    """asinh((x - median) / MAD) column by column.

    Median and MAD come from the training sample only and are reused for
    x_pred. Constant columns get MAD = 1 to avoid dividing by zero.
    """
    medians = np.nanmedian(X_train, axis=0)
    mads    = np.nanmedian(np.abs(X_train - medians), axis=0)
    mads[mads < 1e-10] = 1.0
    return (np.arcsinh((X_train - medians) / mads),
            np.arcsinh((x_pred   - medians) / mads))


# ---------------------------------------------------------------------------
# LASSO estimation for one hour (eq. 17)
# ---------------------------------------------------------------------------
def fit_hour(X_train_t: np.ndarray, y_train: np.ndarray,
             x_pred_t: np.ndarray) -> float:
    """Forecast one hourly price with LASSO.

    The price is asinh-transformed with its own median/MAD, lambda is chosen
    by AIC along the LARS path, and the model is refitted at that lambda with
    coordinate descent. The prediction is transformed back to EUR/MWh.

    LassoLarsIC cannot estimate the noise variance when there are fewer
    observations than regressors (8-week window), so in that case we pass the
    variance of y instead.
    """
    y_med = np.nanmedian(y_train)
    y_mad = np.nanmedian(np.abs(y_train - y_med))
    if y_mad < 1e-10:
        y_mad = 1.0
    y_t = np.arcsinh((y_train - y_med) / y_mad)

    try:
        lars = LassoLarsIC(criterion="aic", fit_intercept=False)
        lars.fit(X_train_t, y_t)
    except Exception:
        noise_var = max(float(np.var(y_t)), 1e-4)
        lars = LassoLarsIC(criterion="aic", fit_intercept=False,
                           noise_variance=noise_var)
        lars.fit(X_train_t, y_t)

    alpha_opt = max(float(lars.alpha_), 1e-8)
    lasso = Lasso(alpha=alpha_opt, fit_intercept=False, max_iter=20000, tol=1e-5)
    lasso.fit(X_train_t, y_t)
    y_hat_t = float(lasso.predict(x_pred_t.reshape(1, -1))[0])
    return float(np.sinh(y_hat_t) * y_mad + y_med)


# ---------------------------------------------------------------------------
# Single-window forecast for one day
# ---------------------------------------------------------------------------
def forecast_one_window(forecast_date: pd.Timestamp,
                        win_days: int,
                        all_dates_sorted: List[str],
                        df_dict: dict,
                        forecasted_prices: dict,
                        exog_override: dict,
                        dow_vec: np.ndarray,
                        x_pred: np.ndarray,
                        use_exogenous: bool,
                        use_holidays: bool = False,
                        use_split_renewables: bool = False,
                        use_energy: bool = False,
                        use_macro_daily: bool = False,
                        use_transmission: bool = False,
                        use_transmission_channels: bool = False) -> np.ndarray:
    """Fit the 24 hourly models on the win_days before forecast_date and
    return the 24 forecasts (NaN where there were fewer than 20 usable days).

    Training rows always use observed data, never our own forecasts.
    """
    cutoff_end   = (forecast_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    cutoff_start = (forecast_date - pd.Timedelta(days=win_days)).strftime("%Y-%m-%d")

    train_dates = [
        d for d in all_dates_sorted
        if cutoff_start <= d <= cutoff_end
        and (pd.Timestamp(d) - REF_DATE).days >= 7   # need 7 days of lags
    ]

    X_rows, y_rows = [], {h: [] for h in range(24)}
    for td in train_dates:
        xr = build_feature_row(td, df_dict,
                               forecasted_prices={}, exog_override={},
                               dow_vec=df_dict[td]["dow"],
                               use_exogenous=use_exogenous,
                               use_holidays=use_holidays,
                               use_split_renewables=use_split_renewables,
                               use_energy=use_energy,
                               use_macro_daily=use_macro_daily,
                               use_transmission=use_transmission,
                               use_transmission_channels=use_transmission_channels)
        if not np.isfinite(xr).all():
            continue
        X_rows.append(xr)
        for h in range(24):
            y_rows[h].append(df_dict[td]["prices"][h])

    preds = np.full(24, np.nan)
    if len(X_rows) < 20 or not np.isfinite(x_pred).all():
        return preds

    X_train = np.array(X_rows, dtype=float)
    X_train_t, x_pred_t = asinh_transform(X_train, x_pred)

    for h in range(24):
        y_train = np.array(y_rows[h], dtype=float)
        valid   = np.isfinite(y_train)
        if valid.sum() < 20:
            continue
        try:
            preds[h] = fit_hour(X_train_t[valid], y_train[valid], x_pred_t)
        except Exception:
            pass

    return preds


# ---------------------------------------------------------------------------
# Full ensemble forecast for one day
# ---------------------------------------------------------------------------
def forecast_one_day(forecast_date: pd.Timestamp,
                     config: ModelConfig,
                     all_dates_sorted: List[str],
                     df_dict: dict,
                     forecasted_prices: dict,
                     exog_override: dict) -> Dict[str, np.ndarray]:
    """Forecast one day with every calibration window.

    Returns {window name: 24 forecasts} plus "ensemble", the plain average
    across windows (windows that failed are skipped by nanmean).
    """
    dow_vec = np.zeros(7)
    dow_vec[forecast_date.dayofweek] = 1
    ds = forecast_date.strftime("%Y-%m-%d")

    x_pred = build_feature_row(ds, df_dict, forecasted_prices, exog_override,
                               dow_vec, use_exogenous=config.use_exogenous,
                               use_holidays=config.use_holidays,
                               use_split_renewables=config.use_split_renewables,
                               use_energy=config.use_energy,
                               use_macro_daily=config.use_macro_daily,
                               use_transmission=config.use_transmission,
                               use_transmission_channels=config.use_transmission_channels)

    results = {}
    for win_name, win_days in config.calibration_windows.items():
        results[win_name] = forecast_one_window(
            forecast_date, win_days, all_dates_sorted,
            df_dict, forecasted_prices, exog_override,
            dow_vec, x_pred, config.use_exogenous, config.use_holidays,
            config.use_split_renewables, config.use_energy, config.use_macro_daily,
            config.use_transmission, config.use_transmission_channels,
        )

    stacked  = np.vstack([results[wn] for wn in config.calibration_windows])
    ensemble = np.nanmean(stacked, axis=0)
    results["ensemble"] = ensemble
    return results
