"""
Case B LEAR model: shared code for rolling.py and static.py.

Same idea as the Case A model (one LASSO per hour on lagged prices, exogenous
forecasts and day-of-week dummies, asinh-transformed), with three changes:
  - one 2-year calibration window instead of the average over four windows;
  - DK1's own exogenous block has four series (load, onshore, offshore,
    solar), which gives the 391-regressor baseline of paper Section 3.1:
      96  prices of d-1, d-2, d-3, d-7
      288 load/onshore/offshore/solar of d, d-1, d-7
      7   day-of-week dummies
  - optional blocks from Section 3.2 and beyond: neighbouring-zone prices and
    renewables, flows on DK1's borders and on the neighbours' borders, energy
    and macro prices, holidays, and a set of net-load/hydro/temperature terms.

Past prices are always the observed ones: in day-ahead forecasting every
earlier price is known when the forecast is made.
"""

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLarsIC, Lasso

# ---------------------------------------------------------------------------
# Column names in Case B lear_dataset.csv
# ---------------------------------------------------------------------------
CROSS_ZONES = ["dk2", "no2", "se4", "de"]

PRICE_COLS       = [f"price_h{h:02d}"            for h in range(24)]
LOAD_COLS        = [f"load_forecast_h{h:02d}"    for h in range(24)]
WIND_COLS        = [f"wind_forecast_h{h:02d}"    for h in range(24)]
SOLAR_COLS       = [f"solar_forecast_h{h:02d}"   for h in range(24)]
OFFSHORE_COLS    = [f"offshore_forecast_h{h:02d}" for h in range(24)]
ONSHORE_COLS     = [f"onshore_forecast_h{h:02d}"  for h in range(24)]
DOW_COLS         = [f"dow_{d}" for d in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]]
ENERGY_COLS      = ["coal_price", "eua_price", "natural_gas_price", "oil_price"]
MACRO_DAILY_COLS = ["eurostoxx50_price", "msci_eu_price", "usd_eur_exchange", "vix"]

# Neighbouring zones: prices and total renewable forecasts
PRICE_XZ_COLS = {z: [f"price_{z}_h{h:02d}"        for h in range(24)] for z in CROSS_ZONES}
REN_XZ_COLS   = {z: [f"ren_forecast_{z}_h{h:02d}" for h in range(24)] for z in CROSS_ZONES}

# DK1's four main interconnectors (paper Section 3.2)
DK1_CHANNELS  = ["de_lu", "dk2", "no2", "se3"]
FLOW_DK1_COLS = {ch: [f"flow_dk1_{ch}_h{h:02d}" for h in range(24)] for ch in DK1_CHANNELS}
FLOW_DK1_IN   = [f"flow_dk1_in_h{h:02d}"  for h in range(24)]
FLOW_DK1_OUT  = [f"flow_dk1_out_h{h:02d}" for h in range(24)]

# Borders of the neighbouring zones, as they appear in the dataset (27 in total)
CROSS_ZONE_FLOWS = {
    "dk2": ["de_lu", "dk1", "nl", "se4"],
    "no2": ["de_lu", "dk1", "gb", "nl", "no1", "no2a", "no5"],
    "se4": ["de_lu", "dk2", "lt", "pl", "se3"],
    "de":  ["at", "be", "ch", "cz", "dk1", "dk2", "fr", "nl", "no2", "pl", "se4"],
}
FLOW_XZ_COLS = {
    (z, ch): [f"flow_{z}_{ch}_h{h:02d}" for h in range(24)]
    for z, channels in CROSS_ZONE_FLOWS.items()
    for ch in channels
}
N_XZ_CHANNELS = sum(len(chs) for chs in CROSS_ZONE_FLOWS.values())  # 27

# DE and DK2 load/wind/solar, used for the net-load terms
LOAD_DE_COLS   = [f"load_forecast_de_h{h:02d}"  for h in range(24)]
WIND_DE_COLS   = [f"wind_forecast_de_h{h:02d}"  for h in range(24)]
SOLAR_DE_COLS  = [f"solar_forecast_de_h{h:02d}" for h in range(24)]
LOAD_DK2_COLS  = [f"load_forecast_dk2_h{h:02d}" for h in range(24)]
WIND_DK2_COLS  = [f"wind_forecast_dk2_h{h:02d}" for h in range(24)]
SOLAR_DK2_COLS = [f"solar_forecast_dk2_h{h:02d}" for h in range(24)]

# Daily weather series (from build_lear_dataset.py)
TEMP_DK1_COL        = "temp_dk1"          # 0.1 C units; 180 = 18 C is the comfort level
PRECIP_NO2_ROLL_COL = "precip_no2_roll14"  # 0.1 mm

REF_DATE = pd.Timestamp("2018-01-01")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """Which regressors a Case B LEAR variant uses (number added in brackets).

    use_exogenous                DK1 load/onshore/offshore/solar; False gives
                                 the AR-only model (103 regressors)
    use_cross_zone_prices        DK2, NO2, SE4, DE prices at d-1, d-7     (+192)
    use_cross_zone_ren           their renewable forecasts at d, d-1, d-7 (+288)
    use_dk1_transmission         flows on DK1's four borders, d-1, d-7    (+192)
    use_cross_zone_transmission  flows on the neighbours' 27 borders      (+1296)
    use_holidays                 Danish holiday dummy                       (+1)
    use_energy                   coal/EUA/gas/oil at d-1, d-7               (+8)
    use_macro_daily              EuroStoxx50/MSCI/USD-EUR/VIX at d-1, d-7   (+8)
    use_interactions             net-load, hydro and temperature terms    (+121)
    window_days                  calibration window in days (730 = 2 years)
    train_cutoff                 last training day, inclusive
    label                        short name used in output file names
    """
    use_exogenous: bool                = True
    use_cross_zone_prices: bool        = False
    use_cross_zone_ren: bool           = False
    use_dk1_transmission: bool         = False
    use_cross_zone_transmission: bool  = False
    use_holidays: bool                 = False
    use_energy: bool                   = False
    use_macro_daily: bool              = False
    use_interactions: bool             = False
    window_days: int                   = 730
    train_cutoff: str                  = "2019-12-31"
    label: str                         = "lear"

    @property
    def n_features(self) -> int:
        # Length of the vector built by build_feature_row
        if not self.use_exogenous:
            return 96 + 7 + (1 if self.use_holidays else 0)
        base = 96 + 4 * 24 * 3 + 7          # 391: AR + (load,onshore,offshore,solar)x3lags + DOW
        if self.use_cross_zone_prices:        base += 4 * 24 * 2
        if self.use_cross_zone_ren:           base += 4 * 24 * 3
        if self.use_dk1_transmission:         base += 4 * 24 * 2
        if self.use_cross_zone_transmission:  base += N_XZ_CHANNELS * 24 * 2
        if self.use_energy:                   base += 4 * 2
        if self.use_macro_daily:              base += 4 * 2
        if self.use_holidays:                 base += 1
        if self.use_interactions:             base += 24 * 5 + 1   # 121
        return base

    @property
    def calibration_windows(self) -> Dict[str, int]:
        # Same interface as Case A, which has several windows
        return {"2y": self.window_days}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_df_dict(data_path: Path) -> dict:
    """Read lear_dataset.csv into a dict keyed by "YYYY-MM-DD".

    Per day: DK1 prices, load, wind, solar, offshore, onshore (24 values
    each); dow (7); energy and macro_daily (4 each); is_holiday; DK1 flows
    (flow_dk1_in/out and flow_dk1_ch per border); neighbouring-zone prices,
    renewables and flows (prices_xz, ren_xz, flow_xz); DE and DK2
    load/wind/solar; temp_dk1 and precip_no2_r14.

    If a column is missing from the file (older version of the dataset), the
    entry is filled with zeros or NaN instead of failing.
    """
    df = pd.read_csv(data_path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Returns the column list only if all columns are in the file
    def _safe_cols(cols):
        present = [c for c in cols if c in df.columns]
        if len(present) < len(cols):
            return None
        return cols

    offshore_cols = _safe_cols(OFFSHORE_COLS)
    onshore_cols  = _safe_cols(ONSHORE_COLS)

    df_dict = {}
    for _, row in df.iterrows():
        ds = row["date"].strftime("%Y-%m-%d")

        entry = {
            "prices":      row[PRICE_COLS].values.astype(float),
            "load":        row[LOAD_COLS].values.astype(float),
            "wind":        row[WIND_COLS].values.astype(float),
            "solar":       row[SOLAR_COLS].values.astype(float),
            "offshore":    row[offshore_cols].values.astype(float) if offshore_cols else
                           np.full(24, np.nan),
            "onshore":     row[onshore_cols].values.astype(float) if onshore_cols else
                           np.full(24, np.nan),
            "dow":         row[DOW_COLS].values.astype(float),
            "energy":      row[ENERGY_COLS].values.astype(float)
                           if all(c in df.columns for c in ENERGY_COLS)
                           else np.zeros(4),
            "macro_daily": row[MACRO_DAILY_COLS].values.astype(float)
                           if all(c in df.columns for c in MACRO_DAILY_COLS)
                           else np.zeros(4),
            "is_holiday":  float(row["is_holiday"]) if "is_holiday" in df.columns else 0.0,
            "flow_dk1_in":  row[FLOW_DK1_IN].values.astype(float)
                            if all(c in df.columns for c in FLOW_DK1_IN) else np.zeros(24),
            "flow_dk1_out": row[FLOW_DK1_OUT].values.astype(float)
                            if all(c in df.columns for c in FLOW_DK1_OUT) else np.zeros(24),
            "flow_dk1_ch": {
                ch: row[FLOW_DK1_COLS[ch]].values.astype(float)
                if all(c in df.columns for c in FLOW_DK1_COLS[ch]) else np.zeros(24)
                for ch in DK1_CHANNELS
            },
            "prices_xz": {
                z: row[PRICE_XZ_COLS[z]].values.astype(float)
                if all(c in df.columns for c in PRICE_XZ_COLS[z]) else np.full(24, np.nan)
                for z in CROSS_ZONES
            },
            "ren_xz": {
                z: row[REN_XZ_COLS[z]].values.astype(float)
                if all(c in df.columns for c in REN_XZ_COLS[z]) else np.full(24, np.nan)
                for z in CROSS_ZONES
            },
            "flow_xz": {
                (z, ch): row[FLOW_XZ_COLS[(z, ch)]].values.astype(float)
                if all(c in df.columns for c in FLOW_XZ_COLS[(z, ch)]) else np.zeros(24)
                for z, channels in CROSS_ZONE_FLOWS.items()
                for ch in channels
            },
            # For the net-load terms
            "load_de":   row[LOAD_DE_COLS].values.astype(float)
                         if all(c in df.columns for c in LOAD_DE_COLS)   else np.zeros(24),
            "wind_de":   row[WIND_DE_COLS].values.astype(float)
                         if all(c in df.columns for c in WIND_DE_COLS)   else np.zeros(24),
            "solar_de":  row[SOLAR_DE_COLS].values.astype(float)
                         if all(c in df.columns for c in SOLAR_DE_COLS)  else np.zeros(24),
            "load_dk2":  row[LOAD_DK2_COLS].values.astype(float)
                         if all(c in df.columns for c in LOAD_DK2_COLS)  else np.zeros(24),
            "wind_dk2":  row[WIND_DK2_COLS].values.astype(float)
                         if all(c in df.columns for c in WIND_DK2_COLS)  else np.zeros(24),
            "solar_dk2": row[SOLAR_DK2_COLS].values.astype(float)
                         if all(c in df.columns for c in SOLAR_DK2_COLS) else np.zeros(24),
            # Weather (defaults: 18 C and no rain)
            "temp_dk1":       float(row[TEMP_DK1_COL])
                              if TEMP_DK1_COL in df.columns else 180.0,
            "precip_no2_r14": float(row[PRECIP_NO2_ROLL_COL])
                              if PRECIP_NO2_ROLL_COL in df.columns else 0.0,
        }
        df_dict[ds] = entry

    return df_dict


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------
def get_price_vec(df_dict: dict, date_str: str, forecasted_prices: dict = None) -> np.ndarray:
    """Observed DK1 prices for one day (NaN if the day is not in the data).

    forecasted_prices is not used; it is kept so the signature matches
    Case A, where forecasts are fed back as lags.
    """
    rec = df_dict.get(date_str)
    return rec["prices"].copy() if rec is not None else np.full(24, np.nan)


def build_feature_row(date_str: str,
                      df_dict: dict,
                      forecasted_prices: dict,
                      dow_vec: np.ndarray,
                      config: ModelConfig) -> np.ndarray:
    """Regressor vector used to forecast the 24 DK1 prices of date_str.

    Always (391 entries):
      0-95     prices of d-1, d-2, d-3, d-7
      96-383   load, onshore, offshore, solar for d, d-1, d-7
      384-390  day-of-week dummies
    Then, if switched on in config, in this order: neighbour prices,
    neighbour renewables, DK1 border flows, neighbour border flows, energy,
    macro, holiday dummy, interaction terms.
    """
    d = pd.Timestamp(date_str)
    parts = []

    # AR blocks: lags 1, 2, 3, 7
    for lag in [1, 2, 3, 7]:
        lag_ds = (d - pd.Timedelta(days=lag)).strftime("%Y-%m-%d")
        parts.append(get_price_vec(df_dict, lag_ds, forecasted_prices))

    # Own exog blocks: load, onshore, offshore, solar x lags 0, 1, 7
    if config.use_exogenous:
        for lag in [0, 1, 7]:
            lag_ds = (d - pd.Timedelta(days=lag)).strftime("%Y-%m-%d")
            rec = df_dict.get(lag_ds)
            parts.append(rec["load"]     if rec is not None else np.full(24, np.nan))
            parts.append(rec["onshore"]  if rec is not None else np.full(24, np.nan))
            parts.append(rec["offshore"] if rec is not None else np.full(24, np.nan))
            parts.append(rec["solar"]    if rec is not None else np.full(24, np.nan))

    parts.append(dow_vec)

    # Cross-zone prices at lags d-1, d-7
    if config.use_cross_zone_prices:
        for lag in [1, 7]:
            lag_ds = (d - pd.Timedelta(days=lag)).strftime("%Y-%m-%d")
            rec = df_dict.get(lag_ds)
            for z in CROSS_ZONES:
                parts.append(rec["prices_xz"][z] if rec is not None
                              else np.full(24, np.nan))

    # Cross-zone ren at lags d, d-1, d-7
    if config.use_cross_zone_ren:
        for lag in [0, 1, 7]:
            lag_ds = (d - pd.Timedelta(days=lag)).strftime("%Y-%m-%d")
            rec = df_dict.get(lag_ds)
            for z in CROSS_ZONES:
                parts.append(rec["ren_xz"][z] if rec is not None
                              else np.full(24, np.nan))

    # DK1 channel flows at lags d-1, d-7
    if config.use_dk1_transmission:
        for lag in [1, 7]:
            lag_ds = (d - pd.Timedelta(days=lag)).strftime("%Y-%m-%d")
            rec = df_dict.get(lag_ds)
            for ch in DK1_CHANNELS:
                parts.append(rec["flow_dk1_ch"][ch] if rec is not None else np.zeros(24))

    # Cross-zone channel flows at lags d-1, d-7
    if config.use_cross_zone_transmission:
        for lag in [1, 7]:
            lag_ds = (d - pd.Timedelta(days=lag)).strftime("%Y-%m-%d")
            rec = df_dict.get(lag_ds)
            for z, channels in CROSS_ZONE_FLOWS.items():
                for ch in channels:
                    parts.append(rec["flow_xz"][(z, ch)] if rec is not None else np.zeros(24))

    # Energy scalars at lags d-1, d-7
    if config.use_energy:
        for lag in [1, 7]:
            lag_ds = (d - pd.Timedelta(days=lag)).strftime("%Y-%m-%d")
            rec = df_dict.get(lag_ds)
            parts.append(rec["energy"] if rec is not None else np.zeros(4))

    # Macro daily at lags d-1, d-7
    if config.use_macro_daily:
        for lag in [1, 7]:
            lag_ds = (d - pd.Timedelta(days=lag)).strftime("%Y-%m-%d")
            rec = df_dict.get(lag_ds)
            parts.append(rec["macro_daily"] if rec is not None else np.zeros(4))

    # Holiday dummy
    if config.use_holidays:
        rec = df_dict.get(date_str)
        parts.append(np.array([rec["is_holiday"] if rec is not None else 0.0]))

    # Interaction terms (121 entries)
    if config.use_interactions:
        rec_d  = df_dict.get(date_str)
        rec_d1 = df_dict.get((d - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))

        # Net load = load forecast minus renewable forecasts, for day d
        if rec_d is not None:
            nl_dk1 = rec_d["load"] - rec_d["onshore"] - rec_d["offshore"] - rec_d["solar"]
            nl_de  = rec_d["load_de"]  - rec_d["wind_de"]  - rec_d["solar_de"]
            nl_dk2 = rec_d["load_dk2"] - rec_d["wind_dk2"] - rec_d["solar_dk2"]
        else:
            nl_dk1 = nl_de = nl_dk2 = np.zeros(24)

        parts.append(nl_dk1)           # NetLoad_DK1              (24)
        parts.append(nl_dk1 ** 2)      # NetLoad_DK1 squared      (24)
        parts.append(nl_dk1 * nl_de)   # NetLoad_DK1 x NetLoad_DE (24)
        parts.append(nl_dk1 * nl_dk2)  # NetLoad_DK1 x NetLoad_DK2(24)

        # Hydro: NO2 rainfall over the last 14 days x DK1-NO2 flow, at d-1
        if rec_d1 is not None:
            precip = rec_d1["precip_no2_r14"]
            flow   = rec_d1["flow_dk1_ch"]["no2"]
            parts.append(precip * flow)
        else:
            parts.append(np.zeros(24))  # Precip x Flow            (24)

        # Squared distance of d-1 temperature from 18 C (heating/cooling demand)
        temp = rec_d1["temp_dk1"] if rec_d1 is not None else 180.0
        parts.append(np.array([(temp - 180.0) ** 2]))  # (1)

    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Variance-stabilising transformation (eq. 18)
# ---------------------------------------------------------------------------
def asinh_transform(X_train: np.ndarray,
                    x_pred: np.ndarray) -> tuple:
    """asinh((x - median) / MAD) column by column, with median and MAD taken
    from the training sample and reused for x_pred."""
    medians = np.nanmedian(X_train, axis=0)
    mads    = np.nanmedian(np.abs(X_train - medians), axis=0)
    mads[mads < 1e-10] = 1.0
    return (np.arcsinh((X_train - medians) / mads),
            np.arcsinh((x_pred   - medians) / mads))


# ---------------------------------------------------------------------------
# LASSO estimation for one hour
# ---------------------------------------------------------------------------
def fit_hour(X_train_t: np.ndarray, y_train: np.ndarray,
             x_pred_t: np.ndarray) -> tuple:
    """Forecast one hourly price with LASSO; same procedure as Case A.

    lambda is chosen by AIC along the LARS path, the model is refitted at that
    lambda and the prediction transformed back to EUR/MWh.
    Returns (forecast, coefficients, lambda).
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
    return float(np.sinh(y_hat_t) * y_mad + y_med), lasso.coef_.copy(), alpha_opt


# ---------------------------------------------------------------------------
# Single-window forecast for one day
# ---------------------------------------------------------------------------
def forecast_one_window(forecast_date: pd.Timestamp,
                        win_days: int,
                        train_dates: List[str],
                        df_dict: dict,
                        forecasted_prices: dict,
                        dow_vec: np.ndarray,
                        x_pred: np.ndarray,
                        config: ModelConfig) -> np.ndarray:
    """Fit the 24 hourly models on train_dates (already cut to the window by
    the caller) and forecast forecast_date.

    Returns (forecasts, coefficients, lambdas); forecasts are NaN for hours
    with fewer than 20 usable training days.
    """
    X_rows, y_rows = [], {h: [] for h in range(24)}
    for td in train_dates:
        # The first 7 days have no lag-7 prices
        if (pd.Timestamp(td) - REF_DATE).days < 7:
            continue
        xr = build_feature_row(td, df_dict, {}, df_dict[td]["dow"], config)
        if not np.isfinite(xr).all():
            continue
        X_rows.append(xr)
        for h in range(24):
            y_rows[h].append(df_dict[td]["prices"][h])

    n_feat = x_pred.shape[0]
    preds  = np.full(24, np.nan)
    coefs  = np.zeros((24, n_feat))
    alphas = np.full(24, np.nan)

    if len(X_rows) < 20 or not np.isfinite(x_pred).all():
        return preds, coefs, alphas

    X_train = np.array(X_rows, dtype=float)
    X_train_t, x_pred_t = asinh_transform(X_train, x_pred)

    for h in range(24):
        y_train = np.array(y_rows[h], dtype=float)
        valid   = np.isfinite(y_train)
        if valid.sum() < 20:
            continue
        try:
            preds[h], coefs[h], alphas[h] = fit_hour(
                X_train_t[valid], y_train[valid], x_pred_t)
        except Exception:
            pass

    return preds, coefs, alphas


# ---------------------------------------------------------------------------
# Forecast for one day
# ---------------------------------------------------------------------------
def forecast_one_day(forecast_date: pd.Timestamp,
                     config: ModelConfig,
                     train_dates: List[str],
                     df_dict: dict,
                     forecasted_prices: dict) -> Dict[str, np.ndarray]:
    """Forecast one day.

    Returns {"2y": forecasts, "ensemble": the same forecasts, "coefs": ...,
    "alphas": ...}; "ensemble" is kept so the output matches Case A.
    """
    dow_vec = np.zeros(7)
    dow_vec[forecast_date.dayofweek] = 1
    ds = forecast_date.strftime("%Y-%m-%d")

    x_pred = build_feature_row(ds, df_dict, forecasted_prices, dow_vec, config)

    preds, coefs, alphas = forecast_one_window(
        forecast_date, config.window_days, train_dates,
        df_dict, forecasted_prices, dow_vec, x_pred, config,
    )

    return {"2y": preds, "ensemble": preds, "coefs": coefs, "alphas": alphas}
