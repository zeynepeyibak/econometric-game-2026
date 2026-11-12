"""
Model settings for Case B, shared by rolling.py and static.py.

Change HOLD_OUT_YEAR to switch the forecast year, and the flags in CONFIG to
switch regressor blocks on and off. The numbers in brackets are how many
regressors each block adds.
"""

from core import ModelConfig

# ---------------------------------------------------------------------------
# Forecast year
# ---------------------------------------------------------------------------
HOLD_OUT_YEAR = 2020  # the case asks for 2020 and 2024

# ---------------------------------------------------------------------------
# Model specification
# ---------------------------------------------------------------------------
CONFIG = ModelConfig(
    use_exogenous=True,
    use_cross_zone_prices=True,  # DK2/NO2/SE4/DE prices at lags d-1, d-7      (+192)
    use_cross_zone_ren=True,  # DK2/NO2/SE4/DE ren at lags d, d-1, d-7      (+288)
    use_dk1_transmission=True,  # DK1-DE/DK2/NO2/SE3 flows at lags d-1, d-7   (+192)
    use_cross_zone_transmission=True,  # all non-DK1 zone flows at lags d-1, d-7      (+1296)
    use_holidays=True,  # Danish public holiday dummy                   (+1)
    use_energy=True,  # coal/EUA/gas/oil at lags d-1, d-7            (+8)
    use_macro_daily=True,  # EuroStoxx50/MSCI/USD-EUR/VIX at d-1, d-7    (+8)
    use_interactions=True,  # NetLoad nonlinear + cross-border + hydro + temp (+121)
    window_days=730,  # rolling window length (2 years)
    train_cutoff=f"{HOLD_OUT_YEAR - 1}-12-31",
    label="lear",
)
