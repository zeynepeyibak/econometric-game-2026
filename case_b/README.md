# Case B: DK1 price forecasting with neighbouring markets

The task ([case_description.pdf](case_description.pdf)): produce
one-day-ahead forecasts of the 24 hourly day-ahead prices in DK1 for every
day of 2024, using information up to the day before (365 x 24 forecasts),
and repeat this for 2020. Then discuss whether the predictive role of
renewable production has changed between the two years, and whether there
are network effects: do prices, production and flows in the neighbouring
zones help to forecast the DK1 price, and has that changed over time?

We extend LEAR (Lago et al. 2021) with prices, renewable production and
cross-border flows of the neighbouring zones (DK2, NO2, SE4, DE) and with
interaction terms, and compare it with the baseline LEAR and with LightGBM.
Forecast accuracy is lower in 2024 than in 2020, while the renewable and
cross-zone variables gain predictive power over the period. Details are in
[report.pdf](report.pdf).

## Data

The data were provided by the case makers and are not included in this
repository (we are not allowed to share them). To run the code, put the
contents of the "Case B" data folder in `data/raw/`:

```
case_b/
  data/
    raw/                      the provided files (see below)
    additional/               holiday_indicator.csv (included)
    interim/                  created by data_prep/process_interim.py
    prepared/DK1/             created by data_prep/build_*.py
  code/
```

The provided data have the same structure as in Case A (see the
[Case A README](../case_a/README.md#data)), for five bidding zones and
2018-2024:

| Folder | Frequency | Content |
|---|---|---|
| `ElectricityPrices/<zone>/Prices` | hourly | day-ahead spot price (EUR/MWh) for DK1, DK2, NO2, SE4 and DE |
| `ElectricityPrices/<zone>/Loads` | hourly | actual total load and day-ahead load forecast (MW) |
| `ElectricityPrices/<zone>/Renewables` | hourly | onshore wind, offshore wind and solar: actual and day-ahead forecast (MW) |
| `ElectricityPrices/<zone>/Transmission` | hourly | physical flows between the zone and each of its neighbours (MW), by direction |
| `EnergySeries` | daily | coal, oil, natural gas and EUA prices |
| `MacroFinance/Daily` | daily | VIX, EURO STOXX 50, MSCI EU index, EUR/USD exchange rate |
| `MacroFinance/Monthly` | monthly | CPI and energy CPI for Denmark, Germany, Norway and Sweden; 1-month and 1-year EURIBOR |
| `MacroFinance/Quarterly` | quarterly | GDP of Denmark, Germany, Norway, Sweden and the euro area |
| `Climate Variables/<zone>` | daily | ECA&D station data per zone: mean temperature, precipitation, sunshine duration, mean wind speed, and cloud cover where available |

`data/additional/holiday_indicator.csv` is not part of the provided data and
is included: a dummy for Danish public holidays by month and day. Since it has
no year, the same dates count as holidays in every year.

Two gaps in the data matter for the models: German prices start on 1 October
2018, and the German day-ahead load forecast is missing for most of 2018 (see
the notes below).

## Code and run order

Every script finds the `case_b` folder from its own location, so it can be
run from any directory.

| Step | Script | What it does | Output |
|---|---|---|---|
| 1 | `data_prep/process_interim.py` | stacks the yearly raw files, one file per source and zone | `data/interim/` |
| 2 | `data_prep/build_dataset.py` | one hourly dataset for DK1 with the series of all zones | `data/prepared/DK1/dataset.csv` |
| 2 | `data_prep/build_lear_dataset.py` | daily dataset, 24 columns per series | `data/prepared/DK1/lear_dataset.csv` |
| 3 | `models/lear/rolling.py` | LEAR re-estimated every day of the hold-out year (the main results) | `results/lear/` |
| 3 | `models/lear/static.py` | LEAR estimated once, for a quick comparison | `results/lear/` |
| 3 | `models/lgbm/predict.py` | LightGBM, re-estimated every day | `results/lgbm/` |
| 3 | `models/structural/structural_model.R` | the Case A structural model on the Case B data | `results/struct/`, `figures/` |
| 4 | `plots/descriptives/descriptives.R` | renewable production and cross-border flows by zone, 2018-2024 | `figures/descriptives/` |
| 4 | `plots/descriptives/cross_country_correlations.py` | price correlations between DK1 and the other zones | `figures/descriptives/` |
| 4 | `plots/models/figures_and_table1.py` | figures 1-4 and Table 1 of the report, forecast comparison plots | `results/figures/` |

The hold-out year (2020 or 2024) and the regressor blocks are set in
`models/lear/config.py`, which `rolling.py`, `static.py` and
`lgbm/predict.py` all read. `models/lear/core.py` holds the shared LEAR code.
`plots/models/figures_and_table1.py` reads the result files named at its
top; those names must match runs that exist in `results/`.

Run times: `static.py` under a minute; `rolling.py` and `lgbm/predict.py`
re-estimate the model for each of the 365/366 days and take hours for a full
year.

## Notes

- All series are aligned in UTC (see the Case A README). DK1 and DK2 prices
  come in UTC; everything else is converted from Danish local time.
- Days with missing German prices or load forecasts are left missing in
  `lear_dataset.csv`. LEAR skips training days with missing regressors, so
  2020 runs that use German data train on fewer days (about 420 of 723 in
  `static.py`). Isolated missing hours are interpolated within each
  variable's own 24 hours.
- In `lear_dataset.csv`, `flow_<zone>_<partner>` columns are net flows (out
  minus in, positive = export); `flow_dk1_in` and `flow_dk1_out` are gross
  totals over all borders.
- Price lags always use observed prices, since all past prices are known when
  a day-ahead forecast is made.
