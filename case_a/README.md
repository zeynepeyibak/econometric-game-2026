# Case A: forecasting DK1 prices for 2024

The task ([case_description.pdf](case_description.pdf)): using only data up to
31 December 2023,

1. forecast the 24 hourly day-ahead prices for each day of the first week of
   2024 (7 x 24 forecasts), and
2. forecast the weekly average of each hourly price for every week of 2024
   (52 x 24 forecasts),

and evaluate the forecasts against the realised 2024 prices.

We compare a structural model, in which load and renewable production are
forecast separately and then fed into a price equation, with LEAR (Lago et
al. 2021), which we extend from day-ahead to longer horizons, and with
LightGBM. The structural model turns out to be very sensitive to the
assumed fuel prices, and LEAR forecasts better at both horizons. Details are
in [report.pdf](report.pdf).

## Data

The data were provided by the case makers and are not included in this
repository (we are not allowed to share them). To run the code, put the
contents of the "Case A" data folder in `data/raw/`:

```
case_a/
  data/
    raw/                      the provided files (see below)
    interim/                  created by data_prep/process_interim.py
    prepared/                 created by data_prep/build_*.py
  code/
```

What the provided data contain (zone DK1; estimation sample 2018-2023, with
2024 used for evaluation):

| Folder | Frequency | Content |
|---|---|---|
| `ElectricityPrices/DK1/Prices` | hourly | day-ahead spot price (EUR/MWh), one file per year, timestamps in UTC |
| `ElectricityPrices/DK1/Loads` | hourly | actual total load and the day-ahead load forecast (MW), Danish local time |
| `ElectricityPrices/DK1/Renewables` | hourly | onshore wind, offshore wind and solar: actual production and day-ahead forecast (MW), plus intraday estimates that are not known a day ahead and are not used |
| `ElectricityPrices/DK1/Transmission` | hourly | physical flows between DK1 and each neighbouring zone (MW), by direction |
| `EnergySeries` | daily | coal, oil, natural gas and EU carbon allowance (EUA) prices: close, open, high, low, volume, % change |
| `MacroFinance/Daily` | daily | VIX, EURO STOXX 50, MSCI EU index, EUR/USD exchange rate |
| `MacroFinance/Monthly` | monthly | Danish CPI and energy CPI |
| `MacroFinance/Quarterly` | quarterly | GDP of Denmark and of the euro area |
| `Climate Variables/DK1` | daily | station-level weather from ECA&D, assigned to DK1: mean temperature (tg), precipitation (rr), sunshine duration (ss), mean wind speed (fg) |

The electricity files have the layout of the ENTSO-E Transparency Platform
exports. A day-ahead forecast is stored in the row of the hour it forecasts,
so it can be used for that hour. Section 5 of the case description has the
full description of every file.

## Code and run order

Every script finds the `case_a` folder from its own location, so it can be
run from any directory.

| Step | Script | What it does | Output |
|---|---|---|---|
| 1 | `data_prep/process_interim.py` | stacks the yearly raw files, one file per source | `data/interim/` |
| 2 | `data_prep/build_dataset.py` | one hourly dataset with all series | `data/prepared/dataset.csv` |
| 2 | `data_prep/build_lear_dataset.py` | daily dataset, 24 columns per series (one per hour) | `data/prepared/lear_dataset.csv` |
| 3 | `models/lear/validate.py` | walk-forward validation of LEAR variants, 2022-2023 | `results/lear/` |
| 3 | `models/lear/predict.py` | LEAR forecast for the first week of 2024 | `results/lear/` |
| 3 | `models/lgbm/predict.py` | LightGBM forecast for the first week of 2024 | `results/lear/` |
| 3 | `models/weekly/predict.py` | weekly LASSO with Fourier terms, all weeks of 2024 | `results/weekly/` |
| 3 | `models/structural/structural_model.R` | structural model and fuel-price scenarios, both tasks | `results/struct/`, `figures/` |
| 4 | `plots/descriptives/*.py` | descriptive figures for prices, loads, fuels and weather | `figures/descriptives/` |
| 4 | `plots/lear/*.py` | first-week forecasts of all models vs. actual prices; validation results | `figures/` |
| 4 | `plots/weekly/compare_weekly_2024.py` | weekly forecasts vs. actual prices | `figures/weekly/` |

`models/lear/core.py` holds the LEAR code shared by the LEAR and LightGBM
scripts. The plots in step 4 read the output of step 3; the comparison plots
also read `results/struct/`, so run the R script first.

Run times on a laptop: `models/lear/predict.py` a few minutes;
`models/lear/validate.py` several hours (104 weekly windows);
`models/lgbm/predict.py` depends on the number of Optuna trials (40 by
default).

## Notes

- All series are aligned in UTC. Prices come in UTC; loads, renewables and
  flows are in Danish local time and are converted, using the "(CEST)" /
  "(CET)" label to tell apart the two 02:00 hours in October.
- The regressor blocks of a LEAR variant are switched on and off with
  `ModelConfig` at the top of `models/lear/validate.py` and
  `models/lear/predict.py`; the output file names are built from the active
  blocks. In `predict.py`, days 2-7 of the forecast week use `CONFIG_AR`,
  which is not part of the file name.
- `plots/lear/validation.py` plots the validation run whose file-name prefix
  is set in `LABEL`.
- When some hours of a day cannot be forecast, the observed prices fill those
  hours in the lags used for the rest of the week.
- `models/weekly/predict.py` uses the same-week prices of 2023 and 2022 as
  regressors in every year of the estimation sample, and fits the scaler on
  the full sample before the cross-validation.
- The structural 2024 scenarios use the realised 2024 day-ahead forecasts of
  load and renewables; only the fuel prices vary. In the saved CSVs,
  `price_pred_baseline` is the December-2023 scenario and
  `price_pred_year_2023` the 2023-path scenario.
