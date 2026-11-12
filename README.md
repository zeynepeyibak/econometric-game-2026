# Econometric Game 2026: forecasting Danish electricity prices

Code and reports from our team's entry in the Econometric Game 2026. Both
cases are about forecasting the hourly day-ahead electricity price in the
western Danish bidding zone DK1. The cases were written by a team from Aarhus
University.

Team: Zeynep Eyibak, Daphne Rutnam, Julian Kindlimann, Leon Kriesmair

| | Task | Methods | Report |
|---|---|---|---|
| [Case A](case_a/) | Forecast hourly prices for the first week of 2024 and weekly average prices for every week of 2024, using data up to 31 Dec 2023 | LEAR, LightGBM, a structural (fundamental) model with fuel-price scenarios, weekly LASSO with Fourier terms | [case_a/report.pdf](case_a/report.pdf) |
| [Case B](case_b/) | One-day-ahead forecasts for every day of 2020 and 2024, using data from the neighbouring zones; how the role of renewables and cross-border flows changed | LEAR with cross-zone regressors, estimated once or re-estimated daily; LightGBM | [case_b/report.pdf](case_b/report.pdf) |

## Repository layout

```
case_a/
  README.md               task, data, how to run, notes
  case_description.pdf    the case as handed out
  report.pdf              our report
  code/
    data_prep/            raw data -> model input
    models/               one folder per model
    plots/                descriptive and model-comparison figures
case_b/
  (same layout)
requirements.txt
```

Each case README has the run order and a description of the data.

## Data

The data were provided by the case makers for the competition and we are not
allowed to redistribute them, so they are not in this repository. The case
READMEs describe what each dataset contains and where the scripts expect it
(`case_a/data/raw/` and `case_b/data/raw/`). Everything else under `data/`,
and the `results/`, `figures/` and `descriptives/` folders, is created by the
scripts and is ignored by git.

All paths are worked out from each script's own location, so the scripts run
from any working directory and nothing needs to be edited.

## Requirements

Python 3.10 or later with the packages in `requirements.txt`, and R 4.x with
tidyverse, zoo, patchwork and xtable.

```
pip install -r requirements.txt
```
