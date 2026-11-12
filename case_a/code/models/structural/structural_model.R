# -----------------------------------------------------------------------------
# Case A: structural (fundamental) price model and 2024 scenario forecasts
# (Section 3.1).
#
# 1. Seasonal regressions for load, wind, solar and net imports, used to show
#    trend and seasonality (overview figure).
# 2. Price model: hourly price on day-ahead load, wind and solar forecasts and
#    the previous day's coal, gas, oil and EUA prices, all interacted with the
#    hour (estimated on 2018-2023).
# 3. 2024 forecasts under six fuel-price scenarios (2023 path, +-20%, 2022
#    path, 2018-2021 average, December 2023 level), saved for the comparison
#    plots in plots/weekly and plots/lear.
# 4. RMSE tables (LaTeX) for the first week of 2024 and by quarter, and a
#    table comparing these with the LEAR / LightGBM results.
#
# Input:  data/prepared/dataset.csv  (from data_prep/build_dataset.py)
# Output: results/struct/data_pred_weekly.csv, data_pred_first_week.csv,
#         errors_first_week.tex, errors_weekly.tex, rmse_comparison.tex
#         figures/model_selection_struct/combined_overview.png
#
# Paths are built from this script's location (root = the case_a folder), so it
# runs from anywhere (RStudio or Rscript). Needs: tidyverse, zoo, patchwork,
# xtable.
# -----------------------------------------------------------------------------

library(tidyverse)

# root = the case_a folder (four levels above this script).
# Found automatically in RStudio and with "Rscript <this file>"; otherwise set
# it by hand, e.g. root <- "/path/to/case_a"
up4 <- function(p) dirname(dirname(dirname(dirname(normalizePath(p)))))
find_root <- function() {
  if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
    p <- rstudioapi::getActiveDocumentContext()$path
    if (nzchar(p)) return(up4(p))
  }
  f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  if (length(f) == 1) return(up4(f))
  stop("Could not find the script location; set root by hand (see above).")
}
if (!exists("root")) root <- find_root()

dir.create(file.path(root, "results/struct"),
           recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(root, "figures/model_selection_struct"),
           recursive = TRUE, showWarnings = FALSE)

dataset <- read_csv(file.path(root, "data/prepared/dataset.csv"))

dataset <- dataset %>%
  # Net imports into DK1 = sum of inflows - sum of outflows (missing = 0)
  mutate(net_inflow_to_DK1 = 
           # Inflows to DK1
           replace_na(`flow_DE-AT-LU_to_DK1`, 0) +
           replace_na(`flow_DE-LU_to_DK1`, 0) +
           replace_na(`flow_DK2_to_DK1`, 0) +
           replace_na(`flow_GB_to_DK1`, 0) +
           replace_na(`flow_NL_to_DK1`, 0) +
           replace_na(`flow_NO2_to_DK1`, 0) +
           replace_na(`flow_SE3_to_DK1`, 0) -
           # Outflows from DK1
           replace_na(`flow_DK1_to_DE-AT-LU`, 0) -
           replace_na(`flow_DK1_to_DE-LU`, 0) -
           replace_na(`flow_DK1_to_DK2`, 0) -
           replace_na(`flow_DK1_to_GB`, 0) -
           replace_na(`flow_DK1_to_NL`, 0) -
           replace_na(`flow_DK1_to_NO2`, 0) -
           replace_na(`flow_DK1_to_SE3`, 0)
  )

# Seasonal models for load, wind, solar and net imports (Section 3.1) ----

# Actual load (in logs), renewables and fuel prices
data_base <- dataset %>% select(time, price, `Actual Total Load (MW)`,
                                      `Renewables_Actual_ONSHORE`,
                                      `Renewables_Actual_SOLAR`,
                                      `Renewables_Actual_OFFSHORE`,
                                      net_inflow_to_DK1, Oil_Price,
                                      Coal_Price, Natural_Gas_Price, EUA_Price) %>%
  rename(load = `Actual Total Load (MW)`,
         onshore = `Renewables_Actual_ONSHORE`,
         solar = `Renewables_Actual_SOLAR`,
         offshore = `Renewables_Actual_OFFSHORE`) %>%
  mutate(log_solar = asinh(solar),
         total_wind = onshore + offshore,
         load = log(load))


# Calendar variables: hour, day of week, trend t (days since 2018-01-01),
# 8 annual harmonics in day of year d, a 4-year cycle in t, and holiday-period
# dummies (late July, Christmas)
data <- data_base %>% 
  mutate(hour = substr(time, start =12, stop = 13),
         hour = as.factor(ifelse(hour == "", "00", hour)),
         date = as.Date(substr(time, start = 1, stop = 10)),
         year  = year(date),
         month = month(date),
         day   = mday(date),
         dow = wday(date, label = TRUE, abbr = TRUE, week_start = 1),
         t    = as.integer(date - as.Date("2018-01-01")),
         d = yday(date),
         sin_season = sin(2 * pi * d / 365),
         cos_season = cos(2 * pi * d / 365),
         sin_season2 = sin(2 * pi * d / 365 * 2),
         cos_season2 = cos(2 * pi * d / 365 *2),
         sin_season3 = sin(2 * pi * d / 365 * 3),
         cos_season3 = cos(2 * pi * d / 365 * 3),
         sin_season4 = sin(2 * pi * d / 365 * 4),
         cos_season4 = cos(2 * pi * d / 365 * 4),
         sin_season5 = sin(2 * pi * d / 365 * 5),
         cos_season5 = cos(2 * pi * d / 365 * 5),
         sin_season6 = sin(2 * pi * d / 365 * 6),
         cos_season6 = cos(2 * pi * d / 365 * 6),
         sin_season7 = sin(2 * pi * d / 365 * 7),
         cos_season7 = cos(2 * pi * d / 365 * 7),
         sin_season8 = sin(2 * pi * d / 365 * 8),
         cos_season8 = cos(2 * pi * d / 365 * 8),
         sin_business = sin(2 * pi * t / 365 / 4),
         cos_business = cos(2 * pi * t / 365 / 4),
         summer_break = as.integer(month == 7  & day >= 18),
         winter_break = as.integer(month == 12 & day >= 25))  %>%
  filter(year < 2024)

# Load predictions ----
model_load <- lm(load ~ hour + hour * t + hour * sin_season + 
                   hour * cos_season + dow * hour + dow +
                   hour * sin_season2 + hour * cos_season2 +
                   hour * sin_season3 + hour * cos_season3 +
                   hour * summer_break + hour * winter_break +
                   hour * sin_business + hour * cos_business, 
                 data = data %>% filter(is.finite(load)))
data$load_pred <- predict(model_load, newdata = data)

# Wind predictions ----
model_total_wind <- lm(total_wind ~ hour + hour * t + hour * sin_season + 
                         hour * cos_season + hour * sin_season2 + hour * cos_season2 +
                         hour * sin_season3 + hour * cos_season3 +
                         hour * sin_season4 + hour * cos_season4 +
                         hour * sin_season5 + hour * cos_season5 + 
                         hour * sin_season6 + hour * cos_season6, 
                       data = data)
data$total_wind_pred <- predict(model_total_wind, newdata = data)

# Solar predictions ----
model_solar <- lm(log_solar ~ hour + hour * t + hour * I(t^2) + 
                    hour * sin_season + hour * sin_season * t +
                    hour * cos_season + hour * cos_season * t + 
                    hour * I(t^2) * sin_season + hour * I(t^2) * cos_season +
                    hour * sin_season2 + hour * cos_season2 +
                    hour * sin_season3 + hour * cos_season3, 
                  data = data)
data$solar_pred <- predict(model_solar, newdata = data)


# Net import predictions ----
model_transmission <- lm(net_inflow_to_DK1 ~ hour + hour * t + hour * I(t^2) + 
                    hour * sin_season + hour * sin_season * t +
                    hour * cos_season + hour * cos_season * t + 
                    hour * I(t^2) * sin_season + hour * I(t^2) * cos_season +
                    hour * sin_season2 + hour * cos_season2 +
                    hour * sin_season3 + hour * cos_season3, 
                  data = data)
data$net_inflow_to_DK1_pred <- predict(model_transmission, newdata = data)


# Long-term trend data (aggregated by year-month) ----
make_time_data <- function(df, actual_col, pred_col) {
  df %>%
    mutate(hour = as.integer(as.character(hour))) %>%
    group_by(year, month) %>%
    summarize(actual = mean(.data[[actual_col]], na.rm = TRUE),
              predicted = mean(.data[[pred_col]], na.rm = TRUE),
              .groups = "drop") %>%
    mutate(date_plot = as.Date(paste(year, month, "01", sep = "-")))
}

# Seasonality data (aggregated by month-day, averaged over all years) ----
make_seasonality_data <- function(df, actual_col, pred_col) {
  df %>%
    mutate(hour = as.integer(as.character(hour))) %>%
    group_by(month, day) %>%
    summarize(actual = mean(.data[[actual_col]], na.rm = TRUE),
              predicted = mean(.data[[pred_col]], na.rm = TRUE),
              .groups = "drop") %>%
    mutate(date_plot = as.Date(paste(2018, month, day, sep = "-"), format = "%Y-%m-%d"))
}

time_load  <- make_time_data(data, "load", "load_pred")

time_wind  <- make_time_data(data, "total_wind", "total_wind_pred")
time_solar <- make_time_data(data, "solar", "solar_pred")

seas_load  <- make_seasonality_data(data, "load", "load_pred")
seas_wind  <- make_seasonality_data(data, "total_wind", "total_wind_pred")
seas_solar <- make_seasonality_data(data, "solar", "solar_pred")

time_transmis  <- make_time_data(data, "net_inflow_to_DK1", "net_inflow_to_DK1_pred")
seas_transmis  <- make_seasonality_data(data, "net_inflow_to_DK1", "net_inflow_to_DK1_pred")

# Toggle predictions on/off ----
show_predictions <- FALSE   # TRUE adds the fitted values to the plots


# Plot helpers ----
plot_time <- function(df, ylabel, show_pred = show_predictions) {
  p <- ggplot(df, aes(x = date_plot)) +
    geom_line(aes(y = actual), colour = "#2166ac", linewidth = 0.5, alpha = 0.6) +
    #geom_smooth(aes(y = actual), colour = "#d6604d", linewidth = 1,
    #            se = FALSE, method = "loess", span = 0.1) +
    scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
    labs(x = NULL, y = ylabel) +
    theme_minimal(base_size = 11) +
    theme(panel.grid.minor = element_blank(),
          axis.text.x = element_text(angle = 45, hjust = 1))
  
  if (show_pred) {
    p <- p + geom_line(aes(y = predicted), colour = "#33a02c", linewidth = 0.5, alpha = 0.8)
  }
  p
}

plot_seasonality <- function(df, ylabel, show_pred = show_predictions) {
  p <- ggplot(df, aes(x = date_plot)) +
    geom_line(aes(y = actual), colour = "#2166ac", linewidth = 0.5, alpha = 0.6) +
    #geom_smooth(aes(y = actual), colour = "#d6604d", linewidth = 1,
    #            se = FALSE, method = "loess", span = 0.1) +
    scale_x_date(date_breaks = "1 month", date_labels = "%b") +
    labs(x = NULL, y = ylabel) +
    theme_minimal(base_size = 11) +
    theme(panel.grid.minor = element_blank(),
          axis.text.x = element_text(angle = 45, hjust = 1))
  
  if (show_pred) {
    p <- p + geom_line(aes(y = predicted), colour = "#33a02c", linewidth = 0.5, alpha = 0.8)
  }
  p
}


# Build 6 plots ----
# Note: load is in logs at this point, although the axis says MW
p1 <- plot_time(time_load,   "Forecasted Load (MW)")  + ggtitle("Long-run trend")
p2 <- plot_seasonality(seas_load,   "Forecasted Load (MW)")  + ggtitle("Seasonality")
p3 <- plot_time(time_wind,   "Forecasted Wind (MW)")
p4 <- plot_seasonality(seas_wind,   "Forecasted Wind (MW)")
p5 <- plot_time(time_solar,  "Forecasted Solar (MW)")
p6 <- plot_seasonality(seas_solar,  "Forecasted Solar (MW)")

p7 <- plot_time(time_transmis,   "Forecasted Transmissions (MW)")  + ggtitle("Long-run trend")
p8 <- plot_seasonality(seas_transmis,  "Forecasted Transmissions (MW)") + ggtitle("Seasonality")

# Combine ----
library(patchwork)
combined <- ((p1 | p2) /
               (p3 | p4) /
               (p5 | p6)) +
  plot_annotation(
    # title    = "Load, Wind and Solar: Long-run Trend and Seasonality",
    theme    = theme(plot.title    = element_text(face = "bold", size = 14),
                     plot.subtitle = element_text(size = 11))
  )
combined

ggsave(filename = file.path(root, "figures/model_selection_struct/combined_overview.png"),
       plot = combined, width = 14, height = 12, dpi = 300)

# Price model: data for 2018-2023 ----
# Day-ahead forecasts of load and renewables, plus the previous day's fuel
# and EUA prices and mean electricity price

data_base_price <- dataset %>% 
  select(time, price, `Day-ahead Total Load Forecast (MW)`,
         Renewables_Dayahead_ONSHORE,
         Renewables_Dayahead_OFFSHORE,
         Renewables_Dayahead_SOLAR,
         net_inflow_to_DK1,
         Oil_Price, Coal_Price, Natural_Gas_Price, EUA_Price) %>%
  rename(load     = `Day-ahead Total Load Forecast (MW)`,
         onshore  = Renewables_Dayahead_ONSHORE,
         solar    = Renewables_Dayahead_SOLAR,
         offshore = Renewables_Dayahead_OFFSHORE) %>%
  mutate(log_solar  = asinh(solar),
         total_wind = onshore + offshore,
         load       = log(load),
         hour = as.factor(ifelse(substr(time, 12, 13) == "", "00", substr(time, 12, 13))),
         date  = as.Date(substr(time, 1, 10)),
         year  = year(date),
         month = month(date),
         day   = mday(date),
         dow   = wday(date, label = TRUE, abbr = TRUE, week_start = 1),
         t     = as.integer(date - as.Date("2018-01-01")),
         d     = yday(date)) %>%
  left_join(
    dataset %>%
      mutate(date = as.Date(substr(time, 1, 10))) %>%
      group_by(date) %>%
      slice(1) %>%                          # one row per day
      ungroup() %>%
      arrange(date) %>%
      mutate(Coal_Price_lag1        = lag(Coal_Price, 1),
             Natural_Gas_Price_lag1 = lag(Natural_Gas_Price, 1),
             Oil_Price_lag1         = lag(Oil_Price, 1),
             EUA_Price_lag1         = lag(EUA_Price, 1)) %>%
      select(date, Coal_Price_lag1, Natural_Gas_Price_lag1, 
             Oil_Price_lag1, EUA_Price_lag1),
    by = "date"
  ) %>%
  left_join(
    dataset %>%
      mutate(date = as.Date(substr(time, 1, 10))) %>%
      group_by(date) %>%
      summarize(price_lag1 = mean(price, na.rm = TRUE), .groups = "drop") %>%
      mutate(price_lag1 = lag(price_lag1, 1)),
    by = "date"
  ) %>%
  mutate(sin_season = sin(2 * pi * d / 365),
cos_season = cos(2 * pi * d / 365),
sin_season2 = sin(2 * pi * d / 365 * 2),
cos_season2 = cos(2 * pi * d / 365 *2),
sin_season3 = sin(2 * pi * d / 365 * 3),
cos_season3 = cos(2 * pi * d / 365 * 3),
sin_season4 = sin(2 * pi * d / 365 * 4),
cos_season4 = cos(2 * pi * d / 365 * 4),
sin_season5 = sin(2 * pi * d / 365 * 5),
cos_season5 = cos(2 * pi * d / 365 * 5),
sin_season6 = sin(2 * pi * d / 365 * 6),
cos_season6 = cos(2 * pi * d / 365 * 6),
sin_season7 = sin(2 * pi * d / 365 * 7),
cos_season7 = cos(2 * pi * d / 365 * 7),
sin_season8 = sin(2 * pi * d / 365 * 8),
cos_season8 = cos(2 * pi * d / 365 * 8),
sin_business = sin(2 * pi * t / 365 / 4),
cos_business = cos(2 * pi * t / 365 / 4),
summer_break = as.integer(month == 7  & day >= 18),
winter_break = as.integer(month == 12 & day >= 25))  %>%
  filter(year < 2024)


# Price model ----
model_price <- lm(price ~ hour + 
                    hour * load + 
                    hour * total_wind + 
                    hour * solar + 
                    hour * Natural_Gas_Price_lag1 +
                    hour * Oil_Price_lag1 + 
                    hour * Coal_Price_lag1 + 
                    hour * EUA_Price_lag1,
                  data = data_base_price)


# 2024 data, built the same way. Note: this uses the realised 2024 day-ahead
# load and renewable forecasts; only the fuel prices are replaced by scenarios.
data_base_price2024 <- dataset %>% 
  select(time, price, `Day-ahead Total Load Forecast (MW)`,
         Renewables_Dayahead_ONSHORE,
         Renewables_Dayahead_OFFSHORE,
         Renewables_Dayahead_SOLAR,
         net_inflow_to_DK1,
         Oil_Price, Coal_Price, Natural_Gas_Price, EUA_Price) %>%
  rename(load     = `Day-ahead Total Load Forecast (MW)`,
         onshore  = Renewables_Dayahead_ONSHORE,
         solar    = Renewables_Dayahead_SOLAR,
         offshore = Renewables_Dayahead_OFFSHORE) %>%
  mutate(log_solar  = asinh(solar),
         total_wind = onshore + offshore,
         load       = log(load),
         hour = as.factor(ifelse(substr(time, 12, 13) == "", "00", substr(time, 12, 13))),
         date  = as.Date(substr(time, 1, 10)),
         year  = year(date),
         month = month(date),
         day   = mday(date),
         dow   = wday(date, label = TRUE, abbr = TRUE, week_start = 1),
         t     = as.integer(date - as.Date("2018-01-01")),
         d     = yday(date)) %>%
  left_join(
    dataset %>%
      mutate(date = as.Date(substr(time, 1, 10))) %>%
      group_by(date) %>%
      slice(1) %>%                          # one row per day
      ungroup() %>%
      arrange(date) %>%
      mutate(Coal_Price_lag1        = lag(Coal_Price, 1),
             Natural_Gas_Price_lag1 = lag(Natural_Gas_Price, 1),
             Oil_Price_lag1         = lag(Oil_Price, 1),
             EUA_Price_lag1         = lag(EUA_Price, 1)) %>%
      select(date, Coal_Price_lag1, Natural_Gas_Price_lag1, 
             Oil_Price_lag1, EUA_Price_lag1),
    by = "date"
  ) %>%
  left_join(
    dataset %>%
      mutate(date = as.Date(substr(time, 1, 10))) %>%
      group_by(date) %>%
      summarize(price_lag1 = mean(price, na.rm = TRUE), .groups = "drop") %>%
      mutate(price_lag1 = lag(price_lag1, 1)),
    by = "date"
  ) %>%
  mutate(sin_season = sin(2 * pi * d / 365),
         cos_season = cos(2 * pi * d / 365),
         sin_season2 = sin(2 * pi * d / 365 * 2),
         cos_season2 = cos(2 * pi * d / 365 *2),
         sin_season3 = sin(2 * pi * d / 365 * 3),
         cos_season3 = cos(2 * pi * d / 365 * 3),
         sin_season4 = sin(2 * pi * d / 365 * 4),
         cos_season4 = cos(2 * pi * d / 365 * 4),
         sin_season5 = sin(2 * pi * d / 365 * 5),
         cos_season5 = cos(2 * pi * d / 365 * 5),
         sin_season6 = sin(2 * pi * d / 365 * 6),
         cos_season6 = cos(2 * pi * d / 365 * 6),
         sin_season7 = sin(2 * pi * d / 365 * 7),
         cos_season7 = cos(2 * pi * d / 365 * 7),
         sin_season8 = sin(2 * pi * d / 365 * 8),
         cos_season8 = cos(2 * pi * d / 365 * 8),
         sin_business = sin(2 * pi * t / 365 / 4),
         cos_business = cos(2 * pi * t / 365 / 4),
         summer_break = as.integer(month == 7  & day >= 18),
         winter_break = as.integer(month == 12 & day >= 25))  %>%
  filter(year == 2024)

# Fuel-price scenarios for 2024 ----
# 30-day trailing averages of the lagged fuel prices
library(zoo)
data_base_price <- data_base_price %>%
  mutate(
    Natural_Gas_Price_smooth_lag1 = rollmean(Natural_Gas_Price_lag1, k = 30, fill = NA, align = "right"),
    Oil_Price_smooth_lag1         = rollmean(Oil_Price_lag1,         k = 30, fill = NA, align = "right"),
    Coal_Price_smooth_lag1        = rollmean(Coal_Price_lag1,        k = 30, fill = NA, align = "right"),
    EUA_Price_smooth_lag1         = rollmean(EUA_Price_lag1,         k = 30, fill = NA, align = "right")
  )

# Smoothed prices by day of year, from earlier years
# 2023 values
price_2023 <- data_base_price %>%
  filter(year == 2023) %>%
  group_by(d) %>%
  slice(1) %>%
  ungroup() %>%
  select(d,
         Natural_Gas_Price_smooth_lag1,
         Oil_Price_smooth_lag1,
         Coal_Price_smooth_lag1,
         EUA_Price_smooth_lag1) %>%
  rename_with(~ paste0(.x, "_2023"), -d)

# 2022 values
price_2022 <- data_base_price %>%
  filter(year == 2022) %>%
  group_by(d) %>%
  slice(1) %>%
  ungroup() %>%
  select(d,
         Natural_Gas_Price_smooth_lag1,
         Oil_Price_smooth_lag1,
         Coal_Price_smooth_lag1,
         EUA_Price_smooth_lag1) %>%
  rename_with(~ paste0(.x, "_2022"), -d)

# 2018-2021 average
price_avg <- data_base_price %>%
  filter(year %in% 2018:2021) %>%
  group_by(d) %>%
  summarize(
    Natural_Gas_Price_smooth_lag1_avg = mean(Natural_Gas_Price_smooth_lag1, na.rm = TRUE),
    Oil_Price_smooth_lag1_avg         = mean(Oil_Price_smooth_lag1,         na.rm = TRUE),
    Coal_Price_smooth_lag1_avg        = mean(Coal_Price_smooth_lag1,        na.rm = TRUE),
    EUA_Price_smooth_lag1_avg         = mean(EUA_Price_smooth_lag1,         na.rm = TRUE),
    .groups = "drop"
  )

# December 2023 average, held constant through 2024
price_dec2023 <- data_base_price %>%
  filter(year == 2023, month == 12) %>%
  summarise(
    Natural_Gas_Price_smooth_lag1_dec2023 = mean(Natural_Gas_Price_lag1, na.rm = TRUE),
    Oil_Price_smooth_lag1_dec2023         = mean(Oil_Price_lag1,         na.rm = TRUE),
    Coal_Price_smooth_lag1_dec2023        = mean(Coal_Price_lag1,        na.rm = TRUE),
    EUA_Price_smooth_lag1_dec2023         = mean(EUA_Price_lag1,         na.rm = TRUE)
  )

# Add all scenario paths to the 2024 data ----
data_base_price2024 <- data_base_price2024 %>%
  left_join(price_2023, by = "d") %>%
  left_join(price_2022, by = "d") %>%
  left_join(price_avg,  by = "d") %>%
  mutate(
    # 2023 scenarios
    Natural_Gas_Price_smooth_lag1_2023_low  = Natural_Gas_Price_smooth_lag1_2023 * 0.8,
    Natural_Gas_Price_smooth_lag1_2023_high = Natural_Gas_Price_smooth_lag1_2023 * 1.2,
    Oil_Price_smooth_lag1_2023_low          = Oil_Price_smooth_lag1_2023 * 0.8,
    Oil_Price_smooth_lag1_2023_high         = Oil_Price_smooth_lag1_2023 * 1.2,
    Coal_Price_smooth_lag1_2023_low         = Coal_Price_smooth_lag1_2023 * 0.8,
    Coal_Price_smooth_lag1_2023_high        = Coal_Price_smooth_lag1_2023 * 1.2,
    EUA_Price_smooth_lag1_2023_low          = EUA_Price_smooth_lag1_2023 * 0.8,
    EUA_Price_smooth_lag1_2023_high         = EUA_Price_smooth_lag1_2023 * 1.2,
    # december 2023 constant prices (same value for all days)
    Natural_Gas_Price_smooth_lag1_dec2023   = price_dec2023$Natural_Gas_Price_smooth_lag1_dec2023,
    Oil_Price_smooth_lag1_dec2023           = price_dec2023$Oil_Price_smooth_lag1_dec2023,
    Coal_Price_smooth_lag1_dec2023          = price_dec2023$Coal_Price_smooth_lag1_dec2023,
    EUA_Price_smooth_lag1_dec2023           = price_dec2023$EUA_Price_smooth_lag1_dec2023
  )

# Which column each scenario uses for each fuel price ----
scenarios <- list(
  baseline = list(
    Natural_Gas_Price_lag1 = "Natural_Gas_Price_smooth_lag1_2023",
    Oil_Price_lag1         = "Oil_Price_smooth_lag1_2023",
    Coal_Price_lag1        = "Coal_Price_smooth_lag1_2023",
    EUA_Price_lag1         = "EUA_Price_smooth_lag1_2023"
  ),
  good_case = list(
    Natural_Gas_Price_lag1 = "Natural_Gas_Price_smooth_lag1_2023_low",
    Oil_Price_lag1         = "Oil_Price_smooth_lag1_2023_low",
    Coal_Price_lag1        = "Coal_Price_smooth_lag1_2023_low",
    EUA_Price_lag1         = "EUA_Price_smooth_lag1_2023_low"
  ),
  bad_case = list(
    Natural_Gas_Price_lag1 = "Natural_Gas_Price_smooth_lag1_2023_high",
    Oil_Price_lag1         = "Oil_Price_smooth_lag1_2023_high",
    Coal_Price_lag1        = "Coal_Price_smooth_lag1_2023_high",
    EUA_Price_lag1         = "EUA_Price_smooth_lag1_2023_high"
  ),
  year_2022 = list(
    Natural_Gas_Price_lag1 = "Natural_Gas_Price_smooth_lag1_2022",
    Oil_Price_lag1         = "Oil_Price_smooth_lag1_2022",
    Coal_Price_lag1        = "Coal_Price_smooth_lag1_2022",
    EUA_Price_lag1         = "EUA_Price_smooth_lag1_2022"
  ),
  avg_2018_2021 = list(
    Natural_Gas_Price_lag1 = "Natural_Gas_Price_smooth_lag1_avg",
    Oil_Price_lag1         = "Oil_Price_smooth_lag1_avg",
    Coal_Price_lag1        = "Coal_Price_smooth_lag1_avg",
    EUA_Price_lag1         = "EUA_Price_smooth_lag1_avg"
  ),
  dec_2023 = list(
    Natural_Gas_Price_lag1 = "Natural_Gas_Price_smooth_lag1_dec2023",
    Oil_Price_lag1         = "Oil_Price_smooth_lag1_dec2023",
    Coal_Price_lag1        = "Coal_Price_smooth_lag1_dec2023",
    EUA_Price_lag1         = "EUA_Price_smooth_lag1_dec2023"
  )
)

# Predict 2024 prices under each scenario ----
for (scenario_name in names(scenarios)) {
  scenario <- scenarios[[scenario_name]]
  
  newdata <- data_base_price2024 %>%
    mutate(
      Natural_Gas_Price_lag1 = .data[[scenario$Natural_Gas_Price_lag1]],
      Oil_Price_lag1         = .data[[scenario$Oil_Price_lag1]],
      Coal_Price_lag1        = .data[[scenario$Coal_Price_lag1]],
      EUA_Price_lag1         = .data[[scenario$EUA_Price_lag1]]
    )
  
  data_base_price2024[[paste0("price_pred_", scenario_name)]] <- predict(model_price, newdata = newdata)
}

# Save weekly and first-week predictions ----
# Note the renaming: in the saved files "price_pred_baseline" is the
# December-2023 scenario and "price_pred_year_2023" is the 2023-path scenario
# (called "baseline" above).
data_pred_weekly <- data_base_price2024 %>%
  mutate(week = pmin(ceiling(d / 7), 52)) %>%
  group_by(week, hour) %>%
  summarise(
    price                                      = mean(price,                     na.rm = TRUE),
    price_pred_baseline   = mean(price_pred_dec_2023,       na.rm = TRUE),
    price_pred_year_2023              = mean(price_pred_baseline,       na.rm = TRUE),
    price_pred_avg_2018_2021  = mean(price_pred_avg_2018_2021,  na.rm = TRUE),
    price_pred_good_case             = mean(price_pred_good_case,      na.rm = TRUE),
    price_pred_bad_case            = mean(price_pred_bad_case,       na.rm = TRUE),
    price_pred_year_2022         = mean(price_pred_year_2022,      na.rm = TRUE),
    .groups = "drop"
  )

data_pred_first_week <- data_base_price2024 %>%
  filter(d %in% 1:7) %>%
  select(date, price, month, day, hour,
         price_pred_baseline  = price_pred_dec_2023,
         price_pred_year_2023             = price_pred_baseline,
         price_pred_avg_2018_2021  = price_pred_avg_2018_2021,
         price_pred_good_case            = price_pred_good_case,
         price_pred_bad_case            = price_pred_bad_case,
         price_pred_year_2022          = price_pred_year_2022)

write.csv(data_pred_weekly,
          file.path(root, "results/struct/data_pred_weekly.csv"),
          row.names = FALSE)
write.csv(data_pred_first_week,
          file.path(root, "results/struct/data_pred_first_week.csv"),
          row.names = FALSE)

# Colours and labels for the (currently commented-out) plots ----
scenario_colours <- c(
  "Actual price"            = "black",
  "Baseline (2023)"         = "#2166ac",
  "Good case (2023 ×0.8)"   = "#33a02c",
  "Bad case (2023 ×1.2)"    = "#d6604d",
  "2022 prices"             = "#984ea3",
  "Avg 2018-2021"           = "#ff7f00",
  "Dec 2023 prices"         = "#a6611a"
)

scenario_labels <- c(
  "price"                    = "Actual price",
  "price_pred_baseline"      = "Baseline (2023)",
  "price_pred_good_case"     = "Good case (2023 ×0.8)",
  "price_pred_bad_case"      = "Bad case (2023 ×1.2)",
  "price_pred_year_2022"     = "2022 prices",
  "price_pred_avg_2018_2021" = "Avg 2018-2021",
  "price_pred_dec_2023"      = "Dec 2023 prices"
)

scenario_linewidths <- c(
  "Actual price"            = 1.2,
  "Baseline (2023)"         = 0.7,
  "Good case (2023 ×0.8)"   = 0.7,
  "Bad case (2023 ×1.2)"    = 0.7,
  "2022 prices"             = 0.7,
  "Avg 2018-2021"           = 0.7,
  "Dec 2023 prices"         = 0.7
)
# 
# The two plots below are switched off; uncomment to redraw them.
# # Plot 1: weekly averages over the year ----
# plot_weekly <- data_pred_weekly %>%
#   filter(hour %in% c("00", "06", "12", "18")) %>%
#   pivot_longer(
#     cols      = c(price, starts_with("price_pred_")),
#     names_to  = "scenario",
#     values_to = "value"
#   ) %>%
#   mutate(scenario = recode(scenario, !!!scenario_labels)) %>%
#   ggplot(aes(x = week, y = value, colour = scenario, linewidth = scenario)) +
#   geom_line() +
#   facet_wrap(~ hour, ncol = 2, labeller = labeller(hour = ~ paste0("Hour ", .x, ":00"))) +
#   scale_colour_manual(values = scenario_colours) +
#   scale_linewidth_manual(values = scenario_linewidths) +
#   scale_x_continuous(breaks = seq(1, 52, by = 4)) +
#   guides(linewidth = "none") +
#   labs(
#     title    = "Predicted vs Actual Electricity Price by Scenario",
#     subtitle = "Weekly averages across selected hours",
#     x        = "Week of year",
#     y        = "Price (EUR/MWh)",
#     colour   = NULL
#   ) +
#   theme_minimal(base_size = 13) +
#   theme(
#     plot.title       = element_text(face = "bold"),
#     strip.text       = element_text(face = "bold"),
#     panel.grid.minor = element_blank(),
#     legend.position  = "bottom"
#   )
# 
# plot_weekly
# ggsave(file.path(root, "figures/pred_struct/plot_pred_weekly.png"),
#        plot = plot_weekly, width = 12, height = 10, dpi = 300)
# 
# # Plot 2: first week ----
# plot_first_week <- data_pred_first_week %>%
#   mutate(
#     hour     = as.integer(as.character(hour)),
#     date     = as.Date(date),
#     datetime = as.POSIXct(date) + hour * 3600
#   ) %>%
#   arrange(datetime) %>%
#   pivot_longer(
#     cols      = c(price, starts_with("price_pred_")),
#     names_to  = "scenario",
#     values_to = "value"
#   ) %>%
#   mutate(scenario = recode(scenario, !!!scenario_labels)) %>%
#   ggplot(aes(x = datetime, y = value, colour = scenario, linewidth = scenario)) +
#   geom_line() +
#   scale_colour_manual(values = scenario_colours) +
#   scale_linewidth_manual(values = scenario_linewidths) +
#   scale_x_datetime(date_breaks = "1 day", date_labels = "%a %d %b",
#                    date_minor_breaks = NULL) +
#   guides(linewidth = "none") +
#   labs(
#     title    = "Predicted vs Actual Electricity Price — First Week of 2024",
#     subtitle = "Hourly resolution",
#     x        = NULL,
#     y        = "Price (EUR/MWh)",
#     colour   = NULL
#   ) +
#   theme_minimal(base_size = 13) +
#   theme(
#     plot.title       = element_text(face = "bold"),
#     panel.grid.minor = element_blank(),
#     axis.text.x      = element_text(angle = 45, hjust = 1),
#     legend.position  = "bottom"
#   )
# 
# plot_first_week
# ggsave(file.path(root, "figures/pred_struct/plot_pred_first_week.png"),
#        plot = plot_first_week, width = 12, height = 6, dpi = 300)

# RMSE tables ----
library(xtable)

rmse <- function(actual, predicted) sqrt(mean((actual - predicted)^2, na.rm = TRUE))

scenario_names <- c(
  "price_pred_dec_2023"      = "1: Dec 2023 prices (baseline)",
  "price_pred_baseline"      = "2: Avg 2023 prices",
  "price_pred_avg_2018_2021" = "3: Pre-crisis avg (2018--2021)",
  "price_pred_good_case"     = "4: 2023 prices $\\times$0.8",
  "price_pred_bad_case"      = "5: 2023 prices $\\times$1.2",
  "price_pred_year_2022"     = "6: 2022 crisis prices"
)

pred_cols <- names(scenario_names)

# The columns of the saved data_pred_first_week / data_pred_weekly were renamed
# (there "price_pred_baseline" is the December-2023 scenario), so the tables
# are computed from separate frames that keep the original scenario names.
data_pred_first_week_eval <- data_base_price2024 %>%
  filter(d %in% 1:7) %>%
  select(date, price, month, day, hour, all_of(pred_cols))

data_pred_weekly_eval <- data_base_price2024 %>%
  mutate(week    = pmin(ceiling(d / 7), 52),
         quarter = case_when(week <= 13 ~ "Q1",
                             week <= 26 ~ "Q2",
                             week <= 39 ~ "Q3",
                             TRUE       ~ "Q4")) %>%
  group_by(quarter, week, hour) %>%
  summarise(across(c(price, all_of(pred_cols)), ~ mean(.x, na.rm = TRUE)),
            .groups = "drop")

# Table 1: RMSE by day, first week of 2024 ----
mean_price1 <- data_pred_first_week_eval %>%
  group_by(day) %>%
  summarise(val = mean(price, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(names_from = day, values_from = val, names_prefix = "Day ") %>%
  mutate(Scenario = "Mean price",
         RMSE     = mean(data_pred_first_week_eval$price, na.rm = TRUE)) %>%
  select(Scenario, everything())

table1 <- map_dfr(pred_cols, function(col) {
  by_day <- data_pred_first_week_eval %>%
    group_by(day) %>%
    summarise(val = rmse(price, .data[[col]]), .groups = "drop") %>%
    pivot_wider(names_from = day, values_from = val, names_prefix = "Day ")
  
  overall_rmse <- rmse(data_pred_first_week_eval$price, data_pred_first_week_eval[[col]])
  
  bind_cols(
    tibble(Scenario = scenario_names[[col]]),
    by_day,
    tibble(RMSE = overall_rmse)
  )
})

table1 <- bind_rows(table1, mean_price1)


# Table 2: RMSE by quarter, weekly predictions 2024 ----
mean_price2 <- data_pred_weekly_eval %>%
  group_by(quarter) %>%
  summarise(val = mean(price, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(names_from = quarter, values_from = val) %>%
  mutate(Scenario = "Mean price",
         RMSE     = mean(data_pred_weekly_eval$price, na.rm = TRUE)) %>%
  select(Scenario, Q1, Q2, Q3, Q4, RMSE)

table2 <- map_dfr(pred_cols, function(col) {
  by_quarter <- data_pred_weekly_eval %>%
    group_by(quarter) %>%
    summarise(val = rmse(price, .data[[col]]), .groups = "drop") %>%
    pivot_wider(names_from = quarter, values_from = val)
  
  overall_rmse <- rmse(data_pred_weekly_eval$price, data_pred_weekly_eval[[col]])
  
  bind_cols(
    tibble(Scenario = scenario_names[[col]]),
    by_quarter,
    tibble(RMSE = overall_rmse)
  ) %>%
    select(Scenario, Q1, Q2, Q3, Q4, RMSE)
})

table2 <- bind_rows(table2, mean_price2)


# Print and save tables 1 and 2 ----
print(
  xtable(table1,
         caption = "Forecast Errors by Day --- First Week of 2024",
         label   = "tab:errors_first_week",
         digits  = 2),
  include.rownames       = FALSE,
  booktabs               = TRUE,
  caption.placement      = "top",
  hline.after            = c(-1, 0, nrow(table1) - 1, nrow(table1)),
  sanitize.text.function = identity
)

print(
  xtable(table2,
         caption = "Forecast Errors by Quarter --- Weekly Predictions 2024",
         label   = "tab:errors_weekly",
         digits  = 2),
  include.rownames       = FALSE,
  booktabs               = TRUE,
  caption.placement      = "top",
  hline.after            = c(-1, 0, nrow(table2) - 1, nrow(table2)),
  sanitize.text.function = identity
)


sink(file.path(root, "results/struct/errors_first_week.tex"))
print(xtable(table1,
             digits  = 2),
      include.rownames       = FALSE,
      booktabs               = TRUE,
      floating               = FALSE,      # only writes \begin{tabular}
      hline.after            = c(-1, 0, nrow(table1) - 1, nrow(table1)),
      sanitize.text.function = identity)
sink()
sink(file.path(root, "results/struct/errors_weekly.tex"))
print(xtable(table2,
             digits  = 2),
      include.rownames       = FALSE,
      booktabs               = TRUE,
      floating               = FALSE,      # only writes \begin{tabular}
      hline.after            = c(-1, 0, nrow(table2) - 1, nrow(table2)),
      sanitize.text.function = identity)
sink()


# Table 3: structural model vs LEAR / LightGBM, first week of 2024 ----

# RMSEs of the other models, copied by hand from their output
external_models <- tibble(
  Scenario = c(
    "LEAR Baseline",
    "LEAR Structural + Holidays + Split Ren",
    "LEAR Structural + ... + Energy + Macro + Trans",
    "LightGBM (full)"
  ),
  RMSE = c(32.71, 20.48, 20.77, 30.99)
)

# RMSEs of the structural scenarios
structural_models <- map_dfr(pred_cols, function(col) {
  overall_rmse <- rmse(data_pred_first_week_eval$price, data_pred_first_week_eval[[col]])
  tibble(
    Scenario = scenario_names[[col]],
    RMSE     = overall_rmse
  )
})

# Stack the two blocks
table_combined <- bind_rows(
  external_models,
  tibble(Scenario = "---", RMSE = NA_real_),  # blank separator row
  structural_models
)

# Print and save ----
addtorow <- list(
  pos     = list(0, nrow(external_models) + 1),
  command = c(
    "\\multicolumn{2}{l}{\\textbf{Benchmark Models}} \\\\\n",
    "\\multicolumn{2}{l}{\\textbf{Fundamental Model}} \\\\\n"
  )
)

print(
  xtable(table_combined,
         caption = "Weekly RMSE Comparison Across Models",
         label   = "tab:rmse_comparison",
         digits  = 2),
  include.rownames       = FALSE,
  booktabs               = TRUE,
  floating               = FALSE,
  hline.after            = c(-1, 0, nrow(external_models),
                             nrow(external_models) + 1,
                             nrow(table_combined)),
  sanitize.text.function = identity,
  add.to.row             = addtorow
)

sink(file.path(root, "results/struct/rmse_comparison.tex"))
print(
  xtable(table_combined, digits = 2),
  include.rownames       = FALSE,
  booktabs               = TRUE,
  floating               = FALSE,
  hline.after            = c(-1, 0, nrow(external_models),
                             nrow(external_models) + 1,
                             nrow(table_combined)),
  sanitize.text.function = identity,
  add.to.row             = addtorow
)
sink()



