# -----------------------------------------------------------------------------
# Case B descriptive figures for DK1 and its neighbours, 2018-2024, with 2020
# and 2024 shaded:
#   Fig 1  renewable production (wind, solar) by zone, 30-day moving average
#   Fig 2  average intraday profile of wind and solar by zone, 2020 vs 2024
#   Fig 3  DK1 net imports and total absolute flow, 30-day moving average
#   Fig 4  the same per border of DK1
#   Output: figures/descriptives/*.png
#
# Input:  data/prepared/DK1/dataset.csv and lear_dataset.csv
# Paths are built from this script's location (root = the case_b folder).
# Runs in RStudio or with Rscript. Needs: tidyverse, zoo, patchwork, grid.
# -----------------------------------------------------------------------------

library(tidyverse)

# root = the case_b folder (four levels above this script).
# Found automatically in RStudio and with "Rscript <this file>"; otherwise set
# it by hand, e.g. root <- "/path/to/case_b"
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

dir.create(file.path(root, "figures/descriptives"), recursive = TRUE, showWarnings = FALSE)

dataset <- read_csv(file.path(root, "data/prepared/DK1/dataset.csv"))
dataset_total <- read_csv(file.path(root, "data/prepared/DK1/lear_dataset.csv"))

# PART 1: DESCRIPTIVES =========================================================

# Fig 1: renewable production over time ----
# Hourly data with actual renewable production and DK1 flows; adds DK1 net
# imports, DK1 total absolute flow, and total wind per zone

data_plot12 <- dataset %>% 
  mutate(hour = substr(time, start =12, stop = 13),
         hour = as.factor(ifelse(hour == "", "00", hour)),
         date = as.Date(substr(time, start = 1, stop = 10)),
         year  = year(date),
         month = month(date),
         day   = mday(date),
         dow = wday(date, label = TRUE, abbr = TRUE, week_start = 1),
         t    = as.integer(date - as.Date("2018-01-01")),
         d = yday(date)
  ) %>% select(year, month, day, hour, price, 
                                  # Actual renewable production variables
                                  matches("^ren_actual_"),
                                  matches("^flow_"),
               matches("^Renewables_Actual_")
                                    ) %>%
  mutate(net_inflow_to_DK1 = 
           # Inflows to DK1
           replace_na(`flow_dk1_DE-LU_to_DK1`, 0) +
           replace_na(`flow_dk1_DK2_to_DK1`,   0) +
           replace_na(`flow_dk1_GB_to_DK1`,     0) +
           replace_na(`flow_dk1_NL_to_DK1`,     0) +
           replace_na(`flow_dk1_NO2_to_DK1`,    0) +
           replace_na(`flow_dk1_SE3_to_DK1`,    0) -
           # Outflows from DK1
           replace_na(`flow_dk1_DK1_to_DE-LU`,  0) -
           replace_na(`flow_dk1_DK1_to_DK2`,    0) -
           replace_na(`flow_dk1_DK1_to_GB`,     0) -
           replace_na(`flow_dk1_DK1_to_NL`,     0) -
           replace_na(`flow_dk1_DK1_to_NO2`,    0) -
           replace_na(`flow_dk1_DK1_to_SE3`,    0),
         
         # Total flow over DK1's borders, both directions
         total_abs_flow_DK1 =
           # Inflows to DK1
           replace_na(`flow_dk1_DE-LU_to_DK1`, 0) +
           replace_na(`flow_dk1_DK2_to_DK1`,   0) +
           replace_na(`flow_dk1_GB_to_DK1`,     0) +
           replace_na(`flow_dk1_NL_to_DK1`,     0) +
           replace_na(`flow_dk1_NO2_to_DK1`,    0) +
           replace_na(`flow_dk1_SE3_to_DK1`,    0) +
           # Outflows from DK1 (added, not subtracted)
           replace_na(`flow_dk1_DK1_to_DE-LU`,  0) +
           replace_na(`flow_dk1_DK1_to_DK2`,    0) +
           replace_na(`flow_dk1_DK1_to_GB`,     0) +
           replace_na(`flow_dk1_DK1_to_NL`,     0) +
           replace_na(`flow_dk1_DK1_to_NO2`,    0) +
           replace_na(`flow_dk1_DK1_to_SE3`,    0))%>% 
  mutate(
  # DK1
  total_wind_dk1 = replace_na(Renewables_Actual_ONSHORE, 0) + 
    replace_na(Renewables_Actual_OFFSHORE, 0),
  
  # DK2
  total_wind_dk2 = replace_na(ren_actual_dk2_ONSHORE, 0) + 
    replace_na(ren_actual_dk2_OFFSHORE, 0),
  
  # NO2 and SE4 have no offshore wind
  total_wind_no2 = replace_na(ren_actual_no2_ONSHORE, 0),
  
  total_wind_se4 = replace_na(ren_actual_se4_ONSHORE, 0),
  
  # DE
  total_wind_de = if_else(
    is.na(ren_actual_de_ONSHORE) & is.na(ren_actual_de_OFFSHORE),
    NA_real_,
    replace_na(ren_actual_de_ONSHORE, 0) + replace_na(ren_actual_de_OFFSHORE, 0)
  )) %>%
  rename(ren_actual_dk1_SOLAR = Renewables_Actual_SOLAR) 

data_plot12_fin <- data_plot12 %>%
  select(
    # Time variables
    year, month, day, hour,
    
    # Price
    price,
    
    # Total wind per zone (computed above)
    total_wind_dk1,
    total_wind_dk2,
    total_wind_no2,
    total_wind_se4,
    total_wind_de,
    
    # Solar per zone (NO2 has none)
    ren_actual_dk1_SOLAR,
    ren_actual_dk2_SOLAR,
    ren_actual_se4_SOLAR,
    ren_actual_de_SOLAR,
    
    # DK1 flow variables
    net_inflow_to_DK1,
    total_abs_flow_DK1
  ) %>%
  rename(
    # Rename solar for consistency
    solar_dk1 = ren_actual_dk1_SOLAR,
    solar_dk2 = ren_actual_dk2_SOLAR,
    solar_se4 = ren_actual_se4_SOLAR,
    solar_de  = ren_actual_de_SOLAR
  )

library(tidyverse)
library(zoo)
library(patchwork)
library(tidyverse)
library(zoo)
library(patchwork)

# Daily means and 30-day trailing moving averages
data_plot1_daily <- data_plot12_fin %>%
  mutate(date = as.Date(paste(year, month, day, sep = "-"))) %>%
  group_by(date) %>%
  summarise(
    across(c(solar_dk1, solar_dk2, solar_se4, solar_de,
             total_wind_dk1, total_wind_dk2, total_wind_no2,
             total_wind_se4, total_wind_de),
           ~ mean(.x, na.rm = TRUE)),
    .groups = "drop"
  ) %>%
  arrange(date) %>%
  mutate(
    across(c(solar_dk1, solar_dk2, solar_se4, solar_de,
             total_wind_dk1, total_wind_dk2, total_wind_no2,
             total_wind_se4, total_wind_de),
           ~ rollmean(.x, k = 30, fill = NA, align = "right"),
           .names = "{.col}_ma30")
  )

# Shading
shading <- tibble(
  xmin = as.Date(c("2020-01-01", "2024-01-01")),
  xmax = as.Date(c("2020-12-31", "2024-12-31")),
  ymin = -Inf,
  ymax = Inf
)

library(tidyverse)
library(zoo)
library(patchwork)
library(grid)

# Text panel with the zone name, used as row label
make_label <- function(region_name, color = "grey20") {
  wrap_elements(
    grid::textGrob(
      label = region_name,
      gp    = grid::gpar(fontsize = 11, fontface = "bold", col = color),
      rot   = 0
    )
  )
}

# Solar and wind plots for one zone (solar_var = NULL gives an empty panel)
make_region_plot <- function(data, solar_var, wind_var,
                             solar_color, wind_color, region) {
  
  # Solar plot or spacer
  if (!is.null(solar_var)) {
    p_s <- ggplot(data, aes(x = date)) +
      geom_rect(data = shading,
                aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
                inherit.aes = FALSE,
                fill = "grey80", alpha = 0.4) +
      geom_line(aes(y = .data[[solar_var]]),
                color = solar_color, linewidth = 0.7, alpha = 0.8) +
      geom_smooth(aes(y = .data[[solar_var]]),
                  method    = "lm",
                  formula   = y ~ x + I(x^2),
                  se        = FALSE,
                  color     = solar_color,
                  linewidth = 1.2) +
      scale_x_date(date_breaks = "2 years", date_labels = "%Y",
                   limits = as.Date(c("2018-01-01", "2024-12-31"))) +
      labs(title = paste0(region, " Solar"), x = NULL, y = "MWh") +
      theme_bw() +
      theme(plot.title       = element_text(hjust = 0.5, face = "bold", size = 9),
            panel.grid.minor = element_blank(),
            axis.text.x      = element_text(size = 7))
  } else {
    p_s <- plot_spacer()
  }
  
  # Wind plot
  if (!is.null(wind_var)) {
    p_w <- ggplot(data, aes(x = date)) +
      geom_rect(data = shading,
                aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
                inherit.aes = FALSE,
                fill = "grey80", alpha = 0.4) +
      geom_line(aes(y = .data[[wind_var]]),
                color = wind_color, linewidth = 0.7, alpha = 0.8) +
      geom_smooth(aes(y = .data[[wind_var]]),
                  method    = "lm",
                  formula   = y ~ x + I(x^2),
                  se        = FALSE,
                  color     = wind_color,
                  linewidth = 1.2) +
      scale_x_date(date_breaks = "2 years", date_labels = "%Y",
                   limits = as.Date(c("2018-01-01", "2024-12-31"))) +
      labs(title = paste0(region, " Wind"), x = NULL, y = "MWh") +
      theme_bw() +
      theme(plot.title       = element_text(hjust = 0.5, face = "bold", size = 9),
            panel.grid.minor = element_blank(),
            axis.text.x      = element_text(size = 7))
  } else {
    p_w <- plot_spacer()
  }
  
  # Return solar and wind as separate plots (label added outside)
  list(solar = p_s, wind = p_w)
}

# Build plots per region
plots_dk1 <- make_region_plot(data_plot1_daily,
                              solar_var   = "solar_dk1_ma30",
                              wind_var    = "total_wind_dk1_ma30",
                              solar_color = "#E69F00",
                              wind_color  = "#E69F00", "DK1")

plots_dk2 <- make_region_plot(data_plot1_daily,
                              solar_var   = "solar_dk2_ma30",
                              wind_var    = "total_wind_dk2_ma30",
                              solar_color = "#56B4E9",
                              wind_color  = "#56B4E9", "DK2")

plots_se4 <- make_region_plot(data_plot1_daily,
                              solar_var   = "solar_se4_ma30",
                              wind_var    = "total_wind_se4_ma30",
                              solar_color = "#CC79A7",
                              wind_color  = "#CC79A7", "SE4")

plots_de  <- make_region_plot(data_plot1_daily,
                              solar_var   = "solar_de_ma30",
                              wind_var    = "total_wind_de_ma30",
                              solar_color = "#D55E00",
                              wind_color  = "#D55E00", "DE")

# NO2 - no solar available, pass NULL
plots_no2 <- make_region_plot(data_plot1_daily,
                              solar_var   = NULL,
                              wind_var    = "total_wind_no2_ma30",
                              solar_color = "#009E73",
                              wind_color  = "#009E73", "NO2")

# Assemble with region labels on the left
# Layout: [label | solar | wind] for each region
p_final <- (
plots_dk1$solar | plots_dk1$wind
) / (
  plots_dk2$solar | plots_dk2$wind
) / (
  plots_se4$solar | plots_se4$wind
) / (
  plots_de$solar  | plots_de$wind
) / (
   plots_no2$solar | plots_no2$wind)
# ) +
#   plot_layout(widths = c(0.5, 1, 1)) +
#   plot_annotation(
#     title    = "30-Day Moving Average of Renewable Production by Region (2018–2024)",
#     subtitle = "Shaded areas: 2020 and 2024 | Thick lines show quadratic trends",
#     theme    = theme(
#       plot.title    = element_text(hjust = 0.5, face = "bold", size = 13),
#       plot.subtitle = element_text(hjust = 0.5, size = 10, color = "grey40")
#     )
#   )

p_final
ggsave(filename = file.path(root, "figures/descriptives/evolution_renewables_longrun.png"),
       plot = p_final, width = 14, height = 12, dpi = 300)

# Fig 2: average intraday profile of solar and wind, 2020 vs 2024 ----
# One row per zone, solar left and wind right

# Mean production by hour of day, separately for 2020 and 2024
data_plot2 <- data_plot12_fin %>%
  filter(year %in% c(2020, 2024)) %>%
  mutate(hour = as.integer(as.character(hour))) %>%
  group_by(year, hour) %>%
  summarise(
    across(c(solar_dk1, solar_dk2, solar_se4, solar_de,
             total_wind_dk1, total_wind_dk2, total_wind_no2,
             total_wind_se4, total_wind_de),
           ~ mean(.x, na.rm = TRUE)),
    .groups = "drop"
  ) %>%
  arrange(year, hour)

# Split by year
d_2020 <- data_plot2 %>% filter(year == 2020)
d_2024 <- data_plot2 %>% filter(year == 2024)

# Shared x axis settings
x_scale <- scale_x_continuous(
  breaks = seq(0, 23, by = 6),
  labels = sprintf("%02d:00", seq(0, 23, by = 6)),
  limits = c(0, 23)
)

# Shared theme
plot_theme <- theme_bw() +
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x      = element_text(size = 7, angle = 45, hjust = 1),
    axis.title.y     = element_text(size = 7),
    axis.text.y      = element_text(size = 7),
    legend.position  = "none"
  )

# DK1 solar ----
ylim_dk1_solar <- range(c(d_2020$solar_dk1, d_2024$solar_dk1), na.rm = TRUE)
ylim_dk1_solar <- c(ylim_dk1_solar[1] - 0.05 * diff(ylim_dk1_solar),
                    ylim_dk1_solar[2] + 0.05 * diff(ylim_dk1_solar))

p_dk1_solar <- ggplot() +
  geom_line(data = d_2020,
            aes(x = hour, y = solar_dk1, linetype = "2020"),
            color = "#E69F00", linewidth = 0.8) +
  geom_line(data = d_2024,
            aes(x = hour, y = solar_dk1, linetype = "2024"),
            color = "#E69F00", linewidth = 0.8) +
  scale_linetype_manual(name = NULL,
                        values = c("2020" = "dashed", "2024" = "solid")) +
  scale_y_continuous(name = "MWh", limits = ylim_dk1_solar) +
  x_scale + labs(x = NULL, title = "DK1 Solar") + plot_theme

# DK1 wind ----
ylim_dk1_wind <- range(c(d_2020$total_wind_dk1, d_2024$total_wind_dk1), na.rm = TRUE)
ylim_dk1_wind <- c(ylim_dk1_wind[1] - 0.05 * diff(ylim_dk1_wind),
                   ylim_dk1_wind[2] + 0.05 * diff(ylim_dk1_wind))

p_dk1_wind <- ggplot() +
  geom_line(data = d_2020,
            aes(x = hour, y = total_wind_dk1, linetype = "2020"),
            color = "#E69F00", linewidth = 0.8) +
  geom_line(data = d_2024,
            aes(x = hour, y = total_wind_dk1, linetype = "2024"),
            color = "#E69F00", linewidth = 0.8) +
  scale_linetype_manual(name = NULL,
                        values = c("2020" = "dashed", "2024" = "solid")) +
  scale_y_continuous(name = "MWh", limits = ylim_dk1_wind) +
  x_scale + labs(x = NULL, title = "DK1 Wind") + plot_theme

# DK2 solar ----
ylim_dk2_solar <- range(c(d_2020$solar_dk2, d_2024$solar_dk2), na.rm = TRUE)
ylim_dk2_solar <- c(ylim_dk2_solar[1] - 0.05 * diff(ylim_dk2_solar),
                    ylim_dk2_solar[2] + 0.05 * diff(ylim_dk2_solar))

p_dk2_solar <- ggplot() +
  geom_line(data = d_2020,
            aes(x = hour, y = solar_dk2, linetype = "2020"),
            color = "#56B4E9", linewidth = 0.8) +
  geom_line(data = d_2024,
            aes(x = hour, y = solar_dk2, linetype = "2024"),
            color = "#56B4E9", linewidth = 0.8) +
  scale_linetype_manual(name = NULL,
                        values = c("2020" = "dashed", "2024" = "solid")) +
  scale_y_continuous(name = "MWh", limits = ylim_dk2_solar) +
  x_scale + labs(x = NULL, title = "DK2 Solar") + plot_theme

# DK2 wind ----
ylim_dk2_wind <- range(c(d_2020$total_wind_dk2, d_2024$total_wind_dk2), na.rm = TRUE)
ylim_dk2_wind <- c(ylim_dk2_wind[1] - 0.05 * diff(ylim_dk2_wind),
                   ylim_dk2_wind[2] + 0.05 * diff(ylim_dk2_wind))

p_dk2_wind <- ggplot() +
  geom_line(data = d_2020,
            aes(x = hour, y = total_wind_dk2, linetype = "2020"),
            color = "#56B4E9", linewidth = 0.8) +
  geom_line(data = d_2024,
            aes(x = hour, y = total_wind_dk2, linetype = "2024"),
            color = "#56B4E9", linewidth = 0.8) +
  scale_linetype_manual(name = NULL,
                        values = c("2020" = "dashed", "2024" = "solid")) +
  scale_y_continuous(name = "MWh", limits = ylim_dk2_wind) +
  x_scale + labs(x = NULL, title = "DK2 Wind") + plot_theme

# SE4 solar ----
ylim_se4_solar <- range(c(d_2020$solar_se4, d_2024$solar_se4), na.rm = TRUE)
ylim_se4_solar <- c(ylim_se4_solar[1] - 0.05 * diff(ylim_se4_solar),
                    ylim_se4_solar[2] + 0.05 * diff(ylim_se4_solar))

p_se4_solar <- ggplot() +
  geom_line(data = d_2020,
            aes(x = hour, y = solar_se4, linetype = "2020"),
            color = "#CC79A7", linewidth = 0.8) +
  geom_line(data = d_2024,
            aes(x = hour, y = solar_se4, linetype = "2024"),
            color = "#CC79A7", linewidth = 0.8) +
  scale_linetype_manual(name = NULL,
                        values = c("2020" = "dashed", "2024" = "solid")) +
  scale_y_continuous(name = "MWh", limits = ylim_se4_solar) +
  x_scale + labs(x = NULL, title = "SE4 Solar") + plot_theme

# SE4 wind ----
ylim_se4_wind <- range(c(d_2020$total_wind_se4, d_2024$total_wind_se4), na.rm = TRUE)
ylim_se4_wind <- c(ylim_se4_wind[1] - 0.05 * diff(ylim_se4_wind),
                   ylim_se4_wind[2] + 0.05 * diff(ylim_se4_wind))

p_se4_wind <- ggplot() +
  geom_line(data = d_2020,
            aes(x = hour, y = total_wind_se4, linetype = "2020"),
            color = "#CC79A7", linewidth = 0.8) +
  geom_line(data = d_2024,
            aes(x = hour, y = total_wind_se4, linetype = "2024"),
            color = "#CC79A7", linewidth = 0.8) +
  scale_linetype_manual(name = NULL,
                        values = c("2020" = "dashed", "2024" = "solid")) +
  scale_y_continuous(name = "MWh", limits = ylim_se4_wind) +
  x_scale + labs(x = NULL, title = "SE4 Wind") + plot_theme

# DE solar ----
ylim_de_solar <- range(c(d_2020$solar_de, d_2024$solar_de), na.rm = TRUE)
ylim_de_solar <- c(ylim_de_solar[1] - 0.05 * diff(ylim_de_solar),
                   ylim_de_solar[2] + 0.05 * diff(ylim_de_solar))

p_de_solar <- ggplot() +
  geom_line(data = d_2020,
            aes(x = hour, y = solar_de, linetype = "2020"),
            color = "#D55E00", linewidth = 0.8) +
  geom_line(data = d_2024,
            aes(x = hour, y = solar_de, linetype = "2024"),
            color = "#D55E00", linewidth = 0.8) +
  scale_linetype_manual(name = NULL,
                        values = c("2020" = "dashed", "2024" = "solid")) +
  scale_y_continuous(name = "MWh", limits = ylim_de_solar) +
  x_scale + labs(x = NULL, title = "DE Solar") + plot_theme

# DE wind ----
ylim_de_wind <- range(c(d_2020$total_wind_de, d_2024$total_wind_de), na.rm = TRUE)
ylim_de_wind <- c(ylim_de_wind[1] - 0.05 * diff(ylim_de_wind),
                  ylim_de_wind[2] + 0.05 * diff(ylim_de_wind))

p_de_wind <- ggplot() +
  geom_line(data = d_2020,
            aes(x = hour, y = total_wind_de, linetype = "2020"),
            color = "#D55E00", linewidth = 0.8) +
  geom_line(data = d_2024,
            aes(x = hour, y = total_wind_de, linetype = "2024"),
            color = "#D55E00", linewidth = 0.8) +
  scale_linetype_manual(name = NULL,
                        values = c("2020" = "dashed", "2024" = "solid")) +
  scale_y_continuous(name = "MWh", limits = ylim_de_wind) +
  x_scale + labs(x = NULL, title = "DE Wind") + plot_theme

# NO2 solar - spacer (no data) ----
p_no2_solar <- plot_spacer()

# NO2 wind ----
ylim_no2_wind <- range(c(d_2020$total_wind_no2, d_2024$total_wind_no2), na.rm = TRUE)
ylim_no2_wind <- c(ylim_no2_wind[1] - 0.05 * diff(ylim_no2_wind),
                   ylim_no2_wind[2] + 0.05 * diff(ylim_no2_wind))

p_no2_wind <- ggplot() +
  geom_line(data = d_2020,
            aes(x = hour, y = total_wind_no2, linetype = "2020"),
            color = "#009E73", linewidth = 0.8) +
  geom_line(data = d_2024,
            aes(x = hour, y = total_wind_no2, linetype = "2024"),
            color = "#009E73", linewidth = 0.8) +
  scale_linetype_manual(name = NULL,
                        values = c("2020" = "dashed", "2024" = "solid")) +
  scale_y_continuous(name = "MWh", limits = ylim_no2_wind) +
  x_scale + labs(x = NULL, title = "NO2 Wind") + plot_theme

# Print each panel to check it before assembling ----
p_dk1_solar
p_dk1_wind
p_dk2_solar
p_dk2_wind
p_se4_solar
p_se4_wind
p_de_solar
p_de_wind
p_no2_wind

# Assemble main grid
p2_final <- (
  p_dk1_solar | p_dk1_wind
) / (
  p_dk2_solar | p_dk2_wind
) / (
  p_se4_solar | p_se4_wind
) / (
  p_de_solar  | p_de_wind
) / (
  p_no2_solar | p_no2_wind
) #   plot_layout(widths = unit(c(0.5, 1, 1), c("cm", "null", "null"))) +
#   plot_annotation(
#     title    = "Average Intraday Renewable Production Profile by Region",
#     subtitle = "Hourly averages across all days | Solid: 2024, Dashed: 2020",
#     theme    = theme(
#       plot.title    = element_text(hjust = 0.5, face = "bold", size = 13),
#       plot.subtitle = element_text(hjust = 0.5, size = 10, color = "grey40")
#     )
#   )

p2_final
ggsave(filename = file.path(root, "figures/descriptives/evolution_renewables_overday.png"),
       plot = p2_final, width = 14, height = 12, dpi = 300)

# Transmission over time ----
# The commented-out block below was a first attempt at merging all flows;
# kept for reference.
# 
# data_plot1222 <- dataset_total %>%
#   select(date,
#     # Actual renewable production (all regions)
#     # Electricity flows
#     matches("^flow_"),
#     matches
#     
#   ) %>%
#   select(
#     date,
#     # Electricity flows for all regions
#     matches("^dk1_"),
#     matches("^dk2_"),
#     matches("^flow_se4_"),
#     matches("^flow_de_"),
#     matches("^no2_")
#   )
# # Step 1: Pivot to long, extract region pair and hour
# data_flows_daily <- data_plot1222 %>%
#   pivot_longer(
#     cols      = -date,
#     names_to  = "variable",
#     values_to = "flow"
#   ) %>%
#   # Extract region pair by removing flow_ prefix and _hXX suffix
#   mutate(
#     region_pair = str_remove(variable, "^flow_") %>%
#       str_remove("_h\\d{2}$")
#   ) %>%
#   # Sum across all hours per date and region pair
#   group_by(date, region_pair) %>%
#   summarise(
#     flow_daily = sum(flow, na.rm = TRUE),
#     .groups = "drop"
#   ) %>%
#   # Pivot wide so one column per region pair
#   pivot_wider(
#     names_from  = region_pair,
#     values_from = flow_daily,
#     names_prefix = "flow_"
#   ) %>%
#   # Ensure date is same type for merging
#   mutate(date = as.Date(date)) %>%
#   arrange(date)
# 
# # Step 2: Merge with data_plot3_daily
# data_plot3_merged <- data_plot3_daily %>%
#   left_join(data_flows_daily, by = "date")
# 
# # Check
# glimpse(data_plot3_merged)
# nrow(data_plot3_daily)
# nrow(data_plot3_merged)  # should be same as above
# 
# # Plot 3: 

# Fig 3: DK1 net imports (left) and total absolute flow (right) ----
df_plot3 <- data_plot12_fin 
library(tidyverse)
library(zoo)
library(patchwork)

# Daily means and 30-day trailing moving averages
data_plot3_daily <- data_plot12_fin %>%
  mutate(date = as.Date(paste(year, month, day, sep = "-"))) %>%
  group_by(date) %>%
  summarise(
    net_inflow_to_DK1  = mean(net_inflow_to_DK1,  na.rm = TRUE),
    total_abs_flow_DK1 = mean(total_abs_flow_DK1, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(date) %>%
  mutate(
    net_inflow_ma30      = rollmean(net_inflow_to_DK1,  k = 30, fill = NA, align = "right"),
    total_abs_flow_ma30  = rollmean(total_abs_flow_DK1, k = 30, fill = NA, align = "right")
  )

# Shading
shading <- tibble(
  xmin = as.Date(c("2020-01-01", "2024-01-01")),
  xmax = as.Date(c("2020-12-31", "2024-12-31")),
  ymin = -Inf,
  ymax = Inf
)

# Net inflow plot (left)
p_net_inflow <- ggplot(data_plot3_daily, aes(x = date)) +
  geom_rect(data = shading,
            aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
            inherit.aes = FALSE,
            fill = "grey80", alpha = 0.4) +
  geom_hline(yintercept = 0, linetype = "dashed",
             color = "grey40", linewidth = 0.4) +
  geom_line(aes(y = net_inflow_ma30), color = "#0072B2", linewidth = 0.7) +
  scale_x_date(
    date_breaks = "1 year",
    date_labels = "%Y",
    limits      = as.Date(c("2018-01-01", "2024-12-31"))
  ) +
  labs(
    title = "Net Inflow to DK1",
    x     = NULL,
    y     = "MWh"
  ) +
  theme_bw() +
  theme(
    plot.title       = element_text(hjust = 0.5, face = "bold"),
    panel.grid.minor = element_blank()
  )

# Total absolute flow plot (right)
p_abs_flow <- ggplot(data_plot3_daily, aes(x = date)) +
  geom_rect(data = shading,
            aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
            inherit.aes = FALSE,
            fill = "grey80", alpha = 0.4) +
  geom_line(aes(y = total_abs_flow_ma30), color = "#D55E00", linewidth = 0.7) +
  scale_x_date(
    date_breaks = "1 year",
    date_labels = "%Y",
    limits      = as.Date(c("2018-01-01", "2024-12-31"))
  ) +
  labs(
    title = "Total Absolute Flow DK1",
    x     = NULL,
    y     = "MWh"
  ) +
  theme_bw() +
  theme(
    plot.title       = element_text(hjust = 0.5, face = "bold"),
    panel.grid.minor = element_blank()
  )

# Combine
p3_combined <- p_net_inflow | p_abs_flow
p3_combined 
ggsave(
  filename = file.path(root, "figures/descriptives/electricity_flows.png"),
  plot     = p3_combined,
  width    = 14,
  height   = 4,    # much shorter than before
  dpi      = 300
)

# Fig 4: net and absolute flow per DK1 border ----
# In lear_dataset.csv the flow_dk1_<partner> columns are already NET flows
# (out - in, seen from DK1; see data_prep/build_lear_dataset.py), so the net import
# from a partner is -flow_dk1_<partner>. Gross volumes per border are not in
# that file, so the "absolute flow" per border is the absolute net flow; only
# the totals (flow_dk1_in/out) are gross.

data_plot444_daily <- dataset_total %>%
  select(date, matches("_dk1_|_dk1$|^dk1_"))

data_flows_dk1_daily <- data_plot444_daily %>%
  pivot_longer(
    cols      = -date,
    names_to  = "variable",
    values_to = "value"
  ) %>%
  # Extract flow pair by removing _hXX suffix
  mutate(
    flow_pair = str_remove(variable, "_h\\d{2}$")
  ) %>%
  # Daily total over the 24 hours
  group_by(date, flow_pair) %>%
  summarise(
    value = sum(value, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  pivot_wider(
    names_from  = flow_pair,
    values_from = value
  ) %>%
  mutate(date = as.Date(date)) %>%
  arrange(date) %>%
  mutate(
    # Net import into DK1 per border (flow_dk1_<p> = out - in, > 0 = export)
    net_inflow_delu = -flow_dk1_de_lu,
    net_inflow_dk2  = -flow_dk1_dk2,
    net_inflow_no2  = -flow_dk1_no2,
    net_inflow_se3  = -flow_dk1_se3,

    # Absolute net flow per border
    abs_flow_delu   = abs(flow_dk1_de_lu),
    abs_flow_dk2    = abs(flow_dk1_dk2),
    abs_flow_no2    = abs(flow_dk1_no2),
    abs_flow_se3    = abs(flow_dk1_se3),

    # Totals over all borders, from the gross in- and outflows
    net_inflow_total = flow_dk1_in - flow_dk1_out,
    abs_flow_total   = flow_dk1_in + flow_dk1_out
  )

# Check
glimpse(data_flows_dk1_daily)

# 30-day trailing moving averages used in the plots
data_plot44_daily <- data_flows_dk1_daily %>%
  arrange(date) %>%
  mutate(across(c(net_inflow_total, net_inflow_dk2, net_inflow_delu,
                  net_inflow_no2, net_inflow_se3,
                  abs_flow_total, abs_flow_dk2, abs_flow_delu,
                  abs_flow_no2, abs_flow_se3),
                ~ rollmean(.x, k = 30, fill = NA, align = "right"),
                .names = "{.col}_ma30"))

# Shared elements
shading <- tibble(
  xmin = as.Date(c("2020-01-01", "2024-01-01")),
  xmax = as.Date(c("2020-12-31", "2024-12-31")),
  ymin = -Inf,
  ymax = Inf
)

x_scale <- scale_x_date(
  date_breaks = "1 year",
  date_labels = "%Y",
  limits      = as.Date(c("2018-01-01", "2024-12-31"))
)

flow_theme <- theme_bw() +
  theme(
    plot.title       = element_text(hjust = 0.5, face = "bold", size = 9),
    panel.grid.minor = element_blank(),
    axis.text.x      = element_text(size = 7, angle = 45, hjust = 1),
    axis.title       = element_text(size = 7),
    axis.text.y      = element_text(size = 7)
  )

shading_layer <- geom_rect(
  data        = shading,
  aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
  inherit.aes = FALSE,
  fill        = "grey80", alpha = 0.4
)

hline_layer <- geom_hline(
  yintercept = 0, linetype = "dashed",
  color = "grey40", linewidth = 0.4
)

# Compute shared y limits per row ----
# Axis limits with 5% padding
add_padding <- function(lim) {
  pad <- 0.05 * diff(lim)
  c(lim[1] - pad, lim[2] + pad)
}

# Limits for net inflow (can be negative)
lim_net_total <- add_padding(range(data_plot44_daily$net_inflow_total_ma30, na.rm = TRUE))
lim_net_dk2   <- add_padding(range(data_plot44_daily$net_inflow_dk2_ma30,   na.rm = TRUE))
lim_net_delu  <- add_padding(range(data_plot44_daily$net_inflow_delu_ma30,  na.rm = TRUE))
lim_net_no2   <- add_padding(range(data_plot44_daily$net_inflow_no2_ma30,   na.rm = TRUE))
lim_net_se3   <- add_padding(range(data_plot44_daily$net_inflow_se3_ma30,   na.rm = TRUE))

# Limits for absolute flow (always from 0)
lim_abs_total <- c(0, max(data_plot44_daily$abs_flow_total_ma30, na.rm = TRUE) * 1.05)
lim_abs_dk2   <- c(0, max(data_plot44_daily$abs_flow_dk2_ma30,   na.rm = TRUE) * 1.05)
lim_abs_delu  <- c(0, max(data_plot44_daily$abs_flow_delu_ma30,  na.rm = TRUE) * 1.05)
lim_abs_no2   <- c(0, max(data_plot44_daily$abs_flow_no2_ma30,   na.rm = TRUE) * 1.05)
lim_abs_se3   <- c(0, max(data_plot44_daily$abs_flow_se3_ma30,   na.rm = TRUE) * 1.05)

# Left column: net inflow ----
p_net_total <- ggplot(data_plot44_daily, aes(x = date)) +
  shading_layer + hline_layer +
  geom_line(aes(y = net_inflow_total_ma30), color = "#0072B2", linewidth = 0.7) +
  geom_smooth(aes(y = net_inflow_total_ma30), method = "lm",
              formula = y ~ x, se = FALSE,
              color = "lightblue", linewidth = 0.5, linetype = "solid") +
  x_scale +
  scale_y_continuous(limits = lim_net_total) +
  labs(title = "Net Inflow — Total", x = NULL, y = "MWh") + flow_theme

p_net_dk2 <- ggplot(data_plot44_daily, aes(x = date)) +
  shading_layer + hline_layer +
  geom_line(aes(y = net_inflow_dk2_ma30), color = "#0072B2", linewidth = 0.7) +
  geom_smooth(aes(y = net_inflow_dk2_ma30), method = "lm",
              formula = y ~ x, se = FALSE,
              color = "lightblue", linewidth = 0.5, linetype = "solid") +
  x_scale +
  scale_y_continuous(limits = lim_net_dk2) +
  labs(title = "Net Inflow — DK2", x = NULL, y = "MWh") + flow_theme

p_net_delu <- ggplot(data_plot44_daily, aes(x = date)) +
  shading_layer + hline_layer +
  geom_line(aes(y = net_inflow_delu_ma30), color = "#0072B2", linewidth = 0.7) +
  geom_smooth(aes(y = net_inflow_delu_ma30), method = "lm",
              formula = y ~ x, se = FALSE,
              color = "lightblue", linewidth = 0.5, linetype = "solid") +
  x_scale +
  scale_y_continuous(limits = lim_net_delu) +
  labs(title = "Net Inflow — DE", x = NULL, y = "MWh") + flow_theme

p_net_no2 <- ggplot(data_plot44_daily, aes(x = date)) +
  shading_layer + hline_layer +
  geom_line(aes(y = net_inflow_no2_ma30), color = "#0072B2", linewidth = 0.7) +
  geom_smooth(aes(y = net_inflow_no2_ma30), method = "lm",
              formula = y ~ x, se = FALSE,
              color = "lightblue", linewidth = 0.5, linetype = "solid") +
  x_scale +
  scale_y_continuous(limits = lim_net_no2) +
  labs(title = "Net Inflow — NO2", x = NULL, y = "MWh") + flow_theme

p_net_se3 <- ggplot(data_plot44_daily, aes(x = date)) +
  shading_layer + hline_layer +
  geom_line(aes(y = net_inflow_se3_ma30), color = "#0072B2", linewidth = 0.7) +
  geom_smooth(aes(y = net_inflow_se3_ma30), method = "lm",
              formula = y ~ x, se = FALSE,
              color = "lightblue", linewidth = 0.5, linetype = "solid") +
  x_scale +
  scale_y_continuous(limits = lim_net_se3) +
  labs(title = "Net Inflow — SE3", x = NULL, y = "MWh") + flow_theme

# Right column: absolute flow (y from 0) ----
p_abs_total <- ggplot(data_plot44_daily, aes(x = date)) +
  shading_layer +
  geom_line(aes(y = abs_flow_total_ma30), color = "#D55E00", linewidth = 0.7) +
  geom_smooth(aes(y = abs_flow_total_ma30), method = "lm",
              formula = y ~ x, se = FALSE,
              color = "#F0A868", linewidth = 0.5, linetype = "solid") +
  x_scale +
  scale_y_continuous(limits = lim_abs_total) +
  labs(title = "Abs. Flow — Total", x = NULL, y = "MWh") + flow_theme

p_abs_dk2 <- ggplot(data_plot44_daily, aes(x = date)) +
  shading_layer +
  geom_line(aes(y = abs_flow_dk2_ma30), color = "#D55E00", linewidth = 0.7) +
  geom_smooth(aes(y = abs_flow_dk2_ma30), method = "lm",
              formula = y ~ x, se = FALSE,
              color = "#F0A868", linewidth = 0.5, linetype = "solid") +
  x_scale +
  scale_y_continuous(limits = lim_abs_dk2) +
  labs(title = "Abs. Flow — DK2", x = NULL, y = "MWh") + flow_theme

p_abs_delu <- ggplot(data_plot44_daily, aes(x = date)) +
  shading_layer +
  geom_line(aes(y = abs_flow_delu_ma30), color = "#D55E00", linewidth = 0.7) +
  geom_smooth(aes(y = abs_flow_delu_ma30), method = "lm",
              formula = y ~ x, se = FALSE,
              color = "#F0A868", linewidth = 0.5, linetype = "solid") +
  x_scale +
  scale_y_continuous(limits = lim_abs_delu) +
  labs(title = "Abs. Flow — DE", x = NULL, y = "MWh") + flow_theme

p_abs_no2 <- ggplot(data_plot44_daily, aes(x = date)) +
  shading_layer +
  geom_line(aes(y = abs_flow_no2_ma30), color = "#D55E00", linewidth = 0.7) +
  geom_smooth(aes(y = abs_flow_no2_ma30), method = "lm",
              formula = y ~ x, se = FALSE,
              color = "#F0A868", linewidth = 0.5, linetype = "solid") +
  x_scale +
  scale_y_continuous(limits = lim_abs_no2) +
  labs(title = "Abs. Flow — NO2", x = NULL, y = "MWh") + flow_theme

p_abs_se3 <- ggplot(data_plot44_daily, aes(x = date)) +
  shading_layer +
  geom_line(aes(y = abs_flow_se3_ma30), color = "#D55E00", linewidth = 0.7) +
  geom_smooth(aes(y = abs_flow_se3_ma30), method = "lm",
              formula = y ~ x, se = FALSE,
              color = "#F0A868", linewidth = 0.5, linetype = "solid") +
  x_scale +
  scale_y_continuous(limits = lim_abs_se3) +
  labs(title = "Abs. Flow — SE3", x = NULL, y = "MWh") + flow_theme

# Assemble ----
p5_final <- (
  p_net_total | p_abs_total
) / (
  p_net_dk2   | p_abs_dk2
) / (
  p_net_delu  | p_abs_delu
) / (
  p_net_no2   | p_abs_no2
) / (
  p_net_se3   | p_abs_se3
) +
  plot_annotation(
    title    = "30-Day Moving Average of DK1 Electricity Flows by Trading Partner (2018–2024)",
    subtitle = "Left: Net Inflow to DK1 | Right: Absolute Flow | Shaded: 2020 and 2024",
    theme    = theme(
      plot.title    = element_text(hjust = 0.5, face = "bold", size = 13),
      plot.subtitle = element_text(hjust = 0.5, size = 10, color = "grey40")
    )
  )

p5_final

ggsave(
  filename = file.path(root, "figures/descriptives/electricity_flows_by_region.png"),
  plot     = p5_final,
  width    = 14,
  height   = 16,
  dpi      = 300
)

# 
# Earlier version of a DK1-NO2 flow figure, switched off:
# # PLOT 4 NORWAY AND DENMARK TRANSMISSIONS
# 
# # Step 1: Prepare data correctly
# data_plot4_daily <- data_plot12 %>%
#   mutate(date = as.Date(paste(year, month, day, sep = "-"))) %>%
#   group_by(date) %>%
#   summarise(
#     flow_dk1_to_no2 = mean(`flow_dk1_DK1_to_NO2`,  na.rm = TRUE),
#     flow_no2_to_dk1 = mean(`flow_dk1_NO2_to_DK1`,  na.rm = TRUE),
#     .groups = "drop"
#   ) %>%
#   arrange(date) %>%
#   mutate(
#     flow_dk1_to_no2_ma30 = rollmean(flow_dk1_to_no2, k = 60, fill = NA, align = "right"),
#     flow_no2_to_dk1_ma30 = rollmean(flow_no2_to_dk1, k = 60, fill = NA, align = "right")
#   )
# 
# # Step 2: Shading
# shading <- tibble(
#   xmin = as.Date(c("2020-01-01", "2024-01-01")),
#   xmax = as.Date(c("2020-12-31", "2024-12-31")),
#   ymin = -Inf,
#   ymax = Inf
# )
# 
# # Step 3: DK1 to NO2 plot (left)
# p_dk1_to_no2 <- ggplot(data_plot4_daily, aes(x = date)) +
#   geom_rect(data = shading,
#             aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
#             inherit.aes = FALSE,
#             fill = "grey80", alpha = 0.4) +
#   geom_hline(yintercept = 0, linetype = "dashed",
#              color = "grey40", linewidth = 0.4) +
#   geom_line(aes(y = flow_dk1_to_no2_ma30),
#             color = "#0072B2", linewidth = 0.7) +
#   geom_line(aes(y = flow_no2_to_dk1_ma30),
#             color = "orange", linewidth = 0.7) +
#   scale_x_date(
#     date_breaks = "1 year",
#     date_labels = "%Y",
#     limits      = as.Date(c("2018-01-01", "2024-12-31"))
#   ) +
#   labs(
#     title = "Flows DK1 <-> NO2",
#     x     = NULL,
#     y     = "MWh"
#   ) +
#   theme_bw() +
#   theme(
#     plot.title       = element_text(hjust = 0.5, face = "bold"),
#     panel.grid.minor = element_blank()
#   )
# 
# # Step 4: NO2 to DK1 plot (right)
# p_no2_to_dk1 <- ggplot(data_plot4_daily, aes(x = date)) +
#   geom_rect(data = shading,
#             aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
#             inherit.aes = FALSE,
#             fill = "grey80", alpha = 0.4) +
#   geom_hline(yintercept = 0, linetype = "dashed",
#              color = "grey40", linewidth = 0.4) +
#   geom_line(aes(y = flow_dk1_to_no2_ma30 - flow_no2_to_dk1_ma30),
#             color = "#0072B2", linewidth = 0.7) +
#   scale_x_date(
#     date_breaks = "1 year",
#     date_labels = "%Y",
#     limits      = as.Date(c("2018-01-01", "2024-12-31"))
#   ) +
#   labs(
#     title = "Net Flow NO2 → DK1",
#     x     = NULL,
#     y     = "MWh"
#   ) +
#   theme_bw() +
#   theme(
#     plot.title       = element_text(hjust = 0.5, face = "bold"),
#     panel.grid.minor = element_blank()
#   )
# 
# # Step 5: Combine
# p4 <- p_dk1_to_no2 | p_no2_to_dk1
# # p4 <- p4 + plot_annotation(
# #   title    = "30-Day Moving Average of Electricity Flows between DK1 and NO2 (2018–2024)",
# #   subtitle = "Shaded areas: 2020 and 2024",
# #   theme    = theme(
# #     plot.title    = element_text(hjust = 0.5, face = "bold", size = 13),
# #     plot.subtitle = element_text(hjust = 0.5, size = 10, color = "grey40")
# #   )
# # )
# 
# p4
# 
# # ONLY TO FROM DK1
# # TOTAL INFLOW OUT FLOW/ SUM ABSOLUTE FLOW
# 
# 
# # Step 6: Save
# ggsave(
#   filename = file.path(root, "figures/descriptives/electricity_flows_DK_NO.png"),
#   plot     = p4,
#   width    = 14,
#   height   = 6,
#   dpi      = 300
# )
# Still to do: price convergence across zones (moving averages, trends,
# correlation matrices for 2020 and 2024, rolling correlations with DK1;
# done in Python in cross_country_correlations.py), and an OLS of the
# price on all production sources.
# Plot 4: Prices of each price zone over time: do they converge? 2018 to 2024

# CONVERGENCE OF PRICES ACROSS REGIONS Moving average and then 

# Same moving average and yearly time trend


# Heatmap with price correlations for both years. Hourly prices correlation:
# Correlation matrices. 2020 and have it for 2024. and 2018 - 2024 
# Plot of rolling correlations DK1 to each other one



# OLS ALL PRODUCTION SOURCES ON PRICE

# Descriptive Table 1:



