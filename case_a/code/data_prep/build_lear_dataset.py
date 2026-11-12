"""
Builds the daily input file for the LEAR and LightGBM models.

One row per UTC date (2018-2024), with every hourly series spread over 24
columns (_h00 ... _h23):

  price_h..              day-ahead price (EUR/MWh)
  load_forecast_h..      day-ahead total load forecast (MW)
  ren_forecast_h..       onshore + offshore + solar day-ahead forecast (MW)
  wind_forecast_h..      onshore + offshore only
  solar_forecast_h..     solar only
  flow_out_h.., flow_in_h..   total physical flows out of / into DK1 (MW)
  flow_<area>_h..        net flow to each neighbour (out - in, so > 0 = export)
  coal/eua/natural_gas/oil_price      daily, forward-filled over non-trading days
  eurostoxx50/msci_eu/usd_eur/vix     daily, same treatment
  dow_Mon ... dow_Sun    day-of-week dummies

Lags and the asinh transform are done later, inside models/lear/core.py.

Input:  data/interim/   (from process_interim.py)
Output: data/prepared/lear_dataset.csv
"""

from pathlib import Path

import pandas as pd

ROOT    = Path(__file__).parent.parent.parent
INTERIM = ROOT / "data/interim"
OUT_DIR = ROOT / "data/prepared"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_mtu_cet_to_utc(series):
    """Start of an MTU interval given in Danish local time, converted to UTC.

    DST is handled by tz_localize: the repeated autumn hour is inferred from
    row order and the missing spring hour is shifted forward.
    """
    start = (series.str.split(" - ").str[0]
             .str.replace(r"\s*\(CES?T\)", "", regex=True)
             .str.strip())
    t = pd.to_datetime(start, format="mixed", dayfirst=True)
    t = t.dt.tz_localize("Europe/Copenhagen",
                          ambiguous="infer",
                          nonexistent="shift_forward")
    return t.dt.tz_convert("UTC").dt.tz_localize(None)


# ---------------------------------------------------------------------------
# Step 1: Prices -> daily wide (UTC)
# ---------------------------------------------------------------------------
print("Step 1: Prices...", flush=True)
prices = pd.read_csv(INTERIM / "ElectricityPrices/DK1/Prices.csv", low_memory=False)
prices["time"] = pd.to_datetime(prices["time"])  # this file is already in UTC
prices["date"] = prices["time"].dt.date
prices["hour"] = prices["time"].dt.hour
price_wide = prices.pivot_table(
    index="date", columns="hour",
    values="Day-ahead Price (EUR/MWh)", aggfunc="first"
)
price_wide.columns = [f"price_h{h:02d}" for h in price_wide.columns]
print(f"  Price wide shape: {price_wide.shape}", flush=True)

# ---------------------------------------------------------------------------
# Step 2: Load forecast -> daily wide (CET/CEST -> UTC)
# ---------------------------------------------------------------------------
print("Step 2: Load forecast...", flush=True)
loads = pd.read_csv(INTERIM / "ElectricityPrices/DK1/Loads.csv", low_memory=False)
loads["time"] = parse_mtu_cet_to_utc(loads["MTU (CET/CEST)"])
loads["date"] = loads["time"].dt.date
loads["hour"] = loads["time"].dt.hour
load_wide = loads.pivot_table(
    index="date", columns="hour",
    values="Day-ahead Total Load Forecast (MW)", aggfunc="first"
)
load_wide.columns = [f"load_forecast_h{h:02d}" for h in load_wide.columns]
print(f"  Load wide shape: {load_wide.shape}", flush=True)

# ---------------------------------------------------------------------------
# Step 3: Renewable forecasts -> total, wind only, solar only
# ---------------------------------------------------------------------------
print("Step 3: Renewable forecast...", flush=True)
ren = pd.read_csv(INTERIM / "ElectricityPrices/DK1/Renewables.csv", low_memory=False)
ren["time"] = parse_mtu_cet_to_utc(ren["MTU (CET/CEST)"])
ren["Day-ahead (MW)"] = pd.to_numeric(ren["Day-ahead (MW)"], errors="coerce")

def _to_wide(series_df, col, prefix):
    # Sum over renewable types per hour, then one row per day and one column per hour
    s = series_df.groupby("time")[col].sum().rename(prefix).reset_index()
    s["date"] = s["time"].dt.date
    s["hour"] = s["time"].dt.hour
    wide = s.pivot_table(index="date", columns="hour", values=prefix, aggfunc="first")
    wide.columns = [f"{prefix}_h{h:02d}" for h in wide.columns]
    return wide

ren_wide   = _to_wide(ren,                                       "Day-ahead (MW)", "ren_forecast")
wind_wide  = _to_wide(ren[ren["Type"].isin(["OFFSHORE","ONSHORE"])], "Day-ahead (MW)", "wind_forecast")
solar_wide = _to_wide(ren[ren["Type"] == "SOLAR"],               "Day-ahead (MW)", "solar_forecast")
print(f"  Ren wide shape:   {ren_wide.shape}", flush=True)
print(f"  Wind wide shape:  {wind_wide.shape}", flush=True)
print(f"  Solar wide shape: {solar_wide.shape}", flush=True)

# ---------------------------------------------------------------------------
# Step 4: Transmission flows -> net flow per neighbour + total in/out
# ---------------------------------------------------------------------------
print("Step 4: Transmission flows...", flush=True)
trans = pd.read_csv(INTERIM / "ElectricityPrices/DK1/Transmission.csv", low_memory=False)
# Each hour appears once per border, so each distinct MTU string is parsed once
# and mapped back. The repeated autumn hour is labelled "(CEST)" / "(CET)" in
# the file, and that label decides which of the two occurrences it is.
unique_mtu = trans["MTU"].drop_duplicates()
_raw_start = unique_mtu.str.split(" - ").str[0]
_is_summer = ~_raw_start.str.contains("(CET)", regex=False)
_starts = _raw_start.str.replace(r"\s*\(CES?T\)", "", regex=True).str.strip()
_naive = pd.to_datetime(_starts, format="mixed", dayfirst=True)
_utc = (_naive.dt.tz_localize("Europe/Copenhagen",
                               ambiguous=_is_summer.to_numpy(),
                               nonexistent="shift_forward")
        .dt.tz_convert("UTC").dt.tz_localize(None))
trans["time"] = trans["MTU"].map(dict(zip(unique_mtu.values, _utc.values)))
trans["Physical Flow (MW)"] = pd.to_numeric(trans["Physical Flow (MW)"], errors="coerce")

# Identify flows into and out of DK1
dk1_out = trans[trans["Out Area"] == "BZN|DK1"].copy()
dk1_in  = trans[trans["In Area"]  == "BZN|DK1"].copy()

def _flow_wide(flow_df, partner_col, areas, prefix):
    """Hourly flows to/from the given partner areas, summed and reshaped to day x hour.

    Hours with no reported flow are set to 0.
    """
    subset = flow_df[flow_df[partner_col].isin(areas)].copy()
    hourly = subset.groupby("time")["Physical Flow (MW)"].sum().rename("flow")
    hourly = hourly.reindex(
        pd.date_range(hourly.index.min(), hourly.index.max(), freq="h"), fill_value=0
    ).reset_index()
    hourly.columns = ["time", "flow"]
    hourly["date"] = hourly["time"].dt.date
    hourly["hour"] = hourly["time"].dt.hour
    wide = hourly.pivot_table(index="date", columns="hour", values="flow", aggfunc="first")
    wide.columns = [f"{prefix}_h{h:02d}" for h in wide.columns]
    return wide.fillna(0)

# Neighbouring bidding zones. Germany appears as DE-AT-LU before the 2018 split
# and as DE-LU after it; the two never overlap, so they are summed into one channel.
CHANNEL_DEFS = {
    "germany": ["BZN|DE-AT-LU", "BZN|DE-LU"],
    "dk2":     ["BZN|DK2"],
    "gb":      ["BZN|GB"],
    "nl":      ["BZN|NL"],
    "no2":     ["BZN|NO2"],
    "se3":     ["BZN|SE3"],
}

channel_wides = {}
for ch, areas in CHANNEL_DEFS.items():
    out_w = _flow_wide(dk1_out, "In Area",  areas, f"_out_{ch}")
    in_w  = _flow_wide(dk1_in,  "Out Area", areas, f"_in_{ch}")
    # Make sure both sides cover the same days before subtracting
    idx = out_w.index.union(in_w.index)
    out_w = out_w.reindex(idx, fill_value=0)
    in_w  = in_w.reindex(idx, fill_value=0)
    # Net flow = DK1 out - DK1 in (positive = exporting)
    net = out_w.values - in_w.values
    net_df = pd.DataFrame(net, index=idx, columns=[f"flow_{ch}_h{h:02d}" for h in range(24)])
    channel_wides[ch] = net_df

# Total flows over all neighbours
all_areas = [a for areas in CHANNEL_DEFS.values() for a in areas]
agg_out_w = _flow_wide(dk1_out, "In Area",  all_areas, "_out_all")
agg_in_w  = _flow_wide(dk1_in,  "Out Area", all_areas, "_in_all")
agg_idx = agg_out_w.index.union(agg_in_w.index)
agg_out_w = agg_out_w.reindex(agg_idx, fill_value=0)
agg_in_w  = agg_in_w.reindex(agg_idx, fill_value=0)
agg_out_w.columns = [f"flow_out_h{h:02d}" for h in range(24)]
agg_in_w.columns  = [f"flow_in_h{h:02d}"  for h in range(24)]

print(f"  Channels: {list(CHANNEL_DEFS.keys())}", flush=True)
for ch, w in channel_wides.items():
    print(f"    flow_{ch}: {w.shape}", flush=True)

# ---------------------------------------------------------------------------
# Step 5: Fuel/carbon prices (daily scalars from EnergySeries.csv)
# ---------------------------------------------------------------------------
print("Step 5: Fuel/carbon prices...", flush=True)
energy = pd.read_csv(INTERIM / "EnergySeries.csv",
                     usecols=["Date", "Coal_Price", "EUA_Price", "Natural_Gas_Price", "Oil_Price"])
energy = energy.rename(columns={
    "Date": "date",
    "Coal_Price":        "coal_price",
    "EUA_Price":         "eua_price",
    "Natural_Gas_Price": "natural_gas_price",
    "Oil_Price":         "oil_price",
})
energy["date"] = pd.to_datetime(energy["date"])
energy = energy.set_index("date").sort_index()
# Fill weekends and holidays with the last trading-day price (bfill only
# touches the first few days of the sample)
full_idx = pd.date_range(energy.index.min(), energy.index.max(), freq="D")
energy = energy.reindex(full_idx).ffill().bfill()
print(f"  Energy prices shape: {energy.shape}  NAs after fill: {energy.isna().sum().sum()}", flush=True)

# ---------------------------------------------------------------------------
# Step 6: Macro-finance daily (EuroStoxx50, MSCI EU, USD/EUR exchange, VIX)
# ---------------------------------------------------------------------------
print("Step 6: Macro-finance daily...", flush=True)
macro_daily = pd.read_csv(INTERIM / "MacroFinance/Daily.csv",
                          usecols=["Date", "EuroStoxx50_Price", "MSCI_EU_Price",
                                   "USD_EUR_Exchange_DEXUSEU", "VIX_VIXCLS"])
macro_daily["EuroStoxx50_Price"] = pd.to_numeric(
    macro_daily["EuroStoxx50_Price"].astype(str).str.replace(",", ""), errors="coerce"
)
macro_daily = macro_daily.rename(columns={
    "Date":                     "date",
    "EuroStoxx50_Price":        "eurostoxx50_price",
    "MSCI_EU_Price":            "msci_eu_price",
    "USD_EUR_Exchange_DEXUSEU": "usd_eur_exchange",
    "VIX_VIXCLS":               "vix",
})
macro_daily["date"] = pd.to_datetime(macro_daily["date"])
macro_daily = macro_daily.set_index("date").sort_index()
full_idx_md = pd.date_range(macro_daily.index.min(), macro_daily.index.max(), freq="D")
macro_daily = macro_daily.reindex(full_idx_md).ffill().bfill()
print(f"  Macro-daily shape: {macro_daily.shape}  NAs after fill: {macro_daily.isna().sum().sum()}", flush=True)

# ---------------------------------------------------------------------------
# Step 7: Join all, add DOW dummies, filter to 2018-2024
# ---------------------------------------------------------------------------
print("Step 7: Joining and adding DOW dummies...", flush=True)
base = (price_wide.join(load_wide,   how="left")
                  .join(ren_wide,    how="left")
                  .join(wind_wide,   how="left")
                  .join(solar_wide,  how="left")
                  .join(agg_out_w,   how="left")
                  .join(agg_in_w,    how="left")
                  .join(energy,      how="left")
                  .join(macro_daily, how="left"))
for ch_df in channel_wides.values():
    base = base.join(ch_df, how="left")
base[base.filter(like="flow_").columns] = base.filter(like="flow_").fillna(0)
base.index = pd.to_datetime(base.index)

for dow, name in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
    base[f"dow_{name}"] = (base.index.dayofweek == dow).astype(int)

base = base.sort_index().reset_index().rename(columns={"date": "date"})
base = base[(base["date"] >= "2018-01-01") & (base["date"] < "2025-01-01")]
base = base.reset_index(drop=True)
n_flow_cols = len([c for c in base.columns if c.startswith("flow_")])
n_energy_cols = len([c for c in base.columns if c.endswith("_price") and not c.startswith("eurostoxx") and not c.startswith("msci")])
n_macro_daily_cols = len([c for c in base.columns if c in ["eurostoxx50_price","msci_eu_price","usd_eur_exchange","vix"]])
print(f"  Shape after filter: {base.shape}  (flow: {n_flow_cols}, energy: {n_energy_cols}, macro_daily: {n_macro_daily_cols})", flush=True)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = OUT_DIR / "lear_dataset.csv"
base.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(f"Rows: {len(base)}, Columns: {base.shape[1]}")

# Quick sanity check on one day
print("\nSpot-check 2022-09-01:")
row = base[base["date"] == "2022-09-01"]
if len(row):
    r = row.iloc[0]
    print(f"  price_h00={float(r['price_h00']):.2f}, price_h12={float(r['price_h12']):.2f}")
    print(f"  load_forecast_h12={float(r['load_forecast_h12']):.1f}")
    print(f"  ren_forecast_h12={float(r['ren_forecast_h12']):.1f}")
    dow_cols = [c for c in base.columns if c.startswith("dow_")]
    active_dow = [c for c in dow_cols if r[c] == 1]
    print(f"  DOW: {active_dow}")

print("\nDone.")
