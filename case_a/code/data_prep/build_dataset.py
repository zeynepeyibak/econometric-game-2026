"""
Step 2 of the Case A data pipeline: interim files -> one wide hourly dataset.

Starts from the hourly DK1 day-ahead prices (2018-2024, UTC) and left-joins
climate, loads, renewables, cross-border flows, macro-finance and energy
prices onto them. Lower-frequency series are forward-filled.

Input:  data/interim/   (from process_interim.py)
Output: data/prepared/dataset.csv
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


def parse_mtu_cet_to_utc_deduped(series):
    """parse_mtu_cet_to_utc for the transmission file, where each hour appears
    once per border: each distinct MTU string is parsed once and mapped back.

    The transmission MTU column has no time-zone header but is in Danish
    local time too: the spring switchover hour is missing from the file, and
    the repeated autumn hour appears twice, labelled "02:00:00 (CEST)" and
    "02:00:00 (CET)". That label decides which of the two it is.
    """
    unique = series.drop_duplicates()
    raw_start = unique.str.split(" - ").str[0]
    is_summer = ~raw_start.str.contains("(CET)", regex=False)
    start = raw_start.str.replace(r"\s*\(CES?T\)", "", regex=True).str.strip()
    naive = pd.to_datetime(start, format="mixed", dayfirst=True)
    utc = (naive.dt.tz_localize("Europe/Copenhagen",
                                 ambiguous=is_summer.to_numpy(),
                                 nonexistent="shift_forward")
           .dt.tz_convert("UTC").dt.tz_localize(None))
    return series.map(dict(zip(unique.values, utc.values)))


# ---------------------------------------------------------------------------
# Step 1-2: Base - Prices
# ---------------------------------------------------------------------------
print("Step 1-2: Loading Prices...", flush=True)
prices = pd.read_csv(INTERIM / "ElectricityPrices/DK1/Prices.csv", low_memory=False)
prices["time"] = pd.to_datetime(prices["time"])
base = prices[["time", "Day-ahead Price (EUR/MWh)"]].rename(
    columns={"Day-ahead Price (EUR/MWh)": "price"}
).copy()
base["date"] = base["time"].dt.date.astype(str)
print(f"  Base rows: {len(base)}", flush=True)

# ---------------------------------------------------------------------------
# Step 3: Climate Variables - daily, one column per variable x station
# ---------------------------------------------------------------------------
print("Step 3: Climate Variables...", flush=True)
clim = pd.read_csv(INTERIM / "Climate Variables/DK1.csv", low_memory=False)
clim_wide = clim.pivot(index="DATE", columns="staid", values=["fg", "rr", "ss", "tg"])
clim_wide.columns = [f"{var}_{staid}" for var, staid in clim_wide.columns]
clim_wide = clim_wide.reset_index().rename(columns={"DATE": "date"})
clim_wide["date"] = clim_wide["date"].astype(str)
base = base.merge(clim_wide, on="date", how="left")
print(f"  After climate join: {base.shape}", flush=True)

# ---------------------------------------------------------------------------
# Step 4: Loads - CET/CEST -> UTC, join on time
# ---------------------------------------------------------------------------
print("Step 4: Loads...", flush=True)
loads = pd.read_csv(INTERIM / "ElectricityPrices/DK1/Loads.csv", low_memory=False)
loads["time"] = parse_mtu_cet_to_utc(loads["MTU (CET/CEST)"])
loads = loads[["time", "Actual Total Load (MW)", "Day-ahead Total Load Forecast (MW)"]].copy()
base = base.merge(loads, on="time", how="left")
print(f"  After loads join: {base.shape}", flush=True)

# ---------------------------------------------------------------------------
# Step 5: Renewables - CET/CEST -> UTC, pivot by Type
# ---------------------------------------------------------------------------
print("Step 5: Renewables...", flush=True)
ren = pd.read_csv(INTERIM / "ElectricityPrices/DK1/Renewables.csv", low_memory=False)
ren["time"] = parse_mtu_cet_to_utc(ren["MTU (CET/CEST)"])
ren["Day-ahead (MW)"] = pd.to_numeric(ren["Day-ahead (MW)"], errors="coerce")
ren["Actual (MW)"]    = pd.to_numeric(ren["Actual (MW)"],    errors="coerce")

ren_da  = ren.pivot_table(index="time", columns="Type", values="Day-ahead (MW)", aggfunc="first")
ren_act = ren.pivot_table(index="time", columns="Type", values="Actual (MW)",    aggfunc="first")
ren_da.columns  = [f"Renewables_Dayahead_{t}"  for t in ren_da.columns]
ren_act.columns = [f"Renewables_Actual_{t}"    for t in ren_act.columns]
ren_wide = ren_da.join(ren_act).reset_index()
base = base.merge(ren_wide, on="time", how="left")
print(f"  After renewables join: {base.shape}", flush=True)

# ---------------------------------------------------------------------------
# Step 6: Transmission - one column per (from, to) flow, 16 in total
# ---------------------------------------------------------------------------
print("Step 6: Transmission...", flush=True)
trans = pd.read_csv(INTERIM / "ElectricityPrices/DK1/Transmission.csv", low_memory=False)
trans["time"] = parse_mtu_cet_to_utc_deduped(trans["MTU"])
trans["Physical Flow (MW)"] = pd.to_numeric(trans["Physical Flow (MW)"], errors="coerce")
trans["out"] = trans["Out Area"].str.replace("BZN|", "", regex=False)
trans["in_"]  = trans["In Area"].str.replace("BZN|", "", regex=False)
trans["col"]  = "flow_" + trans["out"] + "_to_" + trans["in_"]
trans_wide = trans.pivot_table(index="time", columns="col",
                               values="Physical Flow (MW)", aggfunc="first")
trans_wide.columns.name = None
trans_wide = trans_wide.reset_index()
base = base.merge(trans_wide, on="time", how="left")
print(f"  After transmission join: {base.shape}", flush=True)

# ---------------------------------------------------------------------------
# Step 7: MacroFinance - daily, monthly and quarterly series
# ---------------------------------------------------------------------------
print("Step 7: MacroFinance...", flush=True)

# Daily: no observations on weekends/holidays, so carry the last value forward
macro_daily = pd.read_csv(INTERIM / "MacroFinance/Daily.csv", low_memory=False)
macro_daily["date"] = pd.to_datetime(macro_daily["Date"]).dt.date.astype(str)
macro_daily = macro_daily.drop(columns=["Date"])
base = base.merge(macro_daily, on="date", how="left")
daily_cols = [c for c in macro_daily.columns if c != "date"]
base[daily_cols] = base[daily_cols].ffill()

# Monthly: matched to each hour through its year-month; ffill covers months
# not yet published
macro_monthly = pd.read_csv(INTERIM / "MacroFinance/Monthly.csv", low_memory=False)
monthly_val_cols = [c for c in macro_monthly.columns if c != "observation_date"]
macro_monthly = macro_monthly.rename(
    columns={c: f"Monthly_{c}" for c in monthly_val_cols}
)
macro_monthly["ym"] = pd.to_datetime(macro_monthly["observation_date"]).dt.to_period("M").astype(str)
macro_monthly = macro_monthly.drop(columns=["observation_date"])
base["ym"] = pd.to_datetime(base["date"]).dt.to_period("M").astype(str)
base = base.merge(macro_monthly, on="ym", how="left")
monthly_prefixed = [f"Monthly_{c}" for c in monthly_val_cols]
base[monthly_prefixed] = base[monthly_prefixed].ffill()
base = base.drop(columns=["ym"])

# Quarterly: same logic, matched through year-quarter
macro_quarterly = pd.read_csv(INTERIM / "MacroFinance/Quarterly.csv", low_memory=False)
quarterly_val_cols = [c for c in macro_quarterly.columns if c != "observation_date"]
macro_quarterly = macro_quarterly.rename(
    columns={c: f"Quarterly_{c}" for c in quarterly_val_cols}
)
macro_quarterly["yq"] = pd.to_datetime(macro_quarterly["observation_date"]).dt.to_period("Q").astype(str)
macro_quarterly = macro_quarterly.drop(columns=["observation_date"])
base["yq"] = pd.to_datetime(base["date"]).dt.to_period("Q").astype(str)
base = base.merge(macro_quarterly, on="yq", how="left")
quarterly_prefixed = [f"Quarterly_{c}" for c in quarterly_val_cols]
base[quarterly_prefixed] = base[quarterly_prefixed].ffill()
base = base.drop(columns=["yq"])

print(f"  After MacroFinance joins: {base.shape}", flush=True)

# ---------------------------------------------------------------------------
# Step 8: EnergySeries - prices exist on trading days only, so forward-fill
# ---------------------------------------------------------------------------
print("Step 8: EnergySeries...", flush=True)
energy = pd.read_csv(INTERIM / "EnergySeries.csv", low_memory=False)
energy["date"] = pd.to_datetime(energy["Date"]).dt.date.astype(str)
energy = energy.drop(columns=["Date"])
base = base.merge(energy, on="date", how="left")
energy_cols = [c for c in energy.columns if c != "date"]
base[energy_cols] = base[energy_cols].ffill()
print(f"  After EnergySeries join: {base.shape}", flush=True)

# ---------------------------------------------------------------------------
# Finalise and save
# ---------------------------------------------------------------------------
base = base.drop(columns=["date"])
base = base.sort_values("time").reset_index(drop=True)

# Some flow/renewable series are never reported for DK1; drop them
all_null_cols = [c for c in base.columns if base[c].isna().all()]
print(f"\nDropping {len(all_null_cols)} all-NaN columns.", flush=True)
base = base.drop(columns=all_null_cols)

out_path = OUT_DIR / "dataset.csv"
base.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(f"Final shape: {base.shape}")
print(f"Rows: {len(base)}, Columns: {base.shape[1]}")
print(f"\nSpot-check 2022-09-01 00:00 UTC:")
row = base[base["time"] == "2022-09-01 00:00:00"]
if len(row):
    load = pd.to_numeric(row["Actual Total Load (MW)"].values[0], errors="coerce")
    ren  = pd.to_numeric(row["Renewables_Actual_ONSHORE"].values[0], errors="coerce")
    print(f"  price={row['price'].values[0]:.2f}, load_actual={load:.1f}, ren_onshore={ren:.1f}")
