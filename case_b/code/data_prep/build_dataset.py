"""
Step 2 of the Case B data pipeline: one wide hourly dataset for DK1.

Target: DK1 day-ahead price (column "price"), hourly in UTC, 2018-2024.
Joined onto it:
  - DK1 weather (daily, one column per variable x station)
  - DK1 actual and forecast load, renewables by type, flows per border
  - the same price, load, renewable and flow series for DK2, NO2, SE4, DE
  - macro-finance (daily, monthly, quarterly) and energy prices, forward-filled
  - a holiday dummy from data/additional/holiday_indicator.csv

Time stamps: DK1 and DK2 prices come with a UTC column; every other series
is in Danish local time and is converted to UTC, so that all values on one
row refer to the same UTC hour.

Input:   data/interim/  (from process_interim.py)
Output:  data/prepared/DK1/dataset.csv
"""

from pathlib import Path

import pandas as pd

ROOT       = Path(__file__).parent.parent.parent
INTERIM    = ROOT / "data/interim"
ADDITIONAL = ROOT / "data/additional"

TARGET_ZONE = "DK1"
CROSS_ZONES = ["DK2", "NO2", "SE4", "DE"]

UTC_PRICE_ZONES = {"DK1", "DK2"}  # zones whose price files have an MTU (UTC) column

# ENTSO-E area codes per zone (Germany changed code in Oct 2018)
ZONE_BZN = {
    "DK1": ["BZN|DK1"],
    "DK2": ["BZN|DK2"],
    "NO2": ["BZN|NO2"],
    "SE4": ["BZN|SE4"],
    "DE":  ["BZN|DE-LU", "BZN|DE-AT-LU"],
}


# ---------------------------------------------------------------------------
# Time parsers
# ---------------------------------------------------------------------------

def parse_mtu_cet_to_utc(series: pd.Series) -> pd.Series:
    """Start of an MTU interval in Danish local time, converted to UTC.

    The rows must be in time order: tz_localize infers which of the two
    repeated autumn hours is which from their order.
    """
    starts = (series.str.split(" - ").str[0]
              .str.replace(r"\s*\(CES?T\)", "", regex=True)
              .str.strip())
    t = pd.to_datetime(starts, format="mixed", dayfirst=True)
    t = t.dt.tz_localize("Europe/Copenhagen",
                          ambiguous="infer",
                          nonexistent="shift_forward")
    return t.dt.tz_convert("UTC").dt.tz_localize(None)


def parse_mtu_cet_to_utc_deduped(series: pd.Series) -> pd.Series:
    """parse_mtu_cet_to_utc for the transmission files, where each hour appears
    once per border and rows are not in time order: each distinct MTU string
    is parsed once and mapped back.

    The repeated autumn hour is labelled "(CEST)" / "(CET)" in the file, and
    that label decides which of the two occurrences it is, so no particular
    row order is needed.
    """
    unique = series.drop_duplicates()
    raw_start = unique.str.split(" - ").str[0]
    is_summer = ~raw_start.str.contains("(CET)", regex=False)
    starts = raw_start.str.replace(r"\s*\(CES?T\)", "", regex=True).str.strip()
    naive = pd.to_datetime(starts, format="mixed", dayfirst=True)
    utc = (naive.dt.tz_localize("Europe/Copenhagen",
                                 ambiguous=is_summer.to_numpy(),
                                 nonexistent="shift_forward")
           .dt.tz_convert("UTC").dt.tz_localize(None))
    return series.map(dict(zip(unique.values, utc.values)))


def parse_price_time(df: pd.DataFrame, zone: str) -> pd.Series:
    """UTC start of the delivery hour for each price row (see UTC_PRICE_ZONES)."""
    if zone in UTC_PRICE_ZONES:
        starts = df["MTU (UTC)"].str.split(" - ").str[0].str.strip()
        return pd.to_datetime(starts, format="mixed", dayfirst=True)
    else:
        return parse_mtu_cet_to_utc(df["MTU (CET/CEST)"])


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build():
    out_dir = ROOT / f"data/prepared/{TARGET_ZONE}"
    out_dir.mkdir(parents=True, exist_ok=True)
    elec_dir = INTERIM / f"ElectricityPrices/{TARGET_ZONE}"

    # -----------------------------------------------------------------------
    # Step 1: DK1 prices -> base hourly frame
    # -----------------------------------------------------------------------
    print("Step 1: DK1 prices...", flush=True)
    prices = pd.read_csv(elec_dir / "Prices.csv", low_memory=False)
    prices["time"] = parse_price_time(prices, TARGET_ZONE)
    prices = prices.dropna(subset=["time"])
    prices["price"] = pd.to_numeric(prices["Day-ahead Price (EUR/MWh)"], errors="coerce")
    base = prices[["time", "price"]].copy()
    base["date"] = base["time"].dt.date.astype(str)
    base = base.sort_values("time").reset_index(drop=True)
    print(f"  Base rows: {len(base)}", flush=True)

    # -----------------------------------------------------------------------
    # Step 2: DK1 climate variables -> join on date
    # -----------------------------------------------------------------------
    print("Step 2: DK1 climate...", flush=True)
    clim_path = INTERIM / f"Climate Variables/{TARGET_ZONE}.csv"
    if clim_path.exists():
        clim = pd.read_csv(clim_path, low_memory=False)
        clim_wide = clim.pivot(index="DATE", columns="staid",
                               values=[c for c in clim.columns
                                       if c not in ("DATE", "staid")])
        clim_wide.columns = [f"{var}_{staid}" for var, staid in clim_wide.columns]
        clim_wide = clim_wide.reset_index().rename(columns={"DATE": "date"})
        clim_wide["date"] = clim_wide["date"].astype(str)
        base = base.merge(clim_wide, on="date", how="left")
    print(f"  After climate join: {base.shape}", flush=True)

    # -----------------------------------------------------------------------
    # Step 3: DK1 loads -> join on time
    # -----------------------------------------------------------------------
    print("Step 3: DK1 loads...", flush=True)
    loads = pd.read_csv(elec_dir / "Loads.csv", low_memory=False)
    loads["time"] = parse_mtu_cet_to_utc(loads["MTU (CET/CEST)"])
    loads["Actual Total Load (MW)"]          = pd.to_numeric(
        loads["Actual Total Load (MW)"], errors="coerce")
    loads["Day-ahead Total Load Forecast (MW)"] = pd.to_numeric(
        loads["Day-ahead Total Load Forecast (MW)"], errors="coerce")
    loads = loads[["time", "Actual Total Load (MW)",
                   "Day-ahead Total Load Forecast (MW)"]].copy()
    base = base.merge(loads, on="time", how="left")
    print(f"  After loads join: {base.shape}", flush=True)

    # -----------------------------------------------------------------------
    # Step 4: DK1 renewables -> pivot by Type, join on time
    # -----------------------------------------------------------------------
    print("Step 4: DK1 renewables...", flush=True)
    ren = pd.read_csv(elec_dir / "Renewables.csv", low_memory=False)
    ren["time"] = parse_mtu_cet_to_utc(ren["MTU (CET/CEST)"])
    ren["Day-ahead (MW)"] = pd.to_numeric(ren["Day-ahead (MW)"], errors="coerce")
    ren["Actual (MW)"]    = pd.to_numeric(ren["Actual (MW)"],    errors="coerce")
    ren_da  = ren.pivot_table(index="time", columns="Type",
                               values="Day-ahead (MW)", aggfunc="first")
    ren_act = ren.pivot_table(index="time", columns="Type",
                               values="Actual (MW)",    aggfunc="first")
    ren_da.columns  = [f"Renewables_Dayahead_{t}" for t in ren_da.columns]
    ren_act.columns = [f"Renewables_Actual_{t}"   for t in ren_act.columns]
    ren_wide = ren_da.join(ren_act).reset_index()
    base = base.merge(ren_wide, on="time", how="left")
    print(f"  After renewables join: {base.shape}", flush=True)

    # -----------------------------------------------------------------------
    # Step 5: DK1 transmission -> per-channel physical flows
    # -----------------------------------------------------------------------
    print("Step 5: DK1 transmission...", flush=True)
    trans = pd.read_csv(elec_dir / "Transmission.csv", low_memory=False)
    trans["time"] = parse_mtu_cet_to_utc_deduped(trans["MTU"])
    trans["Physical Flow (MW)"] = pd.to_numeric(trans["Physical Flow (MW)"], errors="coerce")
    # Germany is DE-AT-LU before the October 2018 split and DE-LU after; use one name
    for col in ("Out Area", "In Area"):
        trans[col] = trans[col].str.replace("BZN|DE-AT-LU", "BZN|DE-LU", regex=False)
    trans["out"] = trans["Out Area"].str.replace("BZN|", "", regex=False)
    trans["in_"] = trans["In Area"].str.replace("BZN|", "", regex=False)
    trans["col"] = "flow_dk1_" + trans["out"] + "_to_" + trans["in_"]
    trans_wide = trans.pivot_table(index="time", columns="col",
                                    values="Physical Flow (MW)", aggfunc="first")
    trans_wide.columns.name = None
    trans_wide = trans_wide.reset_index()
    base = base.merge(trans_wide, on="time", how="left")
    print(f"  After transmission join: {base.shape}", flush=True)

    # -----------------------------------------------------------------------
    # Step 6: Cross-zone data (DK2, NO2, SE4, DE)
    # -----------------------------------------------------------------------
    print("Step 6: Cross-zone data...", flush=True)
    for z in CROSS_ZONES:
        z_dir = INTERIM / f"ElectricityPrices/{z}"
        z_lc  = z.lower()

        # Prices
        p = pd.read_csv(z_dir / "Prices.csv", low_memory=False)
        p["time"] = parse_price_time(p, z)
        p = p.dropna(subset=["time"])
        p[f"price_{z_lc}"] = pd.to_numeric(p["Day-ahead Price (EUR/MWh)"], errors="coerce")
        base = base.merge(p[["time", f"price_{z_lc}"]], on="time", how="left")

        # Loads (actual + forecast)
        ld = pd.read_csv(z_dir / "Loads.csv", low_memory=False)
        ld["time"] = parse_mtu_cet_to_utc(ld["MTU (CET/CEST)"])
        ld[f"actual_load_{z_lc}"]   = pd.to_numeric(ld["Actual Total Load (MW)"],          errors="coerce")
        ld[f"forecast_load_{z_lc}"] = pd.to_numeric(ld["Day-ahead Total Load Forecast (MW)"], errors="coerce")
        base = base.merge(ld[["time", f"actual_load_{z_lc}",
                               f"forecast_load_{z_lc}"]], on="time", how="left")

        # Renewables (day-ahead + actual, per type)
        rn = pd.read_csv(z_dir / "Renewables.csv", low_memory=False)
        rn["time"] = parse_mtu_cet_to_utc(rn["MTU (CET/CEST)"])
        rn["Day-ahead (MW)"] = pd.to_numeric(rn["Day-ahead (MW)"], errors="coerce")
        rn["Actual (MW)"]    = pd.to_numeric(rn["Actual (MW)"],    errors="coerce")
        rn_da2  = rn.pivot_table(index="time", columns="Type",
                                  values="Day-ahead (MW)", aggfunc="first")
        rn_act2 = rn.pivot_table(index="time", columns="Type",
                                  values="Actual (MW)",    aggfunc="first")
        rn_da2.columns  = [f"ren_dayahead_{z_lc}_{t}"  for t in rn_da2.columns]
        rn_act2.columns = [f"ren_actual_{z_lc}_{t}"    for t in rn_act2.columns]
        rn_wide2 = rn_da2.join(rn_act2).reset_index()
        base = base.merge(rn_wide2, on="time", how="left")

        # Transmission: every border of this zone, as flow_<zone>_<from>_to_<to>.
        # Borders with DK1 therefore appear twice (here and in step 5).
        tr = pd.read_csv(z_dir / "Transmission.csv", low_memory=False)
        tr["time"] = parse_mtu_cet_to_utc_deduped(tr["MTU"])
        tr["Physical Flow (MW)"] = pd.to_numeric(tr["Physical Flow (MW)"], errors="coerce")
        for col in ("Out Area", "In Area"):
            tr[col] = tr[col].str.replace("BZN|DE-AT-LU", "BZN|DE-LU", regex=False)
        tr["out_lc"] = tr["Out Area"].str.replace("BZN|", "", regex=False).str.replace("-", "_").str.lower()
        tr["in_lc"]  = tr["In Area"].str.replace("BZN|", "", regex=False).str.replace("-", "_").str.lower()
        tr["col"] = f"flow_{z_lc}_" + tr["out_lc"] + "_to_" + tr["in_lc"]
        tr_wide = tr.pivot_table(index="time", columns="col",
                                  values="Physical Flow (MW)", aggfunc="first")
        tr_wide.columns.name = None
        tr_wide = tr_wide.reset_index()
        base = base.merge(tr_wide, on="time", how="left")

        print(f"  After {z} join: {base.shape}", flush=True)

    # -----------------------------------------------------------------------
    # Step 7: MacroFinance Daily / Monthly / Quarterly
    # -----------------------------------------------------------------------
    print("Step 7: MacroFinance...", flush=True)
    macro_dir = INTERIM / "MacroFinance"

    # Daily: carry the last value over weekends and holidays
    macro_daily = pd.read_csv(macro_dir / "Daily.csv", low_memory=False)
    macro_daily["date"] = pd.to_datetime(macro_daily["Date"]).dt.date.astype(str)
    macro_daily = macro_daily.drop(columns=["Date"])
    # EuroStoxx50 is stored with thousands separators ("3,456.7")
    if "EuroStoxx50_Price" in macro_daily.columns:
        macro_daily["EuroStoxx50_Price"] = pd.to_numeric(
            macro_daily["EuroStoxx50_Price"].astype(str).str.replace(",", ""),
            errors="coerce"
        )
    base = base.merge(macro_daily, on="date", how="left")
    daily_cols = [c for c in macro_daily.columns if c != "date"]
    base[daily_cols] = base[daily_cols].ffill()

    # Monthly: matched through year-month, ffill for months not yet published
    macro_monthly = pd.read_csv(macro_dir / "Monthly.csv", low_memory=False)
    monthly_val_cols = [c for c in macro_monthly.columns if c != "observation_date"]
    macro_monthly = macro_monthly.rename(
        columns={c: f"Monthly_{c}" for c in monthly_val_cols}
    )
    macro_monthly["ym"] = (pd.to_datetime(macro_monthly["observation_date"])
                           .dt.to_period("M").astype(str))
    macro_monthly = macro_monthly.drop(columns=["observation_date"])
    base["ym"] = pd.to_datetime(base["date"]).dt.to_period("M").astype(str)
    base = base.merge(macro_monthly, on="ym", how="left")
    monthly_prefixed = [f"Monthly_{c}" for c in monthly_val_cols]
    base[monthly_prefixed] = base[monthly_prefixed].ffill()
    base = base.drop(columns=["ym"])

    # Quarterly: same, through year-quarter
    macro_quarterly = pd.read_csv(macro_dir / "Quarterly.csv", low_memory=False)
    quarterly_val_cols = [c for c in macro_quarterly.columns if c != "observation_date"]
    macro_quarterly = macro_quarterly.rename(
        columns={c: f"Quarterly_{c}" for c in quarterly_val_cols}
    )
    macro_quarterly["yq"] = (pd.to_datetime(macro_quarterly["observation_date"])
                              .dt.to_period("Q").astype(str))
    macro_quarterly = macro_quarterly.drop(columns=["observation_date"])
    base["yq"] = pd.to_datetime(base["date"]).dt.to_period("Q").astype(str)
    base = base.merge(macro_quarterly, on="yq", how="left")
    quarterly_prefixed = [f"Quarterly_{c}" for c in quarterly_val_cols]
    base[quarterly_prefixed] = base[quarterly_prefixed].ffill()
    base = base.drop(columns=["yq"])

    print(f"  After MacroFinance joins: {base.shape}", flush=True)

    # -----------------------------------------------------------------------
    # Step 8: EnergySeries (daily, ffill)
    # -----------------------------------------------------------------------
    print("Step 8: EnergySeries...", flush=True)
    energy = pd.read_csv(INTERIM / "EnergySeries.csv", low_memory=False)
    energy["date"] = pd.to_datetime(energy["Date"]).dt.date.astype(str)
    energy = energy.drop(columns=["Date"])
    base = base.merge(energy, on="date", how="left")
    energy_cols = [c for c in energy.columns if c != "date"]
    base[energy_cols] = base[energy_cols].ffill()
    print(f"  After EnergySeries join: {base.shape}", flush=True)

    # -----------------------------------------------------------------------
    # Step 9: Holiday dummy. The file lists (month, day) pairs, so the same
    # dates count as holidays every year; movable holidays like Easter are not
    # captured unless they are in the file.
    # -----------------------------------------------------------------------
    print("Step 9: Holiday indicator...", flush=True)
    hol = pd.read_csv(ADDITIONAL / "holiday_indicator.csv")
    hol_map = {(int(r["month"]), int(r["day"])): int(r["is_holiday"])
               for _, r in hol.iterrows()}
    base["is_holiday"] = base["time"].apply(
        lambda t: hol_map.get((t.month, t.day), 0)
    )
    print(f"  Holiday days in dataset: {base['is_holiday'].sum()}", flush=True)

    # -----------------------------------------------------------------------
    # Keep 2018-2024, drop columns that are entirely empty, save
    # -----------------------------------------------------------------------
    base = base.drop(columns=["date"])
    base = base.sort_values("time").reset_index(drop=True)
    base = base[(base["time"] >= "2018-01-01") & (base["time"] < "2025-01-01")]

    all_null_cols = [c for c in base.columns if base[c].isna().all()]
    print(f"\nDropping {len(all_null_cols)} all-NaN columns.", flush=True)
    base = base.drop(columns=all_null_cols)
    base = base.reset_index(drop=True)

    out_path = out_dir / "dataset.csv"
    base.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"Final shape: {base.shape}")
    print(f"Rows: {len(base)}, Columns: {base.shape[1]}")

    print(f"\nSpot-check 2022-09-01 00:00 UTC:")
    row = base[base["time"] == "2022-09-01 00:00:00"]
    if len(row):
        r = row.iloc[0]
        load_actual = pd.to_numeric(r.get("Actual Total Load (MW)", float("nan")),
                                    errors="coerce")
        print(f"  price={float(r['price']):.2f}, load_actual={load_actual:.1f}")
        for z in CROSS_ZONES:
            col = f"price_{z.lower()}"
            if col in r.index and pd.notna(r[col]):
                print(f"  {col}={float(r[col]):.2f}")


if __name__ == "__main__":
    build()
