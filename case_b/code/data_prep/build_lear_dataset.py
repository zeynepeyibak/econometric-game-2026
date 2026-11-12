"""
Builds the daily model input for Case B (LEAR and LightGBM).

One row per UTC date, 2018-2024. Hourly series are spread over 24 columns
(_h00 ... _h23). <z> is one of dk2, no2, se4, de.

  price_h..                      DK1 day-ahead price (the target)
  price_<z>_h..                  prices in the neighbouring zones
  load_forecast[_<z>]_h..        day-ahead load forecast
  ren / wind / solar / offshore / onshore _forecast[_<z>]_h..
                                 day-ahead renewable forecasts by type
  flow_dk1_<partner>_h..         DK1 net flow per border (out - in, > 0 = export)
  flow_dk1_out_h.., flow_dk1_in_h..   DK1 total outflow and inflow
  flow_<z>_<partner>_h..         net flows of the other zones, per border
  coal/eua/natural_gas/oil_price, eurostoxx50_price, msci_eu_price,
  usd_eur_exchange, vix          daily, forward-filled over non-trading days
  temp_dk1                       DK1 mean temperature over stations (0.1 C)
  precip_no2_roll14              NO2 precipitation, 14-day trailing mean
                                 (proxy for Norwegian hydro inflow)
  dow_Mon ... dow_Sun, is_holiday

Time stamps: DK1 and DK2 prices have a UTC column; everything else is in
Danish local time and converted to UTC, so price_hH and load_forecast_hH on
the same row refer to the same UTC hour. Lags and the asinh transform are
applied later, in models/lear/core.py.

Input:   data/interim/  (from process_interim.py)
         data/additional/holiday_indicator.csv
Output:  data/prepared/DK1/lear_dataset.csv
"""

from pathlib import Path

import pandas as pd

ROOT       = Path(__file__).parent.parent.parent
INTERIM    = ROOT / "data/interim"
ADDITIONAL = ROOT / "data/additional"

TARGET_ZONE = "DK1"
CROSS_ZONES = ["DK2", "NO2", "SE4", "DE"]
ALL_ZONES   = [TARGET_ZONE] + CROSS_ZONES

# ENTSO-E area codes per zone. Germany is DE-AT-LU before the October 2018
# split and DE-LU after; both are treated as one zone.
ZONE_BZN = {
    "DK1": ["BZN|DK1"],
    "DK2": ["BZN|DK2"],
    "NO2": ["BZN|NO2"],
    "SE4": ["BZN|SE4"],
    "DE":  ["BZN|DE-LU", "BZN|DE-AT-LU"],
}

# Zones whose price files have an MTU (UTC) column
UTC_PRICE_ZONES = {"DK1", "DK2"}


# ---------------------------------------------------------------------------
# Time parsers
# ---------------------------------------------------------------------------

def _strip_start(mtu_series: pd.Series) -> pd.Series:
    """"dd/mm/yyyy hh:mm - ..." -> start of the interval, without "(CET)"/"(CEST)"."""
    return (mtu_series.str.split(" - ").str[0]
            .str.replace(r"\s*\(CES?T\)", "", regex=True)
            .str.strip())


def parse_mtu_cet_to_utc(series: pd.Series) -> pd.Series:
    """Start of an MTU interval in Danish local time, converted to UTC.

    The rows must be in time order: tz_localize infers which of the two
    repeated autumn hours is which from their order.
    """
    t = pd.to_datetime(_strip_start(series), format="mixed", dayfirst=True)
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
    is_summer = ~unique.str.split(" - ").str[0].str.contains("(CET)", regex=False)
    naive = pd.to_datetime(_strip_start(unique), format="mixed", dayfirst=True)
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
# Pivot helpers
# ---------------------------------------------------------------------------

def prices_to_wide(zone: str, prefix: str) -> pd.DataFrame:
    # Day-ahead prices of one zone as date x hour
    df = pd.read_csv(INTERIM / f"ElectricityPrices/{zone}/Prices.csv", low_memory=False)
    df["time"] = parse_price_time(df, zone)
    df["val"]  = pd.to_numeric(df["Day-ahead Price (EUR/MWh)"], errors="coerce")
    df = df.dropna(subset=["time"])
    df["date"] = df["time"].dt.date
    df["hour"] = df["time"].dt.hour
    wide = df.pivot_table(index="date", columns="hour", values="val", aggfunc="first")
    wide.columns = [f"{prefix}_h{h:02d}" for h in wide.columns]
    return wide


def load_to_wide(zone: str, prefix: str) -> pd.DataFrame:
    # Day-ahead load forecast of one zone as date x hour
    df = pd.read_csv(INTERIM / f"ElectricityPrices/{zone}/Loads.csv", low_memory=False)
    df["time"] = parse_mtu_cet_to_utc(df["MTU (CET/CEST)"])
    df["val"]  = pd.to_numeric(df["Day-ahead Total Load Forecast (MW)"], errors="coerce")
    df["date"] = df["time"].dt.date
    df["hour"] = df["time"].dt.hour
    wide = df.pivot_table(index="date", columns="hour", values="val", aggfunc="first")
    wide.columns = [f"{prefix}_h{h:02d}" for h in wide.columns]
    return wide


def renewables_to_wides(zone: str,
                         ren_prefix: str,
                         wind_prefix: str,
                         solar_prefix: str,
                         offshore_prefix: str | None = None,
                         onshore_prefix: str | None = None) -> tuple:
    """Day-ahead renewable forecasts of one zone as date x hour tables:
    total, wind (on + offshore), solar and, if prefixes are given, offshore
    and onshore separately."""
    df = pd.read_csv(INTERIM / f"ElectricityPrices/{zone}/Renewables.csv", low_memory=False)
    df["time"] = parse_mtu_cet_to_utc(df["MTU (CET/CEST)"])
    df["val"]  = pd.to_numeric(df["Day-ahead (MW)"], errors="coerce")

    def _agg_wide(mask, prefix):
        sub = df[mask] if mask is not None else df
        s = sub.groupby("time")["val"].sum().reset_index()
        s["date"] = s["time"].dt.date
        s["hour"] = s["time"].dt.hour
        wide = s.pivot_table(index="date", columns="hour", values="val", aggfunc="first")
        wide.columns = [f"{prefix}_h{h:02d}" for h in wide.columns]
        return wide

    ren      = _agg_wide(None, ren_prefix)
    wind     = _agg_wide(df["Type"].isin(["OFFSHORE", "ONSHORE"]), wind_prefix)
    solar    = _agg_wide(df["Type"] == "SOLAR", solar_prefix)

    if offshore_prefix is not None and onshore_prefix is not None:
        offshore = _agg_wide(df["Type"] == "OFFSHORE", offshore_prefix)
        onshore  = _agg_wide(df["Type"] == "ONSHORE",  onshore_prefix)
        return ren, wind, solar, offshore, onshore

    return ren, wind, solar


def _partner_label(bzn_str: str) -> str:
    """"BZN|DE-LU" -> "de_lu", for column names."""
    return bzn_str.replace("BZN|", "").replace("-", "_").lower()


def transmission_net_flows(zone: str,
                            zone_label: str,
                            include_aggregate: bool = False) -> list[pd.DataFrame]:
    """Net physical flow (out - in) between a zone and each of its neighbours.

    Neighbours are read from the file, not fixed in advance. Returns one
    date x hour table per neighbour (flow_<zone>_<partner>_h..) and, with
    include_aggregate, total out- and inflow (flow_<zone>_out/in_h..).
    Hours without data count as zero flow.
    """
    trans = pd.read_csv(INTERIM / f"ElectricityPrices/{zone}/Transmission.csv",
                         low_memory=False)
    trans["time"] = parse_mtu_cet_to_utc_deduped(trans["MTU"])
    trans["Physical Flow (MW)"] = pd.to_numeric(trans["Physical Flow (MW)"], errors="coerce")

    # Use one name for Germany before and after the 2018 split
    trans["Out Area"] = trans["Out Area"].str.replace("BZN|DE-AT-LU", "BZN|DE-LU",
                                                       regex=False)
    trans["In Area"]  = trans["In Area"].str.replace("BZN|DE-AT-LU", "BZN|DE-LU",
                                                      regex=False)

    bz_ids = [a.replace("BZN|DE-AT-LU", "BZN|DE-LU") for a in ZONE_BZN[zone]]
    zone_out = trans[trans["Out Area"].isin(bz_ids)].copy()
    zone_in  = trans[trans["In Area"].isin(bz_ids)].copy()

    # Every area this zone exchanges power with
    out_partners = set(zone_out["In Area"].unique())
    in_partners  = set(zone_in["Out Area"].unique())
    all_partners = sorted(out_partners | in_partners)
    # The raw files sometimes list the zone as its own partner
    all_partners = [p for p in all_partners if p not in bz_ids]

    result_frames = []

    def _hourly_sum(df, partner_col, partner):
        # Hourly flow to/from one partner on a complete hourly grid (gaps = 0)
        sub = df[df[partner_col] == partner].copy()
        h = sub.groupby("time")["Physical Flow (MW)"].sum()
        if h.empty:
            return pd.Series(dtype=float)
        h = h.reindex(pd.date_range(h.index.min(), h.index.max(), freq="h"),
                      fill_value=0)
        return h

    for partner in all_partners:
        p_label = _partner_label(partner)
        col_prefix = f"flow_{zone_label}_{p_label}"

        out_h = _hourly_sum(zone_out, "In Area",  partner)
        in_h  = _hourly_sum(zone_in,  "Out Area", partner)

        if out_h.empty and in_h.empty:
            continue

        # Put both directions on the same hours before subtracting
        idx = out_h.index.union(in_h.index) if not (out_h.empty or in_h.empty) else (
              out_h.index if not out_h.empty else in_h.index)
        out_h = out_h.reindex(idx, fill_value=0)
        in_h  = in_h.reindex(idx, fill_value=0)
        net = (out_h - in_h).rename("flow").reset_index()
        net.columns = ["time", "flow"]
        net["date"] = net["time"].dt.date
        net["hour"] = net["time"].dt.hour
        wide = net.pivot_table(index="date", columns="hour", values="flow", aggfunc="first")
        wide.columns = [f"{col_prefix}_h{h:02d}" for h in wide.columns]
        result_frames.append(wide.fillna(0))

    if include_aggregate:
        # Totals over all neighbours
        for direction, df, partner_col, agg_label in [
            ("out", zone_out, "In Area",  f"flow_{zone_label}_out"),
            ("in",  zone_in,  "Out Area", f"flow_{zone_label}_in"),
        ]:
            h = df[df[partner_col].isin(all_partners)].groupby("time")["Physical Flow (MW)"].sum()
            if h.empty:
                continue
            h = h.reindex(pd.date_range(h.index.min(), h.index.max(), freq="h"),
                          fill_value=0).reset_index()
            h.columns = ["time", "flow"]
            h["date"] = h["time"].dt.date
            h["hour"] = h["time"].dt.hour
            wide = h.pivot_table(index="date", columns="hour", values="flow", aggfunc="first")
            wide.columns = [f"{agg_label}_h{h:02d}" for h in wide.columns]
            result_frames.append(wide.fillna(0))

    return result_frames


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build():
    out_dir = ROOT / f"data/prepared/{TARGET_ZONE}"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []

    # -----------------------------------------------------------------------
    # Prices
    # -----------------------------------------------------------------------
    print("Prices...", flush=True)
    w = prices_to_wide(TARGET_ZONE, "price")
    print(f"  DK1: {w.shape}")
    frames.append(w)

    for z in CROSS_ZONES:
        w = prices_to_wide(z, f"price_{z.lower()}")
        print(f"  {z}: {w.shape}")
        frames.append(w)

    # -----------------------------------------------------------------------
    # Load forecasts
    # -----------------------------------------------------------------------
    print("Load forecasts...", flush=True)
    w = load_to_wide(TARGET_ZONE, "load_forecast")
    print(f"  DK1: {w.shape}")
    frames.append(w)

    for z in CROSS_ZONES:
        w = load_to_wide(z, f"load_forecast_{z.lower()}")
        print(f"  {z}: {w.shape}")
        frames.append(w)

    # -----------------------------------------------------------------------
    # Renewable forecasts (ren / wind / solar)
    # -----------------------------------------------------------------------
    print("Renewable forecasts...", flush=True)
    rw, ww, sw, ofw, onw = renewables_to_wides(
        TARGET_ZONE,
        "ren_forecast", "wind_forecast", "solar_forecast",
        "offshore_forecast", "onshore_forecast",
    )
    print(f"  DK1 ren/wind/solar/offshore/onshore: {rw.shape}")
    frames += [rw, ww, sw, ofw, onw]

    for z in CROSS_ZONES:
        rw, ww, sw, ofw, onw = renewables_to_wides(
            z,
            f"ren_forecast_{z.lower()}",
            f"wind_forecast_{z.lower()}",
            f"solar_forecast_{z.lower()}",
            f"offshore_forecast_{z.lower()}",
            f"onshore_forecast_{z.lower()}",
        )
        print(f"  {z} ren/wind/solar/offshore/onshore: {rw.shape}")
        frames += [rw, ww, sw, ofw, onw]

    # -----------------------------------------------------------------------
    # Transmission flows
    # -----------------------------------------------------------------------
    print("Transmission flows...", flush=True)
    # DK1: per border and in total
    dk1_flow_frames = transmission_net_flows(TARGET_ZONE, "dk1", include_aggregate=True)
    n_ch = sum(1 for f in dk1_flow_frames
               if not any(c.endswith("_out_h00") or c.endswith("_in_h00")
                          for c in f.columns))
    print(f"  DK1: {len(dk1_flow_frames)} frames ({n_ch} channels + agg in/out)")
    frames += dk1_flow_frames

    # Other zones: per border only
    for z in CROSS_ZONES:
        z_frames = transmission_net_flows(z, z.lower(), include_aggregate=False)
        print(f"  {z}: {len(z_frames)} channel frames")
        frames += z_frames

    # -----------------------------------------------------------------------
    # Fuel / carbon prices (daily scalars)
    # -----------------------------------------------------------------------
    print("Fuel/carbon prices...", flush=True)
    energy = pd.read_csv(INTERIM / "EnergySeries.csv",
                         usecols=["Date", "Coal_Price", "EUA_Price",
                                  "Natural_Gas_Price", "Oil_Price"],
                         low_memory=False)
    energy = energy.rename(columns={
        "Date": "date",
        "Coal_Price":        "coal_price",
        "EUA_Price":         "eua_price",
        "Natural_Gas_Price": "natural_gas_price",
        "Oil_Price":         "oil_price",
    })
    energy["date"] = pd.to_datetime(energy["date"])
    energy = energy.set_index("date").sort_index()
    energy = energy.reindex(
        pd.date_range(energy.index.min(), energy.index.max(), freq="D")
    ).ffill().bfill()
    print(f"  Energy prices: {energy.shape}")
    frames.append(energy)

    # -----------------------------------------------------------------------
    # Macro-finance daily scalars
    # -----------------------------------------------------------------------
    print("Macro-finance daily...", flush=True)
    macro = pd.read_csv(INTERIM / "MacroFinance/Daily.csv",
                        usecols=["Date", "EuroStoxx50_Price", "MSCI_EU_Price",
                                 "USD_EUR_Exchange_DEXUSEU", "VIX_VIXCLS"],
                        low_memory=False)
    macro = macro.rename(columns={
        "Date":                    "date",
        "EuroStoxx50_Price":       "eurostoxx50_price",
        "MSCI_EU_Price":           "msci_eu_price",
        "USD_EUR_Exchange_DEXUSEU": "usd_eur_exchange",
        "VIX_VIXCLS":              "vix",
    })
    macro["eurostoxx50_price"] = pd.to_numeric(
        macro["eurostoxx50_price"].astype(str).str.replace(",", ""), errors="coerce"
    )
    macro["date"] = pd.to_datetime(macro["date"])
    macro = macro.set_index("date").sort_index()
    macro = macro.reindex(
        pd.date_range(macro.index.min(), macro.index.max(), freq="D")
    ).ffill().bfill()
    print(f"  Macro daily: {macro.shape}")
    frames.append(macro)

    # -----------------------------------------------------------------------
    # Climate variables (daily scalars)
    # -----------------------------------------------------------------------
    print("Climate variables...", flush=True)

    # DK1 mean temperature over all stations, in 0.1 C (18 C is used later as the
    # heating/cooling threshold)
    clim_dk1 = pd.read_csv(INTERIM / "Climate Variables/DK1.csv",
                            usecols=["DATE", "tg_tg"], low_memory=False)
    clim_dk1["DATE"] = pd.to_datetime(clim_dk1["DATE"])
    clim_dk1["tg_tg"] = pd.to_numeric(clim_dk1["tg_tg"], errors="coerce")
    temp_dk1 = (clim_dk1.groupby("DATE")["tg_tg"]
                .mean()
                .rename("temp_dk1")
                .to_frame())
    temp_dk1 = temp_dk1.reindex(
        pd.date_range(temp_dk1.index.min(), temp_dk1.index.max(), freq="D")
    ).ffill().bfill()
    frames.append(temp_dk1)
    print(f"  DK1 temperature: {temp_dk1.shape}")

    # NO2 precipitation (0.1 mm), averaged over stations, then a 14-day trailing
    # mean as a rough measure of hydro inflow
    clim_no2 = pd.read_csv(INTERIM / "Climate Variables/NO2.csv",
                            usecols=["DATE", "rr_rr"], low_memory=False)
    clim_no2["DATE"] = pd.to_datetime(clim_no2["DATE"])
    clim_no2["rr_rr"] = pd.to_numeric(clim_no2["rr_rr"], errors="coerce")
    precip_no2 = (clim_no2.groupby("DATE")["rr_rr"]
                  .mean()
                  .rename("precip_no2_roll14")
                  .to_frame())
    precip_no2 = precip_no2.reindex(
        pd.date_range(precip_no2.index.min(), precip_no2.index.max(), freq="D")
    ).ffill().bfill()
    precip_no2["precip_no2_roll14"] = (precip_no2["precip_no2_roll14"]
                                        .rolling(14, min_periods=1).mean())
    frames.append(precip_no2)
    print(f"  NO2 precipitation (14-day roll): {precip_no2.shape}")

    # -----------------------------------------------------------------------
    # Holiday dummy. The file lists (month, day) pairs, so the same dates count
    # every year; movable holidays are only covered if they are in the file.
    # -----------------------------------------------------------------------
    print("Holiday indicator...", flush=True)
    hol = pd.read_csv(ADDITIONAL / "holiday_indicator.csv")
    hol_map = {(int(r["month"]), int(r["day"])): int(r["is_holiday"])
               for _, r in hol.iterrows()}

    # -----------------------------------------------------------------------
    # Join everything on date
    # -----------------------------------------------------------------------
    print("Joining all wide frames...", flush=True)
    base = frames[0].copy()
    for f in frames[1:]:
        if f is not None and not f.empty:
            base = base.join(f, how="left")

    # Missing flows are treated as zero
    flow_cols = [c for c in base.columns if "flow_" in c]
    base[flow_cols] = base[flow_cols].fillna(0)

    # Fill single missing hours (mostly around DST changes, e.g. DE on
    # 2020-10-25) by interpolating along the row, separately within each
    # variable's own 24-hour block, so a gap at h00 or h23 is never filled
    # from a column that belongs to a different variable.
    import re as _re
    blocks: dict = {}
    for c in base.columns:
        m = _re.match(r"(.+)_h\d{2}$", c)
        if m and not c.startswith("flow_"):
            blocks.setdefault(m.group(1), []).append(c)
    for cols in blocks.values():
        cols = sorted(cols)
        base[cols] = base[cols].interpolate(axis=1, limit_direction="both")

    base.index = pd.to_datetime(base.index)

    # Day-of-week and holiday dummies (added in one concat; adding columns one
    # by one to a frame this wide triggers pandas performance warnings)
    dow_data = {
        f"dow_{name}": (base.index.dayofweek == dow).astype(int)
        for dow, name in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    }
    dow_data["is_holiday"] = pd.array([
        hol_map.get((d.month, d.day), 0) for d in base.index
    ])
    base = pd.concat([base, pd.DataFrame(dow_data, index=base.index)], axis=1)

    # Keep 2018-2024
    base = base.sort_index().reset_index().rename(columns={"index": "date"})
    base = base[(base["date"] >= "2018-01-01") & (base["date"] < "2025-01-01")]
    base = base.reset_index(drop=True)

    # -----------------------------------------------------------------------
    # Column counts, as a check
    # -----------------------------------------------------------------------
    n_price     = len([c for c in base.columns if c.startswith("price_h")])
    n_xprice    = len([c for c in base.columns
                       if any(c.startswith(f"price_{z.lower()}_h") for z in CROSS_ZONES)])
    n_load      = len([c for c in base.columns if "load_forecast" in c and "_h" in c])
    n_ren       = len([c for c in base.columns if "ren_forecast" in c or
                       "wind_forecast" in c or "solar_forecast" in c or
                       "offshore_forecast" in c or "onshore_forecast" in c])
    n_flow_dk1  = len([c for c in base.columns if c.startswith("flow_dk1")])
    n_flow_other= len([c for c in base.columns if "flow_" in c
                       and not c.startswith("flow_dk1")])
    print(f"\n  Shape: {base.shape}")
    print(f"  DK1 price cols      : {n_price}")
    print(f"  Cross-zone price    : {n_xprice}")
    print(f"  Load forecast cols  : {n_load}")
    print(f"  Renewable cols      : {n_ren}")
    print(f"  DK1 flow cols       : {n_flow_dk1}")
    print(f"  Other zone flow cols: {n_flow_other}")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    out_path = out_dir / "lear_dataset.csv"
    base.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"Rows: {len(base)}, Columns: {base.shape[1]}")

    # Quick sanity check on one day
    print("\nSpot-check 2022-09-01:")
    row = base[base["date"] == "2022-09-01"]
    if len(row):
        r = row.iloc[0]
        print(f"  price_h00={float(r['price_h00']):.2f}, "
              f"price_h12={float(r['price_h12']):.2f}")
        print(f"  load_forecast_h12={float(r['load_forecast_h12']):.1f}")
        for z in CROSS_ZONES:
            col = f"price_{z.lower()}_h12"
            if col in r.index and pd.notna(r[col]):
                print(f"  {col}={float(r[col]):.2f}")
        print(f"  DOW: {[c for c in base.columns if c.startswith('dow_') and r[c]==1]}")

    print("\nDone.")


if __name__ == "__main__":
    build()
