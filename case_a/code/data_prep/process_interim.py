"""
Step 1 of the Case A data pipeline: raw files -> interim files.

Reads data/raw/ and writes data/interim/:
  1. ElectricityPrices/DK1: the yearly files in each subfolder (Loads, Prices,
     Renewables, Transmission) are stacked into one CSV per subfolder. Prices
     are kept for 2018-2024 only, with an added UTC "time" column.
  2. Climate Variables/DK1: the four variable files are merged on (DATE, staid),
     with the station details kept once.
  3. EnergySeries: coal, EUA, gas and oil prices are merged on Date.
  4. MacroFinance: daily, monthly and quarterly series are merged on their date.

Paths are anchored at the case_a folder, worked out from this script's own
location, so it can be run from anywhere.

Next step: build_dataset.py / build_lear_dataset.py.
"""

import csv
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent          # .../case_a
RAW  = ROOT / "data/raw"
OUT  = ROOT / "data/interim"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stream_concat(files: list[Path], out_path: Path, extra_col: dict | None = None):
    """Stack CSV files row by row without loading them into memory.

    extra_col: optional {"column_name": value or function(path)}, appended as
    the last column. Files are passed in filename order, which is already
    chronological, so the output is not re-sorted.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header_written = False
    with out_path.open("w", newline="", encoding="utf-8") as out_f:
        writer = None
        for path in files:
            print(f"    {path.name}")
            extra_val = None
            if extra_col:
                col_name, col_fn = next(iter(extra_col.items()))
                extra_val = col_fn(path) if callable(col_fn) else col_fn

            with path.open(newline="", encoding="utf-8") as in_f:
                reader = csv.reader(in_f)
                file_header = next(reader)

                # Some files were saved with a pandas index (empty first header); drop it
                has_index_col = file_header[0].strip() == ""
                if has_index_col:
                    file_header = file_header[1:]

                if not header_written:
                    fieldnames = file_header + ([col_name] if extra_col else [])
                    writer = csv.writer(out_f)
                    writer.writerow(fieldnames)
                    header_written = True

                for row in reader:
                    if has_index_col:
                        row = row[1:]
                    if extra_col:
                        row = row + [extra_val]
                    writer.writerow(row)


def outer_join(dataframes: list[pd.DataFrame], on: str) -> pd.DataFrame:
    result = dataframes[0]
    for df in dataframes[1:]:
        result = pd.merge(result, df, on=on, how="outer")
    return result


def prefix_cols(df: pd.DataFrame, prefix: str, skip: str) -> pd.DataFrame:
    return df.rename(columns={c: f"{prefix}_{c}" for c in df.columns if c != skip})


def standardize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="mixed", dayfirst=False).dt.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Task 1: ElectricityPrices/DK1 - stream-concatenate each subfolder
# ---------------------------------------------------------------------------

def task1_electricity():
    base = RAW / "ElectricityPrices/DK1"
    out_dir = OUT / "ElectricityPrices/DK1"

    # Loads
    print("  Loads...")
    stream_concat(sorted((base / "Loads").glob("*.csv")), out_dir / "Loads.csv")

    # Prices: stacked, then restricted to 2018-2024 with a "time" column (UTC
    # start of the hour), which data_prep/build_dataset.py reads
    print("  Prices...")
    stream_concat(sorted((base / "Prices").glob("*.csv")), out_dir / "Prices.csv")
    prices = pd.read_csv(out_dir / "Prices.csv", low_memory=False)
    prices["time"] = pd.to_datetime(
        prices["MTU (UTC)"].str.split(" - ").str[0].str.strip(),
        format="mixed", dayfirst=True,
    )
    prices = prices[(prices["time"] >= "2018-01-01") & (prices["time"] < "2025-01-01")]
    prices.to_csv(out_dir / "Prices.csv", index=False)

    # Renewables - add Type column extracted from filename
    print("  Renewables...")
    _type_re = re.compile(r"FORECAST_(OFFSHORE|ONSHORE|SOLAR)_")

    def extract_type(path: Path) -> str:
        m = _type_re.search(path.stem)
        return m.group(1) if m else "UNKNOWN"

    stream_concat(
        sorted((base / "Renewables").glob("*.csv")),
        out_dir / "Renewables.csv",
        extra_col={"Type": extract_type},
    )

    # Transmission
    print("  Transmission...")
    stream_concat(sorted((base / "Transmission").glob("*.csv")), out_dir / "Transmission.csv")


# ---------------------------------------------------------------------------
# Task 2: Climate Variables/DK1 - one row per (DATE, station)
#
# Each raw file holds one variable (fg, rr, ss, tg) for all stations, plus the
# station details. Output columns: DATE, staid, the station details once
# (name, zone, coordinates, elevation), then fg, rr, ss, tg. This is the
# layout data_prep/build_dataset.py expects.
# ---------------------------------------------------------------------------

STATION_COLS = ["station_name", "ZONE_ID", "LAT_DEC", "LON_DEC", "elevation_m"]


def task2_climate():
    base = RAW / "Climate Variables/DK1"
    out_dir = OUT / "Climate Variables"
    out_dir.mkdir(parents=True, exist_ok=True)

    values, stations = [], []
    for f in sorted(base.glob("*.csv")):
        var = f.stem.split("_")[0]   # "fg", "rr", "ss", "tg"
        df = pd.read_csv(f, low_memory=False)
        values.append(df[["DATE", "staid", var]])
        stations.append(df[["staid"] + STATION_COLS])

    result = outer_join(values, on=["DATE", "staid"])
    station_info = pd.concat(stations).drop_duplicates("staid")
    station_info["station_name"] = station_info["station_name"].str.strip()
    result = (result.merge(station_info, on="staid", how="left")
                    [["DATE", "staid"] + STATION_COLS + ["fg", "rr", "ss", "tg"]]
                    .sort_values(["DATE", "staid"]))
    result.to_csv(out_dir / "DK1.csv", index=False)


# ---------------------------------------------------------------------------
# Task 3: EnergySeries - outer join by Date with prefix (full stem)
# ---------------------------------------------------------------------------

def task3_energy():
    base = RAW / "EnergySeries"
    out_dir = OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for f in sorted(base.glob("*.csv")):
        prefix = f.stem          # "Coal", "EUA", "Natural_Gas", "Oil"
        df = pd.read_csv(f, low_memory=False)
        df["Date"] = standardize_date(df["Date"])
        df = prefix_cols(df, prefix, skip="Date")
        dfs.append(df)

    result = outer_join(dfs, on="Date")
    result.to_csv(out_dir / "EnergySeries.csv", index=False)


# ---------------------------------------------------------------------------
# Task 4: MacroFinance - Daily, Monthly, Quarterly
# ---------------------------------------------------------------------------

def task4_macro():
    base = RAW / "MacroFinance"
    out_dir = OUT / "MacroFinance"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Daily: columns prefixed with the file name. Some files call the date
    # column "observation_date", so rename it to "Date" first.
    print("  MacroFinance Daily...")
    dfs = []
    for f in sorted((base / "Daily").glob("*.csv")):
        df = pd.read_csv(f, low_memory=False)
        if "observation_date" in df.columns:
            df = df.rename(columns={"observation_date": "Date"})
        df["Date"] = standardize_date(df["Date"])
        df = prefix_cols(df, f.stem, skip="Date")
        dfs.append(df)
    outer_join(dfs, on="Date").to_csv(out_dir / "Daily.csv", index=False)

    # Monthly - no prefix
    print("  MacroFinance Monthly...")
    dfs = [pd.read_csv(f, low_memory=False) for f in sorted((base / "Monthly").glob("*.csv"))]
    outer_join(dfs, on="observation_date").to_csv(out_dir / "Monthly.csv", index=False)

    # Quarterly - no prefix
    print("  MacroFinance Quarterly...")
    dfs = [pd.read_csv(f, low_memory=False) for f in sorted((base / "Quarterly").glob("*.csv"))]
    outer_join(dfs, on="observation_date").to_csv(out_dir / "Quarterly.csv", index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Task 1: ElectricityPrices/DK1")
    task1_electricity()

    print("Task 2: Climate Variables/DK1")
    task2_climate()

    print("Task 3: EnergySeries")
    task3_energy()

    print("Task 4: MacroFinance")
    task4_macro()

    print("\nDone. Output written to:", OUT)
