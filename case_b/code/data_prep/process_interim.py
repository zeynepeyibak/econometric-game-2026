"""
Step 1 of the Case B data pipeline: raw files -> interim files.

Same steps as in Case A, but for five bidding zones (DK1, DK2, NO2, SE4, DE):
  1. ElectricityPrices/<zone>: yearly files in each subfolder (Loads, Prices,
     Renewables, Transmission) stacked into one CSV per subfolder.
  2. Climate Variables/<zone>: the gzipped variable files (fg, rr, ss, tg and,
     where available, cc) merged on (DATE, staid).
  3. EnergySeries: coal, EUA, gas and oil prices merged on Date.
  4. MacroFinance: daily, monthly and quarterly series merged on their date.

Input:   data/raw/
Output:  data/interim/
Next:    build_dataset.py / build_lear_dataset.py
"""

import csv
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
RAW  = ROOT / "data/raw"
OUT  = ROOT / "data/interim"

ZONES = ["DK1", "DK2", "NO2", "SE4", "DE"]

# ---------------------------------------------------------------------------
# Helpers (as in Case A, except stream_concat also drops double index columns)
# ---------------------------------------------------------------------------

def stream_concat(files: list[Path], out_path: Path, extra_col: dict | None = None):
    """Stack CSV files row by row without loading them into memory.

    extra_col: optional {"column_name": value or function(path)}, appended as
    the last column.
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

                # Some files were saved with one or two pandas index columns; drop them
                n_strip = 0
                while (n_strip < len(file_header) and
                       (file_header[n_strip].strip() == "" or
                        file_header[n_strip].startswith("Unnamed:"))):
                    n_strip += 1
                if n_strip:
                    file_header = file_header[n_strip:]

                if not header_written:
                    fieldnames = file_header + ([col_name] if extra_col else [])
                    writer = csv.writer(out_f)
                    writer.writerow(fieldnames)
                    header_written = True

                for row in reader:
                    if n_strip:
                        row = row[n_strip:]
                    if extra_col:
                        row = row + [extra_val]
                    writer.writerow(row)


def outer_join(dataframes: list[pd.DataFrame], on) -> pd.DataFrame:
    result = dataframes[0]
    for df in dataframes[1:]:
        result = pd.merge(result, df, on=on, how="outer")
    return result


def prefix_cols(df: pd.DataFrame, prefix: str, skip: str) -> pd.DataFrame:
    return df.rename(columns={c: f"{prefix}_{c}" for c in df.columns if c != skip})


def standardize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="mixed", dayfirst=False).dt.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Task 1: ElectricityPrices - stream-concatenate each subfolder per zone
# ---------------------------------------------------------------------------

def task1_electricity():
    _type_re = re.compile(r"FORECAST_(OFFSHORE|ONSHORE|SOLAR)_")

    def extract_type(path: Path) -> str:
        m = _type_re.search(path.stem)
        return m.group(1) if m else "UNKNOWN"

    for zone in ZONES:
        print(f"  Zone: {zone}")
        base = RAW / f"ElectricityPrices/{zone}"
        out_dir = OUT / f"ElectricityPrices/{zone}"

        print("    Loads...")
        stream_concat(sorted((base / "Loads").glob("*.csv")), out_dir / "Loads.csv")

        print("    Prices...")
        stream_concat(sorted((base / "Prices").glob("*.csv")), out_dir / "Prices.csv")

        print("    Renewables...")
        stream_concat(
            sorted((base / "Renewables").glob("*.csv")),
            out_dir / "Renewables.csv",
            extra_col={"Type": extract_type},
        )

        print("    Transmission...")
        stream_concat(
            sorted((base / "Transmission").glob("*.csv")),
            out_dir / "Transmission.csv",
        )


# ---------------------------------------------------------------------------
# Task 2: Climate Variables - one file per zone
#
# Raw files are {var}_{zone}_2018_2024.csv.gz. Cloud cover (cc) does not exist
# for DK1 and DK2, so missing files are skipped. Columns other than DATE and
# staid get the variable code as prefix.
# ---------------------------------------------------------------------------

def task2_climate():
    VARS = ["fg", "rr", "ss", "tg", "cc"]
    out_dir = OUT / "Climate Variables"
    out_dir.mkdir(parents=True, exist_ok=True)

    for zone in ZONES:
        print(f"  Zone: {zone}")
        zone_dir = RAW / f"Climate Variables/{zone}"
        dfs = []

        for var in VARS:
            gz_path = zone_dir / f"{var}_{zone}_2018_2024.csv.gz"
            if not gz_path.exists():
                print(f"    {var}: not found, skipping")
                continue
            print(f"    {var}: {gz_path.name}")
            df = pd.read_csv(gz_path, compression="gzip", low_memory=False)

            # Station metadata, not needed
            drop_cols = [c for c in df.columns
                         if c in ("ZONE_ID", "LAT_DEC", "LON_DEC", "elevation_m", "station_name")]
            df = df.drop(columns=drop_cols, errors="ignore")

            df = prefix_cols(df, var, skip="DATE")
            # prefix_cols also renamed staid; put it back as a join key
            if f"{var}_staid" in df.columns:
                df = df.rename(columns={f"{var}_staid": "staid"})

            dfs.append(df)

        if not dfs:
            print(f"    No climate files found for {zone}, skipping.")
            continue

        result = outer_join(dfs, on=["DATE", "staid"])
        result.to_csv(out_dir / f"{zone}.csv", index=False)
        print(f"    Saved {zone}.csv: {result.shape}")


# ---------------------------------------------------------------------------
# Task 3: EnergySeries - same as Case A
# ---------------------------------------------------------------------------

def task3_energy():
    base = RAW / "EnergySeries"
    out_dir = OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for f in sorted(base.glob("*.csv")):
        prefix = f.stem
        df = pd.read_csv(f, low_memory=False)
        df["Date"] = standardize_date(df["Date"])
        df = prefix_cols(df, prefix, skip="Date")
        dfs.append(df)

    result = outer_join(dfs, on="Date")
    result.to_csv(out_dir / "EnergySeries.csv", index=False)
    print(f"  EnergySeries saved: {result.shape}")


# ---------------------------------------------------------------------------
# Task 4: MacroFinance - Daily (4 files), Monthly (10 files), Quarterly (5 files)
# ---------------------------------------------------------------------------

def task4_macro():
    base = RAW / "MacroFinance"
    out_dir = OUT / "MacroFinance"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Daily: columns prefixed with the file name
    print("  MacroFinance Daily...")
    dfs = []
    for f in sorted((base / "Daily").glob("*.csv")):
        df = pd.read_csv(f, low_memory=False)
        if "observation_date" in df.columns:
            df = df.rename(columns={"observation_date": "Date"})
        df["Date"] = standardize_date(df["Date"])
        df = prefix_cols(df, f.stem, skip="Date")
        dfs.append(df)
    daily = outer_join(dfs, on="Date")
    daily.to_csv(out_dir / "Daily.csv", index=False)
    print(f"    Daily saved: {daily.shape}")

    # Monthly: no prefix. The EURIBOR files come in a different layout (DATE,
    # TIME PERIOD, long value name) and are reshaped to match the FRED files.
    print("  MacroFinance Monthly...")
    dfs = []
    for f in sorted((base / "Monthly").glob("*.csv")):
        df = pd.read_csv(f, low_memory=False)
        if "DATE" in df.columns and "observation_date" not in df.columns:
            # EURIBOR dates are month-end; move them to the first of the month
            # so they line up with the other monthly series
            val_col = [c for c in df.columns if c not in ("DATE", "TIME PERIOD")][0]
            short_name = f.stem  # e.g. "EURIBOR_1month"
            df = df[["DATE", val_col]].rename(
                columns={"DATE": "observation_date", val_col: short_name}
            )
            df["observation_date"] = (
                pd.to_datetime(df["observation_date"])
                .dt.to_period("M").dt.to_timestamp()
                .dt.strftime("%Y-%m-%d")
            )
        dfs.append(df)
    monthly = outer_join(dfs, on="observation_date")
    monthly.to_csv(out_dir / "Monthly.csv", index=False)
    print(f"    Monthly saved: {monthly.shape}")

    # Quarterly: no prefix
    print("  MacroFinance Quarterly...")
    dfs = [pd.read_csv(f, low_memory=False) for f in sorted((base / "Quarterly").glob("*.csv"))]
    quarterly = outer_join(dfs, on="observation_date")
    quarterly.to_csv(out_dir / "Quarterly.csv", index=False)
    print(f"    Quarterly saved: {quarterly.shape}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Task 1: ElectricityPrices (5 zones)")
    task1_electricity()

    print("\nTask 2: Climate Variables (5 zones, gzip)")
    task2_climate()

    print("\nTask 3: EnergySeries")
    task3_energy()

    print("\nTask 4: MacroFinance")
    task4_macro()

    print(f"\nDone. Output written to: {OUT}")
