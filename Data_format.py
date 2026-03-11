#!/usr/bin/env python3
"""
Merge headerless FX CSVs into one file per pair and save to your training data folder,
and resample them to 15-minute candles in the format expected by the trading script.

Assumptions for each source CSV:
- No header row
- Column order: Date, Time, Open, High, Low, Close, Volume
- Date format like 'YYYY.MM.DD', Time like 'HH:MM'
- Filename contains a 6-letter FX pair (e.g., EURUSD) somewhere (e.g., DAT_MT_EURUSD_M1_202502.csv)

Output:
- One CSV per pair written to the --output_data_dir folder, named <PAIR>.csv
- 15-minute candles
- Columns: time, open, high, low, close, volume
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

PAIR_REGEX = re.compile(r"([A-Z]{6})")

# Target timeframe for resampling
TARGET_TIMEFRAME = "5min"  # 15-minute candles


def detect_pair_from_name(path: Path) -> Optional[str]:
    m = PAIR_REGEX.search(path.stem.upper())
    return m.group(1) if m else None


def read_headerless_csv(path: Path, encoding: Optional[str] = None) -> pd.DataFrame:
    """
    Read a single raw CSV (likely M1) and return a datetime-indexed OHLCV dataframe.
    Index is tz-aware UTC if possible.
    """
    cols = ["Date", "Time", "Open", "High", "Low", "Close", "Volume"]
    df = pd.read_csv(
        path,
        names=cols,
        header=None,
        encoding=encoding or "utf-8",
        low_memory=False,
    )

    # Build datetime index (assume UTC)
    dt = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        errors="coerce",
        utc=True,
    )

    # Drop rows with invalid datetime
    mask = ~dt.isna()
    df = df.loc[mask].copy()
    dt = dt.loc[mask]
    df.index = dt

    # Keep OHLCV
    out = df[["Open", "High", "Low", "Close"]].copy()
    if "Volume" in df.columns:
        out["Volume"] = df["Volume"]
    else:
        out["Volume"] = 0.0

    # Enforce numeric types (coerce errors to NaN then drop)
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.dropna()
    out = out.sort_index()

    return out


def resample_to_15m(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Take merged M1 OHLCV and resample to 15-minute candles.
    """
    # Ensure we have the expected columns
    merged = merged[["Open", "High", "Low", "Close", "Volume"]].copy()

    # Resample to 15-minute OHLCV
    df_15m = (
        merged.resample(TARGET_TIMEFRAME)
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna()
    )

    return df_15m


def merge_folder(
    input_dir: Path,
    output_data_dir: Path,
    encoding: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Path]:
    input_dir = input_dir.expanduser().resolve()
    output_data_dir = output_data_dir.expanduser().resolve()
    output_data_dir.mkdir(parents=True, exist_ok=True)

    files = list(input_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    groups: Dict[str, List[Path]] = {}
    for f in files:
        pair = detect_pair_from_name(f)
        if not pair:
            if verbose:
                print(f"[skip] Could not detect pair in filename: {f.name}")
            continue
        groups.setdefault(pair, []).append(f)

    if not groups:
        raise ValueError("No files with detectable FX pair names (e.g., EURUSD) found.")

    outputs: Dict[str, Path] = {}
    for pair, paths in sorted(groups.items()):
        if verbose:
            print(f"\n[{pair}] merging {len(paths)} files...")
            for p in sorted(paths):
                print(f"   - {p.name}")

        frames = []
        for p in sorted(paths):
            try:
                df = read_headerless_csv(p, encoding=encoding)
                frames.append(df)
            except Exception as e:
                print(f"   ! Skipping {p.name}: {e}")

        if not frames:
            print(f"   ! No valid data for {pair}; skipping.")
            continue

        # Merge all raw frames (e.g. M1 data) and clean duplicates
        merged = pd.concat(frames, axis=0)
        merged = merged[~merged.index.duplicated(keep="last")]
        merged = merged.sort_index()

        # Resample to 15-minute candles
        merged_15m = resample_to_15m(merged)

        # Rename columns to match other script expectations (lowercase)
        merged_15m = merged_15m.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "tick_volume",
            }
        )

        out_path = output_data_dir / f"{pair}.csv"

        # Write with 'time' as a column and OHLCV lowercase
        out = merged_15m.copy()
        out.insert(0, "time", out.index.astype(str))
        out.reset_index(drop=True, inplace=True)
        out.to_csv(out_path, index=False)

        if verbose:
            print(
                f"   → wrote {out_path}: rows={len(out)}, "
                f"start={merged_15m.index.min()}, end={merged_15m.index.max()}"
            )

        outputs[pair] = out_path

    return outputs


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Merge headerless FX CSVs per pair into <PAIR>.csv, "
            "resampled to 15-minute candles for your trading script."
        )
    )
    ap.add_argument(
        "input_dir",
        help="Folder containing the raw CSV files (e.g., monthly/minute data).",
    )
    ap.add_argument(
        "--output_data_dir",
        required=True,
        help=(
            "Training data folder to write merged <PAIR>.csv files "
            "(your project's data folder)."
        ),
    )
    ap.add_argument("--encoding", default=None, help="CSV encoding (default utf-8).")
    ap.add_argument("-q", "--quiet", action="store_true", help="Less verbose output.")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_data_dir)

    results = merge_folder(
        input_dir, output_dir, encoding=args.encoding, verbose=not args.quiet
    )

    if not args.quiet:
        print("\nDone. Merged pairs (15m candles):")
        for pair, path in results.items():
            print(f" - {pair}: {path}")


if __name__ == "__main__":
    main()
