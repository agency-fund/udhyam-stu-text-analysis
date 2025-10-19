"""Clean Udhyam raw chat transcripts and write a tidy CSV."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence, Tuple

import pandas as pd


DEFAULT_INPUT_PATH = os.getenv(
    "UDHYAM_RAW_DATA_PATH",
    "data/raw/AI Help (Udhyam) student usage_Table - Sheet1.csv",
)

DEFAULT_OUTPUT_PATH = "data/cleaned/messages_cleaned.csv"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean Udhyam chat transcripts exported from Google Sheets."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help="Path to the raw CSV export (defaults to UDHYAM_RAW_DATA_PATH or the bundled sample).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Where the cleaned CSV should be written (default: data/cleaned/messages_cleaned.csv).",
    )
    return parser.parse_args(argv)


def parse_session_id(session_id: str) -> Tuple[pd.Timestamp, str]:
    session_id_str = str(session_id)
    datetime_str = session_id_str[:19]
    whatsapp_id = session_id_str[19:]
    dt = pd.to_datetime(datetime_str, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    if pd.isna(dt):
        raise ValueError(f"Could not parse datetime from session_id '{session_id}'")
    return dt, whatsapp_id


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if "session_id" not in df.columns:
        raise KeyError("Expected column 'session_id' in the raw export")

    dt_series, whatsapp_series = zip(*df["session_id"].map(parse_session_id))
    cleaned = df.copy()
    cleaned["datetime"] = list(dt_series)
    cleaned["whatsapp_id"] = list(whatsapp_series)
    cleaned = cleaned.drop(columns=["session_id"], errors="ignore")

    rename_map = {
        "Question": "user_msg",
        "Category": "user_msg_category",
        "Response": "ai_msg",
        "translated_answer": "ai_msg_en",
    }
    cleaned = cleaned.rename(columns={k: v for k, v in rename_map.items() if k in cleaned.columns})
    cleaned = cleaned.drop(columns=["cal_role"], errors="ignore")

    # Move datetime/whatsapp_id to the front for readability
    front_cols = ["datetime", "whatsapp_id"]
    remaining = [col for col in cleaned.columns if col not in front_cols]
    cleaned = cleaned[front_cols + remaining]
    return cleaned


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output)

    print(f"Loading raw data from {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} rows with columns: {df.columns.tolist()}")

    cleaned = clean_dataframe(df)
    ensure_parent_dir(output_path)
    cleaned.to_csv(output_path, index=False)
    print(f"Cleaned data written to {output_path}")


if __name__ == "__main__":
    main()
