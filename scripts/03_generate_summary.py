"""Generate summary tables for translated Udhyam chat messages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/cleaned/messages_translated.csv")
    parser.add_argument("--datetime-summary", default="data/analysis/datetime_overview.json")
    parser.add_argument("--cal-state-summary", default="data/analysis/cal_state_by_user.csv")
    parser.add_argument("--message-stats", default="data/analysis/message_stats.csv")
    return parser.parse_args(argv)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def compute_datetime_summary(df: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {
        "total_messages": int(df.shape[0]),
        "unique_users": int(df["whatsapp_id"].nunique()) if "whatsapp_id" in df.columns else None,
    }

    if "datetime" in df.columns:
        dt_series = pd.to_datetime(df["datetime"], errors="coerce")
        dt_series = dt_series.dropna()
        if not dt_series.empty:
            summary["start_datetime"] = dt_series.min().isoformat()
            summary["end_datetime"] = dt_series.max().isoformat()
            summary["total_days"] = int((dt_series.max() - dt_series.min()).days) + 1
            daily_counts = (
                dt_series.dt.date.value_counts().sort_index().rename_axis("date").rename("messages")
            )
            summary["messages_by_date"] = [
                {"date": str(index), "messages": int(value)} for index, value in daily_counts.items()
            ]
    return summary


def prepare_cal_state_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "cal_state" not in df.columns or "whatsapp_id" not in df.columns:
        return pd.DataFrame(columns=["whatsapp_id", "cal_state", "message_count"])

    grouped = (
        df.assign(cal_state=df["cal_state"].fillna("Unknown"))
        .groupby(["whatsapp_id", "cal_state"], dropna=False)
        .size()
        .reset_index(name="message_count")
        .sort_values(["whatsapp_id", "message_count"], ascending=[True, False])
    )
    return grouped


def compute_message_stats(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for column, label in [("user_msg_en", "user"), ("ai_msg_en", "assistant")]:
        if column not in df.columns:
            continue
        series = df[column].fillna("")
        lengths = series.astype(str).map(len)
        tokens = series.astype(str).str.split()
        token_counts = tokens.map(len)
        records.append(
            {
                "role": label,
                "messages": int((series.str.strip() != "").sum()),
                "avg_characters": float(lengths.mean()),
                "median_characters": float(lengths.median()),
                "avg_tokens": float(token_counts.mean()),
                "median_tokens": float(token_counts.median()),
            }
        )

    return pd.DataFrame(records)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    df = pd.read_csv(args.input)

    datetime_summary = compute_datetime_summary(df)
    cal_state_summary = prepare_cal_state_summary(df)
    message_stats = compute_message_stats(df)

    datetime_path = Path(args.datetime_summary)
    cal_state_path = Path(args.cal_state_summary)
    message_stats_path = Path(args.message_stats)

    ensure_parent(datetime_path)
    ensure_parent(cal_state_path)
    ensure_parent(message_stats_path)

    datetime_path.write_text(json.dumps(datetime_summary, indent=2))
    cal_state_summary.to_csv(cal_state_path, index=False)
    message_stats.to_csv(message_stats_path, index=False)


if __name__ == "__main__":
    main()
