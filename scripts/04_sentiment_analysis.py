"""Sentiment analysis for translated Udhyam chat messages."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from tqdm import tqdm

try:
    from transformers import pipeline  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "Missing dependency: install transformers (and torch).\n"
        "Run `pip install transformers torch` within your environment."
    ) from exc


DEFAULT_INPUT = "data/cleaned/messages_translated.csv"
OUTPUT_BASE = Path("data/analysis")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Translated CSV input")
    parser.add_argument(
        "--model",
        default="nlptown/bert-base-multilingual-uncased-sentiment",
        help="Hugging Face model to use for sentiment analysis",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Prediction batch size")
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional limit for debugging (process first N rows per role)",
    )
    return parser.parse_args(argv)


def ensure_output_dirs() -> None:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)


def chunked(iterable: Iterable, size: int) -> Iterable[list]:
    chunk: list = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def analyse_texts(texts: pd.Series, sentiment_model, batch_size: int) -> list[dict[str, object]]:
    texts = texts.fillna("").astype(str)
    predictions: list[dict[str, object]] = []

    iterator = chunked(texts.tolist(), batch_size)
    total = len(texts)
    for start_index, batch in enumerate(iterator):
        outputs = sentiment_model(batch)
        for offset, (text, pred) in enumerate(zip(batch, outputs)):
            label = pred["label"]
            score = float(pred["score"])
            try:
                stars = int(label.split()[0])
            except (IndexError, ValueError):
                stars = None
            predictions.append(
                {
                    "text": text,
                    "label": label,
                    "score": score,
                    "stars": stars,
                }
            )
        tqdm.write(f"Processed {(start_index + 1) * batch_size}/{total} messages", end="\r")
    tqdm.write("" )
    return predictions


def build_dataframe(df: pd.DataFrame, role: str, texts: pd.Series, preds: list[dict[str, object]]) -> pd.DataFrame:
    result = df.loc[texts.index, ["whatsapp_id", "datetime"]].copy()
    result.insert(0, "role", role)
    result["message"] = texts.values
    for key in ["label", "score", "stars"]:
        result[f"sentiment_{key}"] = [pred.get(key) for pred in preds]
    return result


def aggregate_overview(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    for role, frame in frames.items():
        frame = frame.copy()
        frame = frame[frame["message"].str.strip() != ""]
        if frame.empty:
            continue
        counts = frame["sentiment_label"].value_counts().to_dict()
        record = {
            "role": role,
            "messages": int(frame.shape[0]),
            "mean_score": float(frame["sentiment_score"].mean()),
            "median_score": float(frame["sentiment_score"].median()),
            "mean_stars": float(frame["sentiment_stars"].dropna().mean()) if frame["sentiment_stars"].notna().any() else None,
        }
        for sentiment_label, count in counts.items():
            record[f"count_{sentiment_label.replace(' ', '_')}"] = int(count)
        records.append(record)
    return pd.DataFrame(records)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    ensure_output_dirs()

    df = pd.read_csv(args.input)

    sentiment_model = pipeline(
        "sentiment-analysis",
        model=args.model,
        truncation=True,
        max_length=512,
    )

    role_columns = {
        "user": "user_msg_en",
        "assistant": "ai_msg_en",
    }

    per_role_results: dict[str, pd.DataFrame] = {}

    for role, column in role_columns.items():
        if column not in df.columns:
            continue
        texts = df[column].fillna("")
        if args.max_items is not None:
            texts = texts.head(args.max_items)
        clean_mask = texts.str.strip() != ""
        texts = texts[clean_mask]
        if texts.empty:
            continue
        print(f"Analyzing {len(texts)} {role} messages...")
        predictions = analyse_texts(texts, sentiment_model, args.batch_size)
        role_df = build_dataframe(df.loc[texts.index], role, texts, predictions)
        per_role_results[role] = role_df

    if not per_role_results:
        raise SystemExit("No messages found for sentiment analysis.")

    user_df = per_role_results.get("user", pd.DataFrame())
    assistant_df = per_role_results.get("assistant", pd.DataFrame())
    combined = pd.concat(per_role_results.values(), ignore_index=True)

    if not user_df.empty:
        user_df.to_csv(OUTPUT_BASE / "sentiment_user.csv", index=False)
    if not assistant_df.empty:
        assistant_df.to_csv(OUTPUT_BASE / "sentiment_ai.csv", index=False)
    combined.to_csv(OUTPUT_BASE / "sentiment_all.csv", index=False)

    overview = aggregate_overview(
        {role: frame.rename(columns={"sentiment_label": "sentiment_label", "sentiment_score": "sentiment_score"}) for role, frame in per_role_results.items()}
    )
    overview.to_csv(OUTPUT_BASE / "sentiment_overview.csv", index=False)

    print("✓ Sentiment analysis completed. Outputs written to data/analysis/.")


if __name__ == "__main__":
    main()
