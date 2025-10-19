"""Topic modeling for translated Udhyam chat messages using BERTopic.

Messages are in English after translation from Hindi, Punjabi, and Hinglish.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

try:
    from bertopic import BERTopic  # type: ignore
    from sklearn.feature_extraction.text import CountVectorizer
except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "Missing dependencies for topic modeling.\n"
        "Install with: pip install bertopic[visualization] sentence-transformers"
    ) from exc

import nltk
from nltk.corpus import stopwords


DEFAULT_INPUT = "data/cleaned/messages_translated.csv"
OUTPUT_BASE = Path("data/analysis")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Translated CSV input")
    parser.add_argument("--min-topic-size", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=8, help="Top keywords per topic")
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional limit per role for debugging",
    )
    return parser.parse_args(argv)


def ensure_stopwords() -> None:
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:  # pragma: no cover - first-time setup
        nltk.download("stopwords", quiet=True)


def ensure_output_dirs() -> None:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)


def build_stopwords() -> list[str]:
    """Build stopword list for English (messages translated from Hindi/Punjabi/Hinglish)."""
    ensure_stopwords()
    english = set(stopwords.words("english"))

    # Add common filler words and platform-specific terms
    extra = {
        "eh",
        "ah",
        "uh",
        "mm",
        "mmm",
        "okay",
        "ok",
        "yeah",
        "yes",
        "no",
        "thanks",
        "thank",
        "please",
        "hello",
        "hi",
        "hey",
    }
    return list(english | extra)


def create_topic_model(stop_words: list[str], min_topic_size: int) -> BERTopic:
    """Create BERTopic model configured for English text.

    Uses English language model since all messages are translated to English
    from Hindi, Punjabi, and Hinglish.
    """
    vectorizer_model = CountVectorizer(stop_words=stop_words, ngram_range=(1, 2), min_df=2)
    return BERTopic(
        language="english",  # Changed from "multilingual" since data is English-only
        calculate_probabilities=False,
        vectorizer_model=vectorizer_model,
        min_topic_size=min_topic_size,
        verbose=False,
    )


def summarise_topics(topic_model: BERTopic, role: str, top_n: int) -> pd.DataFrame:
    info = topic_model.get_topic_info()
    info = info[info.Topic != -1].copy()
    info.insert(0, "message_role", role)
    info.rename(columns={"Name": "keywords", "Count": "message_count", "Topic": "topic_id"}, inplace=True)

    rows = []
    for topic_id in info["topic_id"].tolist():
        terms = topic_model.get_topic(topic_id)
        if not terms:
            continue
        keywords = ", ".join(word for word, _ in terms[:top_n])
        rows.append({
            "message_role": role,
            "topic_id": topic_id,
            "keywords": keywords,
            "message_count": int(info.loc[info.topic_id == topic_id, "message_count"].iloc[0]),
        })
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    ensure_output_dirs()

    df = pd.read_csv(args.input)

    stop_words = build_stopwords()

    results = []
    topic_keywords_frames: list[pd.DataFrame] = []

    role_columns = {
        "user": "user_msg_en",
        "assistant": "ai_msg_en",
    }

    for role, column in role_columns.items():
        if column not in df.columns:
            continue
        texts = df[column].fillna("").astype(str)
        mask = texts.str.strip() != ""
        texts = texts[mask]
        if args.max_docs is not None:
            texts = texts.head(args.max_docs)
        if texts.empty:
            continue

        print(f"Fitting BERTopic for {role} ({len(texts)} messages)...")
        topic_model = create_topic_model(stop_words, args.min_topic_size)
        topics, _ = topic_model.fit_transform(texts.tolist())

        info = topic_model.get_topic_info()
        info.insert(0, "message_role", role)
        results.append(info)

        keywords_df = summarise_topics(topic_model, role, args.top_n)
        topic_keywords_frames.append(keywords_df)

        info.to_csv(OUTPUT_BASE / f"topic_info_{role}.csv", index=False)

        message_assignments = pd.DataFrame(
            {
                "message_role": role,
                "topic_id": topics,
            },
            index=texts.index,
        )
        message_assignments["message"] = texts.values
        message_assignments.to_csv(OUTPUT_BASE / f"message_topics_{role}.csv", index=False)

    if topic_keywords_frames:
        pd.concat(topic_keywords_frames, ignore_index=True).to_csv(
            OUTPUT_BASE / "topic_keywords.csv", index=False
        )
    else:
        pd.DataFrame(columns=["message_role", "topic_id", "keywords", "message_count"]).to_csv(
            OUTPUT_BASE / "topic_keywords.csv", index=False
        )

    print("✓ Topic modeling completed. Outputs written to data/analysis/.")


if __name__ == "__main__":
    main()
