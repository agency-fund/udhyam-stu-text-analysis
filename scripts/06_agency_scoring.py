#!/usr/bin/env python3
"""Score agency for every translated user and assistant message using BERTAgent."""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from bertagent import BERTAgent, TOKENIZER_PARAMS
from transformers import AutoModelForSequenceClassification, AutoTokenizer


TORCH_LOAD_ERROR_SNIPPET = "Due to a serious vulnerability issue in `torch.load`"


class _SafeBERTAgent:
    """Fallback BERTAgent loader that enforces safetensor weights."""

    _MODEL_ID = "EnchantedStardust/bertagent-best"

    def __init__(self, device: str):
        self.factor = 1.0
        self.bias = 0.0
        self.tokenizer_params = TOKENIZER_PARAMS
        self.tokenizer = AutoTokenizer.from_pretrained(self._MODEL_ID, do_lower_case=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self._MODEL_ID,
            num_labels=1,
            use_safetensors=True,
        )
        self.model.to(device)
        self.model.eval()

    def predict(self, sentences: List[str]) -> List[float]:
        sentences = [re.sub(r"\s\s+", " ", sent).strip() for sent in sentences]
        batch_encodings = self.tokenizer(
            sentences,
            None,
            **self.tokenizer_params,
            return_tensors="pt",
        )
        batch_encodings.to(self.model.device)
        with torch.no_grad():
            logits = self.model(**batch_encodings)["logits"].cpu().numpy()
        scores = logits * self.factor + self.bias
        batch_encodings.to(self.model.device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return scores.ravel().tolist()


def load_agent(device: str):
    try:
        return BERTAgent(device=device)
    except ValueError as exc:
        if TORCH_LOAD_ERROR_SNIPPET in str(exc):
            print(
                "Falling back to safetensors-based BERTAgent loader because torch<2.6 "
                "cannot load pickle weights.",
                file=sys.stderr,
            )
            return _SafeBERTAgent(device=device)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score agency for every translated message (user and assistant)."
    )
    parser.add_argument("--input", required=True, help="Path to messages_translated.csv")
    parser.add_argument("--output-user", required=True, help="Output parquet for user messages")
    parser.add_argument(
        "--output-assistant", required=True, help="Output parquet for assistant messages"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for BERTAgent inference"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum whitespace-token length per message (extra tokens are truncated)",
    )
    parser.add_argument(
        "--pos-threshold",
        type=float,
        default=0.3,
        help="Threshold for classifying a message as high agency",
    )
    parser.add_argument(
        "--neg-threshold",
        type=float,
        default=-0.3,
        help="Threshold for classifying a message as low agency",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        help="Device to run inference on (defaults to auto-detect)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device(preferred: str | None) -> str:
    if preferred:
        if preferred == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


def truncate_tokens(text: str, max_tokens: int) -> str:
    tokens = text.split()
    if len(tokens) > max_tokens:
        return " ".join(tokens[:max_tokens])
    return text


def score_messages(
    texts: pd.Series,
    agent,
    batch_size: int,
    max_tokens: int,
) -> pd.Series:
    trimmed = texts.fillna("").astype(str).str.strip()
    scores = np.full(len(trimmed), np.nan, dtype=float)

    if trimmed.empty:
        return pd.Series(scores, index=trimmed.index, name="agency_score")

    if hasattr(agent, "score"):
        score_fn = agent.score
    elif hasattr(agent, "predict"):
        score_fn = agent.predict
    else:
        raise AttributeError("BERTAgent must expose either a 'score' or 'predict' method")

    valid_idx = trimmed.index[trimmed != ""]
    if not len(valid_idx):
        return pd.Series(scores, index=trimmed.index, name="agency_score")

    truncated_texts = [truncate_tokens(trimmed.loc[idx], max_tokens) for idx in valid_idx]

    for start in range(0, len(truncated_texts), batch_size):
        batch_texts = truncated_texts[start : start + batch_size]
        batch_scores = score_fn(batch_texts)
        if not isinstance(batch_scores, Iterable):
            raise TypeError("BERTAgent returned non-iterable scores")
        batch_indices = valid_idx[start : start + len(batch_scores)]
        if len(batch_indices) != len(batch_scores):
            raise RuntimeError("Mismatch between batch size and returned score count")
        for idx, score in zip(batch_indices, batch_scores):
            scores[trimmed.index.get_loc(idx)] = float(score)

    return pd.Series(scores, index=trimmed.index, name="agency_score")


def build_role_frame(
    base_df: pd.DataFrame,
    role: str,
    text_column: str,
    translated_column: str,
    scores: pd.Series,
    pos_threshold: float,
    neg_threshold: float,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "message_id": [f"{role}_{idx}" for idx in base_df.index],
            "row_index": base_df.index,
            "message_role": role,
        },
        index=base_df.index,
    )

    for column in ("datetime", "whatsapp_id"):
        if column in base_df.columns:
            frame[column] = base_df[column]

    if text_column in base_df.columns:
        frame["text_original"] = base_df[text_column]
    else:
        frame["text_original"] = ""

    if translated_column in base_df.columns:
        frame["text_translated"] = base_df[translated_column]
    else:
        frame["text_translated"] = frame["text_original"]

    frame["agency_score"] = scores.reindex(base_df.index, fill_value=np.nan)

    def _classify(series: pd.Series, threshold: float, comparator) -> pd.Series:
        result = pd.Series(pd.NA, index=series.index, dtype="boolean")
        mask = series.notna()
        result.loc[mask] = comparator(series.loc[mask], threshold)
        return result

    frame["is_high"] = _classify(frame["agency_score"], pos_threshold, lambda s, t: s >= t)
    frame["is_low"] = _classify(frame["agency_score"], neg_threshold, lambda s, t: s <= t)

    return frame.reset_index(drop=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    device = detect_device(args.device)
    agent = load_agent(device=device)

    def _series_or_empty(column: str) -> pd.Series:
        if column in df.columns:
            series = df[column]
        else:
            series = pd.Series(["" for _ in range(len(df))], index=df.index, dtype=str)
        return series

    user_scores = score_messages(
        _series_or_empty("user_msg_en"), agent, args.batch_size, args.max_tokens
    )
    assistant_scores = score_messages(
        _series_or_empty("ai_msg_en"), agent, args.batch_size, args.max_tokens
    )

    user_frame = build_role_frame(
        df,
        role="user",
        text_column="user_msg",
        translated_column="user_msg_en",
        scores=user_scores,
        pos_threshold=args.pos_threshold,
        neg_threshold=args.neg_threshold,
    )

    assistant_frame = build_role_frame(
        df,
        role="assistant",
        text_column="ai_msg",
        translated_column="ai_msg_en",
        scores=assistant_scores,
        pos_threshold=args.pos_threshold,
        neg_threshold=args.neg_threshold,
    )

    user_path = Path(args.output_user)
    assistant_path = Path(args.output_assistant)
    user_path.parent.mkdir(parents=True, exist_ok=True)
    assistant_path.parent.mkdir(parents=True, exist_ok=True)

    user_frame.to_parquet(user_path, index=False)
    assistant_frame.to_parquet(assistant_path, index=False)

    print(
        "Agency scoring complete."
        f"\n User messages processed: {len(user_frame):,}"
        f"\n Assistant messages processed: {len(assistant_frame):,}"
        f"\n User parquet: {user_path}"
        f"\n Assistant parquet: {assistant_path}"
    )


if __name__ == "__main__":
    main()
