import argparse
import os
import time
from typing import Dict, Tuple

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INPUT_COST_PER_1M = 0.150
OUTPUT_COST_PER_1M = 0.600

SYSTEM_PROMPT = (
    "You are a professional translator for Indian languages.\n"
    "Translate the user's message to natural English.\n"
    "The message may include Hindi, Punjabi, Hinglish (Roman script), English, "
    "or any mixture.\n"
    "Detect the languages automatically and keep any fluent English segments as-is.\n"
    "Return only the translation without adding explanations."
)

DEFAULT_MODEL = os.getenv("OPENAI_TRANSLATION_MODEL", "gpt-4o-mini")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate user messages to English using the OpenAI API."
    )
    parser.add_argument(
        "--input",
        default="data/messages_cleaned.csv",
        help="Path to the cleaned messages CSV file.",
    )
    parser.add_argument(
        "--output",
        default="data/messages_translated_openai.csv",
        help="Path where the translated CSV will be written.",
    )
    parser.add_argument(
        "--report",
        default="data/translation_cost_report.txt",
        help="Path where the translation cost report will be written.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for the model.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=500,
        help="Maximum tokens allowed in the translation.",
    )
    parser.add_argument(
        "--throttle",
        type=float,
        default=0.0,
        help="Optional delay (in seconds) inserted between API calls.",
    )
    return parser.parse_args()


def extract_text(response) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    for item in getattr(response, "output", []):
        for content in getattr(item, "content", []):
            if getattr(content, "type", "") in {"output_text", "text"}:
                return content.text

    raise ValueError("No text content returned by the model.")


def translate_with_openai(
    text: str, model: str, temperature: float, max_output_tokens: int
) -> Tuple[str, int, int]:
    if not text:
        return "", 0, 0

    try:
        response = client.responses.create(
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            ],
        )

        translated_text = extract_text(response).strip()
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

        return translated_text, input_tokens, output_tokens

    except Exception as exc:
        print(f"\nError translating text: {text[:50]}... | Error: {exc}")
        return text, 0, 0


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    print("Loading cleaned data...")
    messages = pd.read_csv(args.input)
    print(f"\nTotal messages: {len(messages)}")

    if "user_msg" not in messages.columns:
        raise KeyError("Column 'user_msg' not found in the input data.")

    translation_cache: Dict[str, str] = {}
    translated_messages = []
    total_input_tokens = 0
    total_output_tokens = 0
    api_calls = 0

    print("\n" + "=" * 80)
    print("TRANSLATING USER MESSAGES WITH OPENAI...")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(
        f"Cost: ${INPUT_COST_PER_1M} per 1M input tokens, "
        f"${OUTPUT_COST_PER_1M} per 1M output tokens"
    )

    for idx, raw_text in enumerate(tqdm(messages["user_msg"], desc="Translating")):
        normalized_text = "" if pd.isna(raw_text) else str(raw_text).strip()

        if not normalized_text:
            translated_messages.append("")
            continue

        cached_translation = translation_cache.get(normalized_text)
        if cached_translation is not None:
            translated_messages.append(cached_translation)
            continue

        translation, input_tokens, output_tokens = translate_with_openai(
            normalized_text,
            model=args.model,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
        )

        translation_cache[normalized_text] = translation
        translated_messages.append(translation)

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        api_calls += 1

        if api_calls % 50 == 0:
            current_cost = (
                total_input_tokens / 1_000_000 * INPUT_COST_PER_1M
                + total_output_tokens / 1_000_000 * OUTPUT_COST_PER_1M
            )
            print(
                f"\n  Progress: {api_calls} unique requests | "
                f"Cost so far: ${current_cost:.4f}"
            )

        if args.throttle > 0:
            time.sleep(args.throttle)

    if "user_msg_en" in messages.columns:
        messages.drop(columns=["user_msg_en"], inplace=True)

    insert_at = messages.columns.get_loc("user_msg") + 1
    messages.insert(insert_at, "user_msg_en", translated_messages)

    total_cost = (
        total_input_tokens / 1_000_000 * INPUT_COST_PER_1M
        + total_output_tokens / 1_000_000 * OUTPUT_COST_PER_1M
    )

    original_clean = messages["user_msg"].fillna("").astype(str).str.strip()
    translated_clean = messages["user_msg_en"].fillna("").astype(str).str.strip()
    changed_mask = original_clean != translated_clean
    translated_rows = int(changed_mask.sum())
    unchanged_rows = len(messages) - translated_rows
    non_empty_inputs = int((original_clean != "").sum())

    print("\n" + "=" * 80)
    print("TRANSLATION COMPLETE")
    print("=" * 80)
    print(f"\nUnique API calls: {api_calls}")
    print(f"Total input tokens: {total_input_tokens:,}")
    print(f"Total output tokens: {total_output_tokens:,}")
    print(f"Total cost: ${total_cost:.4f}")
    if api_calls:
        print(f"Average cost per API call: ${total_cost / api_calls:.6f}")

    print("\n" + "=" * 80)
    print("SAMPLE TRANSLATIONS")
    print("=" * 80)
    print(messages[["user_msg", "user_msg_en"]].head(20))

    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(
        f"Rows translated (changed output): {translated_rows} "
        f"({translated_rows / len(messages) * 100:.1f}%)"
    )
    print(
        f"Rows unchanged: {unchanged_rows} "
        f"({unchanged_rows / len(messages) * 100:.1f}%)"
    )

    messages.to_csv(args.output, index=False)
    print(f"\nTranslated data saved to: {args.output}")

    cost_report = f"""
OPENAI TRANSLATION COST REPORT
{'=' * 80}

Model: {args.model}
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

Dataset:
- Total rows: {len(messages):,}
- Non-empty inputs: {non_empty_inputs:,}
- Unique API calls: {api_calls:,}

Translation Outcomes:
- Rows changed by translation: {translated_rows:,}
- Rows unchanged: {unchanged_rows:,}

Token Usage:
- Input tokens: {total_input_tokens:,}
- Output tokens: {total_output_tokens:,}
- Total tokens: {total_input_tokens + total_output_tokens:,}

Cost Breakdown:
- Input cost: ${total_input_tokens / 1_000_000 * INPUT_COST_PER_1M:.4f}
- Output cost: ${total_output_tokens / 1_000_000 * OUTPUT_COST_PER_1M:.4f}
- Total cost: ${total_cost:.4f}
- Average cost per API call: ${
        total_cost / api_calls if api_calls else 0
    :.6f}

Pricing:
- Input: ${INPUT_COST_PER_1M} per 1M tokens
- Output: ${OUTPUT_COST_PER_1M} per 1M tokens
"""

    with open(args.report, "w") as file:
        file.write(cost_report)

    print(f"\nCost report saved to: {args.report}")


if __name__ == "__main__":
    main()
