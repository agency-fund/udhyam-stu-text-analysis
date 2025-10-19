import argparse
import os
import re
import time
from collections import defaultdict
from typing import Dict, List, Tuple

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

INDIC_UNICODE_RANGES = [
    (0x0900, 0x097F),  # Devanagari
    (0x0980, 0x09FF),  # Bengali
    (0x0A00, 0x0A7F),  # Gurmukhi
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B00, 0x0B7F),  # Odia
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
]

COMMON_ENGLISH_WORDS = {
    "a",
    "about",
    "after",
    "all",
    "also",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "but",
    "by",
    "can",
    "come",
    "could",
    "day",
    "do",
    "for",
    "from",
    "get",
    "give",
    "go",
    "good",
    "have",
    "he",
    "her",
    "here",
    "him",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "know",
    "like",
    "long",
    "look",
    "make",
    "man",
    "many",
    "me",
    "more",
    "my",
    "new",
    "no",
    "not",
    "now",
    "of",
    "on",
    "one",
    "only",
    "or",
    "other",
    "our",
    "out",
    "people",
    "say",
    "see",
    "she",
    "so",
    "some",
    "take",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "thing",
    "think",
    "this",
    "those",
    "time",
    "to",
    "two",
    "up",
    "use",
    "very",
    "want",
    "was",
    "we",
    "well",
    "what",
    "when",
    "which",
    "who",
    "will",
    "with",
    "work",
    "would",
    "year",
    "you",
    "your",
}

ROMANIZED_HINDI_MARKERS = {
    "hai",
    "hain",
    "ho",
    "hona",
    "hogaa",
    "hoga",
    "honge",
    "hun",
    "hu",
    "hua",
    "haii",
    "hai?",
    "kya",
    "kyu",
    "kyun",
    "kyon",
    "kaise",
    "kaisey",
    "kahaan",
    "kahan",
    "kab",
    "kar",
    "kare",
    "karen",
    "karenge",
    "kariye",
    "karo",
    "karna",
    "karne",
    "karte",
    "nahi",
    "nahin",
    "mujhe",
    "hum",
    "hume",
    "humko",
    "batao",
    "kripya",
    "mein",
    "mera",
    "meri",
    "mere",
    "tha",
    "thi",
    "tum",
    "unka",
    "unki",
    "unke",
    "vala",
    "vala?",
    "wala",
    "wale",
    "wali",
    "ky",
    "kr",
    "raha",
    "rahe",
    "rahi",
    "sakta",
    "sakte",
    "sakthi",
    "sakti",
}

DEFAULT_MODEL = os.getenv("OPENAI_TRANSLATION_MODEL", "gpt-4o-mini")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate user messages to English using the OpenAI API."
    )
    parser.add_argument(
        "--input",
        default="data/cleaned/messages_cleaned.csv",
        help="Path to the cleaned messages CSV file.",
    )
    parser.add_argument(
        "--output",
        default="data/cleaned/messages_translated.csv",
        help="Path where the translated CSV will be written.",
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


def contains_indic_script(text: str) -> bool:
    for char in text:
        codepoint = ord(char)
        for lower, upper in INDIC_UNICODE_RANGES:
            if lower <= codepoint <= upper:
                return True
    return False


def has_latin_letters(text: str) -> bool:
    return any("A" <= ch <= "Z" or "a" <= ch <= "z" for ch in text)


def tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+", text.lower())


def english_word_ratio(text: str) -> float:
    tokens = tokenize_words(text)
    if not tokens:
        return 0.0

    matches = 0
    for token in tokens:
        if token in COMMON_ENGLISH_WORDS:
            matches += 1
            continue
        stripped = token.rstrip("s")
        if stripped in COMMON_ENGLISH_WORDS:
            matches += 1

    return matches / len(tokens)


def is_probably_english(text: str) -> bool:
    if not text:
        return False

    if any(ord(char) >= 128 for char in text):
        return False

    ratio = english_word_ratio(text)
    return ratio >= 0.4


def contains_romanized_hindi(text: str) -> bool:
    tokens = tokenize_words(text)
    if not tokens:
        return False
    hits = sum(token in ROMANIZED_HINDI_MARKERS for token in tokens)
    if hits >= 2:
        return True
    if hits == 1 and english_word_ratio(text) < 0.3:
        return True
    return False


def needs_retranslation(original: str, translated: str) -> bool:
    original = original or ""
    translated = translated or ""

    if not translated.strip():
        return bool(original.strip())

    if contains_indic_script(translated):
        return True

    if has_latin_letters(original) and not has_latin_letters(translated):
        return True

    if is_probably_english(original):
        if contains_romanized_hindi(translated):
            return True
        english_drop = english_word_ratio(original) - english_word_ratio(translated)
        if english_drop >= 0.2:
            return True

    return False


def translate_with_openai(
    text: str,
    model: str,
    temperature: float,
    max_output_tokens: int,
    force_english: bool = False,
) -> Tuple[str, int, int]:
    if not text:
        return "", 0, 0

    try:
        system_prompt = SYSTEM_PROMPT
        if force_english:
            system_prompt += (
                "\nRespond strictly in English using the Latin script."
                " If the input is already English, repeat it without changes."
            )

        response = client.responses.create(
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
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
    text_indices: Dict[str, List[int]] = defaultdict(list)
    translated_messages = []
    retry_candidates = set()
    failed_retries = []
    total_input_tokens = 0
    total_output_tokens = 0
    api_calls = 0
    reused_translations = 0

    if os.path.exists(args.output):
        try:
            previous = pd.read_csv(args.output)
            if (
                len(previous) == len(messages)
                and "user_msg" in previous.columns
                and "user_msg_en" in previous.columns
            ):
                prev_user = previous["user_msg"].fillna("").astype(str).str.strip()
                curr_user = messages["user_msg"].fillna("").astype(str).str.strip()
                if prev_user.equals(curr_user):
                    for orig, trans in zip(prev_user, previous["user_msg_en"]):
                        normalized_orig = str(orig).strip()
                        normalized_trans = "" if pd.isna(trans) else str(trans).strip()
                        if not normalized_orig:
                            continue
                        if needs_retranslation(normalized_orig, normalized_trans):
                            continue
                        if normalized_orig not in translation_cache:
                            translation_cache[normalized_orig] = normalized_trans
                    print(
                        f"\nReused {len(translation_cache)} cached translation(s) "
                        "from existing output file."
                    )
            else:
                print(
                    "\nExisting translation output found but could not be reused "
                    "(shape or required columns mismatch)."
                )
        except Exception as error:  # noqa: BLE001
            print(f"\nWarning: Failed to load existing translations. Reason: {error}")

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

        text_indices[normalized_text].append(idx)

        cached_translation = translation_cache.get(normalized_text)
        if cached_translation is not None:
            translated_messages.append(cached_translation)
            reused_translations += 1
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

        if needs_retranslation(normalized_text, translation):
            retry_candidates.add(normalized_text)

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

    if retry_candidates:
        print(
            f"\nRetrying {len(retry_candidates)} translation(s) that were not confidently English..."
        )

    for text in retry_candidates:
        translation, input_tokens, output_tokens = translate_with_openai(
            text,
            model=args.model,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            force_english=True,
        )

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        api_calls += 1

        if needs_retranslation(text, translation):
            failed_retries.append(text)
            continue

        translation_cache[text] = translation
        for index in text_indices[text]:
            translated_messages[index] = translation

    successful_retries = len(retry_candidates) - len(failed_retries)

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
    print(f"Non-empty inputs: {non_empty_inputs}")
    if reused_translations:
        print(f"Rows served from cache (no API call): {reused_translations}")
    if retry_candidates:
        print(
            f"\nAdditional translation attempts: {len(retry_candidates)} "
            f"(successful: {successful_retries}, failed: {len(failed_retries)})"
        )
    if failed_retries:
        print(
            "\nWarning: Some messages could not be confidently translated to English "
            "even after retries."
        )
        for text in failed_retries[:10]:
            print(f"  - {text[:75]}{'...' if len(text) > 75 else ''}")
        if len(failed_retries) > 10:
            print(f"  ...and {len(failed_retries) - 10} more.")

    messages.to_csv(args.output, index=False)
    print(f"\nTranslated data saved to: {args.output}")

    print("\nHint: track token usage above if you need manual cost records.")


if __name__ == "__main__":
    main()
