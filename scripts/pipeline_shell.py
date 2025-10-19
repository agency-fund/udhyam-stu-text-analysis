"""Launch an interactive shell with common pipeline artefacts pre-loaded."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd


BASE_TABLES = {
    "messages_cleaned": Path("data/cleaned/messages_cleaned.csv"),
}

TRANSLATION_TABLES = {
    "messages_translated": Path("data/cleaned/messages_translated.csv"),
}


def load_tables(include_translation: bool) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    for name, path in BASE_TABLES.items():
        if path.exists():
            tables[name] = pd.read_csv(path)

    if include_translation:
        for name, path in TRANSLATION_TABLES.items():
            if path.exists():
                tables[name] = pd.read_csv(path)
    return tables


def launch_shell(namespace: Dict[str, object]) -> None:
    banner = "Pipeline shell with tables loaded: " + ", ".join(sorted(namespace.keys()))
    try:
        from IPython import start_ipython

        start_ipython(argv=["--simple-prompt"], user_ns=namespace)
    except Exception:  # pragma: no cover - fallback if IPython missing
        import code

        code.interact(banner=banner, local=namespace)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-openai",
        action="store_true",
        help="Load OpenAI translated messages if the file exists.",
    )
    parser.add_argument(
        "--extra",
        nargs="*",
        default=[],
        help="Additional CSV files to load (name=path syntax supported).",
    )
    return parser.parse_args()


def parse_extra(entries: list[str]) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    for entry in entries:
        if "=" in entry:
            name, path_str = entry.split("=", 1)
        else:
            path_str = entry
            name = Path(path_str).stem
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Extra table '{entry}' not found at {path}")
        tables[name] = pd.read_csv(path)
    return tables


def main() -> None:
    args = parse_args()
    namespace: Dict[str, object] = {}
    namespace.update(load_tables(args.include_openai))
    namespace.update(parse_extra(args.extra))

    if not namespace:
        print("No tables found. Run the pipeline first or provide --extra paths.")
        return

    launch_shell(namespace)


if __name__ == "__main__":
    main()
