"""Generate lightweight summary statistics for the cleaned messages dataset."""

import json
from pathlib import Path

import pandas as pd


def main(cleaned_path: Path, output_path: Path) -> None:
    df = pd.read_csv(cleaned_path)

    summary = {
        "rows": int(df.shape[0]),
        "columns": list(df.columns),
        "timestamp_range": {
            "min": df["datetime"].min() if "datetime" in df.columns else None,
            "max": df["datetime"].max() if "datetime" in df.columns else None,
        },
        "translation_columns": [col for col in df.columns if col.endswith("_en")],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    cleaned = Path(snakemake.input.cleaned)  # type: ignore[name-defined]
    output = Path(snakemake.output.summary)  # type: ignore[name-defined]
    main(cleaned, output)
