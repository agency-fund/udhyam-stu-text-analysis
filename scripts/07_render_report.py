"""Render the Udhyam report Quarto file to HTML for GitHub Pages."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Sequence

FALLBACK_HTML = """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>Udhyam Student Text Analysis Report</title>
</head>
<body>
  <h1>Udhyam Student Text Analysis Report</h1>
  <p>The HTML report could not be rendered automatically because the Quarto CLI
  is not installed. Install Quarto from <a href=\"https://quarto.org\">quarto.org</a>
  and run:</p>
  <pre>quarto render {input_qmd}</pre>
  <p>This placeholder ensures GitHub Pages remains available until the full
  rendering step is executed.</p>
</body>
</html>
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to the QMD file")
    parser.add_argument("--output", required=True, help="Target HTML output path")
    return parser.parse_args(argv)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    input_qmd = Path(args.input)
    output_html = Path(args.output)

    ensure_parent(output_html)

    try:
        command = [
            "quarto",
            "render",
            input_qmd.name,
            "--to",
            "html",
            "--output",
            output_html.name,
        ]

        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            cwd=str(input_qmd.parent),
        )
        rendered_path = input_qmd.parent / output_html.name
        if rendered_path != output_html:
            output_html.write_bytes(rendered_path.read_bytes())
    except FileNotFoundError:
        output_html.write_text(FALLBACK_HTML.format(input_qmd=input_qmd))
    except subprocess.CalledProcessError as exc:  # pragma: no cover - rare runtime issue
        fallback = FALLBACK_HTML.format(input_qmd=input_qmd)
        fallback += (
            "\n<!-- Rendering failed with the following stderr: \n"
            + exc.stderr
            + "\n-->"
        )
        output_html.write_text(fallback)


if __name__ == "__main__":
    main()
