"""Create the Udhyam student text analysis Quarto report."""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path
from typing import Sequence


REPORT_TEMPLATE = """---
title: "Udhyam Student Text Analysis Report"
date: {report_date}
format:
  html:
    toc: true
    toc-depth: 3
    theme: cosmo
execute:
  echo: false
  warning: false
  message: false
---

```{{python}}
import json
from pathlib import Path
import pandas as pd

datetime_overview = json.loads(Path("{datetime_json}").read_text())
cal_state_by_user = pd.read_csv("{cal_state_csv}")
message_stats = pd.read_csv("{message_stats_csv}")

sentiment_overview_path = Path("{sentiment_csv}")
topic_keywords_path = Path("{topics_csv}")

sentiment_overview = pd.read_csv(sentiment_overview_path) if sentiment_overview_path.exists() else pd.DataFrame()
topic_keywords = pd.read_csv(topic_keywords_path) if topic_keywords_path.exists() else pd.DataFrame()
```

# 1. Conversation Timeline Overview

```{{python}}
from pandas import json_normalize

summary_items = [
    ("Total messages", datetime_overview.get("total_messages", "NA")),
    ("Unique users", datetime_overview.get("unique_users", "NA")),
    ("Start", datetime_overview.get("start_datetime", "NA")),
    ("End", datetime_overview.get("end_datetime", "NA")),
    ("Coverage (days)", datetime_overview.get("total_days", "NA")),
]
pd.DataFrame(summary_items, columns=["Metric", "Value"])
```

```{{python}}
timeline = json_normalize(datetime_overview.get("messages_by_date", []))
if not timeline.empty:
    timeline["date"] = pd.to_datetime(timeline["date"])
    timeline.sort_values("date", inplace=True)
    timeline
else:
    print("No timeline data available.")
```

# 2. CAL State by Learner (whatsapp_id)

```{{python}}
if not cal_state_by_user.empty:
    cal_state_by_user
else:
    print("No CAL state information available.")
```

# 3. Message Length & Volume Summary

```{{python}}
if not message_stats.empty:
    message_stats
else:
    print("Message statistics not available.")
```

# 4. Sentiment Snapshot

```{{python}}
if not sentiment_overview.empty:
    sentiment_overview
else:
    print("Sentiment analysis outputs not available. Ensure translation data exists and rerun the sentiment pipeline step.")
```

# 5. Topic Highlights

```{{python}}
if not topic_keywords.empty:
    topic_keywords
else:
    print("Topic modeling outputs not available. Ensure translation data exists and rerun the topic modeling step.")
```

---

_Report generated automatically via Snakemake on {report_date}._
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datetime-json", default="data/analysis/datetime_overview.json")
    parser.add_argument("--cal-state-csv", default="data/analysis/cal_state_by_user.csv")
    parser.add_argument("--message-stats-csv", default="data/analysis/message_stats.csv")
    parser.add_argument("--sentiment-csv", default="data/analysis/sentiment_overview.csv")
    parser.add_argument("--topics-csv", default="data/analysis/topic_keywords.csv")
    parser.add_argument("--output", default="docs/udhyam_stu_text_analysis_report.qmd")
    return parser.parse_args(argv)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    output_path = Path(args.output)
    ensure_parent(output_path)

    output_dir = output_path.parent

    datetime_rel = os.path.relpath(args.datetime_json, start=output_dir)
    cal_state_rel = os.path.relpath(args.cal_state_csv, start=output_dir)
    message_stats_rel = os.path.relpath(args.message_stats_csv, start=output_dir)
    sentiment_rel = os.path.relpath(args.sentiment_csv, start=output_dir)
    topics_rel = os.path.relpath(args.topics_csv, start=output_dir)

    content = REPORT_TEMPLATE.format(
        report_date=date.today().isoformat(),
        datetime_json=datetime_rel,
        cal_state_csv=cal_state_rel,
        message_stats_csv=message_stats_rel,
        sentiment_csv=sentiment_rel,
        topics_csv=topics_rel,
    )

    output_path.write_text(content)


if __name__ == "__main__":
    main()
