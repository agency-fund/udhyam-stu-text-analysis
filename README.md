# Udhyam Student Text Analysis

Data cleaning and translation pipeline for Udhyam's AI chatbot interactions.
The workflow is orchestrated with **Snakemake**, giving you the reproducibility
and dependency tracking of R's `targets`/`drake`, plus easy DAG visualisation
and shortcuts for loading results into an interactive session.

## Repository Structure

```
.
├── Snakefile                     # Snakemake workflow definition
├── config/
│   └── pipeline.yaml             # User-adjustable configuration (raw-data path, flags)
├── data/
│   └── cleaned/                  # Cleaned artefacts (created by the workflow)
├── scripts/
│   ├── 01_load_and_clean.py      # Deterministic cleaning CLI
│   ├── 02_translate_messages.py  # OpenAI translation with Hinglish heuristics
│   ├── messages_summary.py       # Generates JSON summary (Snakemake script)
│   └── pipeline_shell.py         # Launch an analysis shell with outputs pre-loaded
├── requirements.txt              # Python dependencies (Snakemake included)
├── Makefile                      # Convenience wrappers around Snakemake
└── README.md
```

## Setup

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure OpenAI access** (required for the OpenAI translation rule):

   ```bash
   export OPENAI_API_KEY="sk-..."
   # Optional: export OPENAI_TRANSLATION_MODEL="gpt-4o-mini"
   ```

3. **Point to the raw CSV export**.

   - By default the pipeline looks for `data/raw/AI_Help_Udhyam_student_usage_Table_Sheet1.csv`.
   - To customise, edit `config/pipeline.yaml` (change `raw_data:`), or override at runtime:

     ```bash
     snakemake --config raw_data=/absolute/path/to/export.csv --cores 1
     ```

4. **Toggle the OpenAI step** by editing `config/pipeline.yaml`:

   ```yaml
   enable_openai_translation: false   # enable when you want to re-run OpenAI translation
   ```

## Running the Workflow

The `Snakefile` covers three core rules:

| Rule               | Outputs                                                              | Purpose |
|--------------------|----------------------------------------------------------------------|---------|
| `clean_messages`   | `data/cleaned/messages_cleaned.csv`                                   | Cleans the raw export, parses `session_id`, renames/reorders columns, and prepares the canonical dataset for downstream steps. |
| `messages_summary` | `data/cleaned/messages_summary.json`                                 | Stores quick QC metadata (row count, column list, datetime range, translation columns). |
| `openai_translate` | `data/cleaned/messages_translated.csv` (optional)                    | Produces an English translation using OpenAI and logs token usage in the console. Enabled when `enable_openai_translation` is true. |

Use Snakemake directly or via the Makefile wrappers:

```bash
make pipeline-status      # Overview of what is up-to-date (uses `snakemake --summary`)
make pipeline-run         # Run the full DAG (defaults to a single local core)
make pipeline-graph       # Emit DAG as pipeline_graph.dot (Graphviz DOT format)
make pipeline-graph-png   # Render DAG straight to pipeline_graph.png (requires Graphviz `dot`)
make pipeline-shell       # Launch an IPython shell with pipeline outputs pre-loaded
```

Need more control? Call Snakemake yourself:

```bash
# Dry-run to inspect planned actions
snakemake --cores 1 --snakefile Snakefile -n

# Run with a custom config on the fly
snakemake --cores 4 --config raw_data=/path/to/export.csv enable_openai_translation=true
```

## Visualising the DAG

Generate a DOT file or PNG via the Makefile targets above. Manually:

```bash
snakemake --cores 1 --dag > pipeline_graph.dot
dot -Tpng pipeline_graph.dot -o pipeline_graph.png
```

Opening `pipeline_graph.png` reveals the precise order and dependency structure
validated before execution.

## Interactive Analysis Shell

Once the pipeline has run, start a shell with key outputs ready to use:

```bash
python scripts/pipeline_shell.py --include-openai
# -> IPython (if installed) or a basic REPL with `messages_cleaned`
#    and (if available) `messages_translated` DataFrames.
```

Use `--extra name=path/to/file.csv` to load additional CSV artefacts on the fly.

## Script Highlights

- `scripts/01_load_and_clean.py`
  - Pure cleaning step exposed via a simple CLI (`--input`, `--output`).
  - Parses `session_id`, renames columns, and emits `messages_cleaned.csv` under
    `data/cleaned/`.

- `scripts/02_translate_messages.py`
  - Uses OpenAI's API with Hinglish detection and usage-cost tracking.
  - Configure model, temperature, throttle, and output paths via CLI. The
    cleaned data is translated only when OpenAI translation is enabled.

- `scripts/messages_summary.py`
  - Snakemake-friendly script that reads the cleaned CSV and writes JSON
    metadata for quick QC.

- `scripts/pipeline_shell.py`
  - Utility for quickly dropping into an exploratory Python session with
    pipeline artefacts loaded as pandas DataFrames.

## Data Schema (core fields)

- `datetime` – Timestamp derived from `session_id`.
- `whatsapp_id` – User identifier parsed from `session_id`.
- `user_msg` – Original user message (mixed languages).
- `user_msg_en` – English translation (available in the OpenAI-translated CSV).
- `user_msg_category` – Label assigned to the user message.
- `ai_msg` / `ai_msg_en` – Chatbot response in original language / English.
- `cal_state`, `cal_feedback` – Conversation metadata.

---

Extend the workflow by editing the `Snakefile`: add new rules with explicit
inputs/outputs, and Snakemake will automatically infer scheduling, caching, and
visualisation. The Makefile shortcuts and helper scripts will continue to work
without modification.
