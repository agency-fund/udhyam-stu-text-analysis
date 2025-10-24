import shlex

configfile: "config/pipeline.yaml"

RAW_DATA = config["raw_data"]
ENABLE_OPENAI = config.get("enable_openai_translation", True)
ENABLE_OPENAI_TOPIC_REPR = config.get("enable_openai_topic_representation", False)
OPENAI_TOPIC_MODEL = config.get("openai_topic_model", "gpt-4o-mini")
OPENAI_TOPIC_MODEL_ARG = shlex.quote(OPENAI_TOPIC_MODEL)

ALL_OUTPUTS = [
    "data/cleaned/messages_cleaned.csv",
    "data/analysis/datetime_overview.json",
    "data/analysis/cal_state_by_user.csv",
    "data/analysis/message_stats.csv",
    "data/analysis/sentiment_overview.csv",
    "data/analysis/sentiment_user.csv",
    "data/analysis/sentiment_ai.csv",
    "data/analysis/sentiment_all.csv",
    "data/analysis/topic_keywords.csv",
    "data/analysis/topic_info_user.csv",
    "data/analysis/topic_info_assistant.csv",
    "data/analysis/message_topics_user.csv",
    "data/analysis/message_topics_assistant.csv",
    "data/analysis/agency_user.parquet",
    "data/analysis/agency_assistant.parquet",
    "docs/index.html",
    "docs/pipeline_graphs/pipeline_dag.png",
]

if ENABLE_OPENAI:
    ALL_OUTPUTS.append("data/cleaned/messages_translated.csv")
    
    rule openai_translate:
        input:
            cleaned="data/cleaned/messages_cleaned.csv"
        output:
            translated="data/cleaned/messages_translated.csv"
        shell:
            (
                "python scripts/02_translate_messages.py "
                "--input {input.cleaned:q} --output {output.translated:q}"
            )


rule all:
    input:
        ALL_OUTPUTS


rule clean_messages:
    input:
        raw=RAW_DATA
    output:
        cleaned="data/cleaned/messages_cleaned.csv"
    shell:
        (
            "python scripts/01_load_and_clean.py "
            "--input {input.raw:q} --output {output.cleaned:q}"
        )


rule generate_summary:
    input:
        translated="data/cleaned/messages_translated.csv"
    output:
        datetime_json="data/analysis/datetime_overview.json",
        cal_state_csv="data/analysis/cal_state_by_user.csv",
        message_stats="data/analysis/message_stats.csv"
    shell:
        (
            "python scripts/03_generate_summary.py "
            "--input {input.translated:q} --datetime-summary {output.datetime_json:q} "
            "--cal-state-summary {output.cal_state_csv:q} --message-stats {output.message_stats:q}"
        )


rule sentiment_analysis:
    input:
        translated="data/cleaned/messages_translated.csv"
    output:
        overview="data/analysis/sentiment_overview.csv",
        user="data/analysis/sentiment_user.csv",
        assistant="data/analysis/sentiment_ai.csv",
        combined="data/analysis/sentiment_all.csv"
    shell:
        (
            "python scripts/04_sentiment_analysis.py "
            "--input {input.translated:q}"
        )


rule topic_modeling:
    input:
        translated="data/cleaned/messages_translated.csv"
    params:
        openai_args=(
            f" --use-openai-representation --openai-model {OPENAI_TOPIC_MODEL_ARG}"
            if ENABLE_OPENAI_TOPIC_REPR
            else ""
        )
    output:
        keywords="data/analysis/topic_keywords.csv",
        user_topics="data/analysis/topic_info_user.csv",
        assistant_topics="data/analysis/topic_info_assistant.csv",
        user_assignments="data/analysis/message_topics_user.csv",
        assistant_assignments="data/analysis/message_topics_assistant.csv"
    shell:
        (
            "python scripts/05_topic_modeling.py "
            "--input {input.translated:q}{params.openai_args}"
        )


rule agency_scoring:
    input:
        translated="data/cleaned/messages_translated.csv"
    output:
        user="data/analysis/agency_user.parquet",
        assistant="data/analysis/agency_assistant.parquet"
    shell:
        (
            "python scripts/06_agency_scoring.py "
            "--input {input.translated:q} "
            "--output-user {output.user:q} "
            "--output-assistant {output.assistant:q}"
        )


rule render_report:
    input:
        report="docs/udhyam_stu_text_analysis_report.qmd",
        agency_user="data/analysis/agency_user.parquet",
        agency_assistant="data/analysis/agency_assistant.parquet"
    output:
        html="docs/index.html"
    shell:
        (
            "python scripts/07_render_report.py "
            "--input {input.report:q} --output {output.html:q}"
        )


rule pipeline_dag:
    input:
        snakefile="Snakefile"
    output:
        dag="docs/pipeline_graphs/pipeline_dag.png"
    shell:
        (
            "mkdir -p docs/pipeline_graphs && "
            "snakemake --snakefile {input.snakefile:q} --cores 1 --dag | "
            "dot -Tpng > {output.dag:q}"
        )
