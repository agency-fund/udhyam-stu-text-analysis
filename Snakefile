configfile: "config/pipeline.yaml"

RAW_DATA = config["raw_data"]
ENABLE_OPENAI = config.get("enable_openai_translation", True)

ALL_OUTPUTS = [
    "data/cleaned/messages_cleaned.csv",
    "data/cleaned/messages_summary.json",
]

if ENABLE_OPENAI:
    ALL_OUTPUTS.append("data/cleaned/messages_translated.csv")


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


rule messages_summary:
    input:
        cleaned="data/cleaned/messages_cleaned.csv"
    output:
        summary="data/cleaned/messages_summary.json"
    script:
        "scripts/messages_summary.py"


if ENABLE_OPENAI:

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
