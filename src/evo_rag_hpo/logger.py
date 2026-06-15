"""Logging and token-usage helpers."""

from __future__ import annotations


def simple_ollama_parser(result):
    """Parse token usage metadata from LangChain/Ollama generations for Ragas."""

    from ragas.cost import TokenUsage

    input_tokens = 0
    output_tokens = 0
    for gen_list in result.generations:
        for gen in gen_list:
            if hasattr(gen, "message") and hasattr(gen.message, "usage_metadata"):
                usage = gen.message.usage_metadata or {}
                input_tokens += usage.get("input_tokens", 0)
                output_tokens += usage.get("output_tokens", 0)
    return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)


def load_column_as_list(csv_path: str, column: str) -> list[str]:
    """Load a CSV column as a list of strings."""

    import pandas as pd

    df = pd.read_csv(csv_path)
    return df[column].astype(str).tolist()
