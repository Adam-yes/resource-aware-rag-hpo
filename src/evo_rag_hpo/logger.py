"""Token-usage parsing and small data-loading helpers for the evaluation pipeline.

Despite the module name, this file contains two pure helpers used across the pipeline rather
than logging configuration (which lives in :mod:`evo_rag_hpo.runtime`). The name is kept for
continuity with the original codebase.
"""

from __future__ import annotations


def simple_ollama_parser(result):
    """Extract total input/output token counts from a LangChain/Ollama LLM result for RAGAS.

    RAGAS tracks evaluation cost via a ``token_usage_parser``. Its built-in parsers expect the
    OpenAI-style usage object, whereas Ollama (through LangChain) attaches usage to each
    generation's ``message.usage_metadata``. This parser walks the nested ``result.generations``
    structure (a list of lists) and sums the per-message token counts so that RAGAS can report
    accurate token consumption for local models.

    Args:
        result: A LangChain ``LLMResult`` whose generations carry Ollama usage metadata.

    Returns:
        ragas.cost.TokenUsage: Aggregated input and output token totals.
    """

    from ragas.cost import TokenUsage

    input_tokens = 0
    output_tokens = 0
    for gen_list in result.generations:
        for gen in gen_list:
            # Guard against generations lacking a message or usage metadata so a single
            # malformed generation cannot crash scoring.
            if hasattr(gen, "message") and hasattr(gen.message, "usage_metadata"):
                usage = gen.message.usage_metadata or {}
                input_tokens += usage.get("input_tokens", 0)
                output_tokens += usage.get("output_tokens", 0)
    return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)


def load_column_as_list(csv_path: str, column: str) -> list[str]:
    """Load a single CSV column as a list of strings.

    Used to prepare batch inputs - for example the ``user_input`` (question) column fed to
    ``rag_chain.abatch`` - in the list form the batch APIs expect.

    Args:
        csv_path: Path to the CSV file.
        column: Name of the column to extract.

    Returns:
        The column values coerced to ``str``, in file order.
    """

    import pandas as pd

    df = pd.read_csv(csv_path)
    return df[column].astype(str).tolist()
