"""Shared schema constants for experiment logs and evaluation metrics."""

from __future__ import annotations

FACTUAL_CORRECTNESS_COLUMN = "factual_correctness(mode=f1)"

LOG_FIELDNAMES = [
    "Hash Id",
    "chunk_size",
    "chunk_overlap",
    "top_k",
    "temperature",
    "model_name",
    "time_run_async_rag_chain",
    "time_aevaluate",
    "usage_metadata_evaluation",
    "question",
    "generated_answer",
    "reference_answer",
    "f1_score",
    "reference_contexts",
    "retrieved_contexts",
    "usage_metadata_question",
    "response_metadata_question",
    "persona_name",
    "query_style",
    "query_length",
    "synthesizer_name",
    "source_file",
]

HPO_HISTORY_FIELDNAMES = ["Hash Id", "gen", "ind_id", "fitness", "params_list"]
