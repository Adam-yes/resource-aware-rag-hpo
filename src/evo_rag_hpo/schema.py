"""Shared schema constants for experiment logs and evaluation metrics.

Centralizing the column names here keeps the log writer (:mod:`evo_rag_hpo.evaluate`), the
history writer (:mod:`evo_rag_hpo.elitism`), the metric aggregator
(:mod:`evo_rag_hpo.runtime`), and the analysis notebooks in agreement on a single source of
truth. A change to the RAGAS metric name or a log column only has to be made once.
"""

from __future__ import annotations

# Exact column name RAGAS emits for FactualCorrectness in F1 mode. Used to locate the
# per-question score in the results frame; a mismatch surfaces as an explicit KeyError.
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
