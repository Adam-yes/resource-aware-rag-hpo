"""Runtime helpers for context sizing, validation, and deterministic execution."""

from __future__ import annotations

import logging
import random
from typing import Any

from .schema import FACTUAL_CORRECTNESS_COLUMN

logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    """Configure module logging for command-line entrypoints."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s:%(name)s:%(message)s",
    )


def set_deterministic_seed(seed: int) -> None:
    """Seed supported local RNGs from the configured experiment seed."""

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        logger.debug("NumPy is not installed; skipped NumPy RNG seeding.")


def calculate_num_ctx(params: dict[str, Any], config: dict[str, Any]) -> int:
    """Estimate and cap the Ollama context window needed for one RAG candidate."""

    inference = config["inference"]
    retrieved_tokens = int(params["chunk_size"] * params["top_k"])
    estimated = retrieved_tokens + int(inference["prompt_token_headroom"]) + int(inference["answer_token_budget"])
    min_ctx = int(inference["min_num_ctx"])
    max_ctx = int(inference["max_num_ctx"])
    num_ctx = max(min_ctx, min(estimated, max_ctx))

    if estimated > max_ctx:
        logger.warning(
            "Estimated context need %s exceeds configured max_num_ctx=%s for params=%s; using capped context.",
            estimated,
            max_ctx,
            params,
        )
    return num_ctx


def ensure_equal_lengths(results: list[Any], references: Any) -> None:
    """Fail fast when generated responses and references do not align."""

    reference_count = len(references)
    result_count = len(results)
    if result_count != reference_count:
        raise ValueError(f"Evaluation length mismatch: {result_count} RAG results for {reference_count} references.")


def metric_mean(eval_df: Any, nan_policy: str) -> float:
    """Validate and aggregate the configured factual-correctness metric."""

    if FACTUAL_CORRECTNESS_COLUMN not in eval_df.columns:
        raise KeyError(f"Missing expected Ragas metric column: {FACTUAL_CORRECTNESS_COLUMN}")

    series = eval_df[FACTUAL_CORRECTNESS_COLUMN]
    if nan_policy == "zero":
        series = series.fillna(0)
    elif nan_policy == "drop":
        series = series.dropna()
    elif series.isna().any():
        raise ValueError(f"Metric column {FACTUAL_CORRECTNESS_COLUMN} contains NaN values.")

    if len(series) == 0:
        raise ValueError(f"Metric column {FACTUAL_CORRECTNESS_COLUMN} contains no scorable values.")
    return float(series.mean())


def failure_fitness(config: dict[str, Any]) -> tuple[float]:
    """Return the configured deterministic score for failed candidates."""

    return (float(config["evaluation"]["failed_candidate_fitness"]),)
