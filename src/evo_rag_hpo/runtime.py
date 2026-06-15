"""Small, dependency-light runtime helpers shared across the optimization modules.

These utilities are deliberately free of heavy imports (LangChain, RAGAS, DEAP) so they can be
unit-tested and reused without spinning up the full inference stack: logging configuration,
deterministic seeding, length validation, and metric aggregation.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from .schema import FACTUAL_CORRECTNESS_COLUMN

logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging for the command-line entry points.

    Args:
        level: A logging level name (e.g. ``"INFO"``, ``"DEBUG"``). Unknown names fall back to
            ``INFO`` so a typo in configuration never silences logging entirely.
    """

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s:%(name)s:%(message)s",
    )


def set_deterministic_seed(seed: int) -> None:
    """Seed the local RNGs used by the genetic algorithm.

    Seeds Python's ``random`` (population init, selection, crossover, mutation) and, when
    available, NumPy's RNG. LLM inference determinism is governed separately by the per-model
    ``seed``/``temperature`` settings and the Ollama backend.
    """

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        logger.debug("NumPy is not installed; skipped NumPy RNG seeding.")


def ensure_equal_lengths(results: list[Any], references: Any) -> None:
    """Assert that generated results and references align one-to-one.

    Pairing answers with references via ``zip`` would silently truncate to the shorter sequence
    if their lengths diverged, corrupting every downstream score. Failing fast here turns a
    subtle data bug into an explicit, debuggable error.

    Raises:
        ValueError: If the two collections differ in length.
    """

    reference_count = len(references)
    result_count = len(results)
    if result_count != reference_count:
        raise ValueError(f"Evaluation length mismatch: {result_count} RAG results for {reference_count} references.")


def metric_mean(eval_df: Any, nan_policy: str) -> float:
    """Aggregate the per-question factual-correctness column into a single fitness scalar.

    Args:
        eval_df: The RAGAS results frame (or any object exposing ``columns`` and column access).
        nan_policy: How to treat invalid (NaN) per-question scores:

            * ``"drop"`` - exclude NaNs from the mean. Mirrors ``pandas.Series.mean`` and
              reproduces the original experiment, where 2 of ~7,600 evaluations were invalid.
            * ``"zero"`` - replace NaNs with 0 before averaging (penalizes failures).
            * ``"raise"`` - treat any NaN as a hard error.

    Returns:
        The mean factual-correctness F1 score across the scorable questions.

    Raises:
        KeyError: If the expected metric column is missing (a measurement-contract violation).
        ValueError: If, after applying the policy, there are no scorable values, or if
            ``nan_policy="raise"`` and a NaN is present.
    """

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
