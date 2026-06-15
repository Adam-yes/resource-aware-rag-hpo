"""Score a candidate RAG configuration with RAGAS factual correctness (the fitness signal).

This module defines the *fitness function* of the genetic algorithm. For one individual it:

1. decodes the genotype into concrete hyperparameters,
2. runs the Naive-RAG pipeline over the full 50-question benchmark,
3. evaluates each generated answer against its reference answer with the RAGAS
   ``FactualCorrectness`` metric in F1 mode, judged by a fixed local LLM (``qwen3-coder:30b``
   at temperature 0 for deterministic, reproducible verdicts), and
4. appends a detailed per-question record to the computation log and returns the mean F1 score
   as the individual's fitness.

Factual correctness decomposes both the generated and the reference answer into atomic facts
and computes the F1 score over their overlap. Defining fitness on the *generated answer* -
rather than on retrieval statistics - aligns the optimization target with the quality a human
operator actually experiences.

Reproduction notes
------------------
* The judge runs with the **fixed** ``inference.num_ctx`` context window (5120 in the original
  study), matching the generation side. The judge is a measurement instrument, so its
  configuration is held constant across all candidates.
* Fitness is the **NaN-skipping mean** of the per-question F1 scores (``nan_policy: drop``). In
  the original run only 2 of ~7,600 question evaluations returned an invalid score; skipping
  those NaNs - exactly what ``pandas.Series.mean`` does by default - reproduces the published
  aggregate fitness.
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
import warnings
from typing import Any

from .config import decode_individual, genotype_hash, load_config, resolve_project_path
from .logger import simple_ollama_parser
from .rag_chain_pipeline import run_async_rag_chain
from .runtime import configure_logging, ensure_equal_lengths, metric_mean
from .schema import FACTUAL_CORRECTNESS_COLUMN, LOG_FIELDNAMES

logger = logging.getLogger(__name__)


async def run_async_evaluate(individual: list[int], config: dict[str, Any] | None = None) -> tuple[float]:
    """Evaluate one genotype and return its fitness as a single-element tuple.

    DEAP expects fitness values to be tuples (it supports multi-objective optimization), so the
    scalar mean F1 score is wrapped in a one-tuple ``(score,)``.

    Args:
        individual: The genotype - a list of integer indices into the search space, in the order
            ``[chunk_size, chunk_overlap, top_k, temperature, model_name]``.
        config: The resolved configuration. Loaded from defaults when ``None``.

    Returns:
        ``(mean_factual_correctness_f1,)`` over the benchmark questions.
    """

    cfg = config or load_config()

    decoded = decode_individual(individual, cfg)
    ind_hash = genotype_hash(individual)

    import pandas as pd
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from ragas import EvaluationDataset, aevaluate
    from ragas.metrics import FactualCorrectness
    from ragas.run_config import RunConfig

    # Load the reference answers (and per-question metadata) once for this evaluation.
    evaluation_path = resolve_project_path(cfg["paths"]["evaluation_dataset"])
    reference_df = pd.read_csv(evaluation_path)

    # 1. Run the RAG pipeline over all questions and time it. ``rag_time`` is logged to support
    #    quality-versus-latency analysis after the search.
    start = time.time()
    rag_results = await run_async_rag_chain(decoded, cfg)
    rag_time = time.time() - start

    # Fail fast if the pipeline did not return one answer per reference; silently zipping
    # mismatched lengths would corrupt the alignment between answers and references.
    ensure_equal_lengths(rag_results, reference_df["reference"])

    # 2. Assemble the RAGAS evaluation dataset by pairing each generated answer with its
    #    reference answer in benchmark order.
    dataset = EvaluationDataset.from_list(
        [
            {"reference": ref, "response": result["answer"].content}
            for result, ref in zip(rag_results, reference_df["reference"])
        ]
    )

    # 3. Configure the judge. Held constant across candidates: fixed model, temperature 0, fixed
    #    context window, pinned seed -> deterministic, comparable verdicts.
    inference = cfg["inference"]
    judge_llm = ChatOllama(
        model=cfg["models"]["judge"],
        temperature=0,
        num_ctx=inference["num_ctx"],
        keep_alive=inference["llm_keep_alive"],
        seed=cfg["optimization"]["random_seed"],
    )
    # The embedding model is instantiated to keep it warm for the metric backend; the F1
    # factual-correctness metric itself is driven by the judge LLM.
    _ = OllamaEmbeddings(model=cfg["models"]["embedding"], keep_alive=inference["embedding_keep_alive"])
    factual_correctness = FactualCorrectness(mode="f1", atomicity="low", coverage="high", llm=judge_llm)

    eval_cfg = cfg["evaluation"]
    eval_run_config = RunConfig(
        timeout=eval_cfg["timeout"],
        max_retries=eval_cfg["max_retries"],
        max_wait=eval_cfg["max_wait"],
        max_workers=eval_cfg["max_workers"],
        seed=cfg["optimization"]["random_seed"],
    )

    # 4. Run the RAGAS evaluation and time it. ``simple_ollama_parser`` lets RAGAS account for
    #    Ollama token usage, which differs from the OpenAI-style metadata RAGAS expects.
    start = time.time()
    eval_results = await aevaluate(
        dataset,
        metrics=[factual_correctness],
        run_config=eval_run_config,
        token_usage_parser=simple_ollama_parser,
    )
    eval_time = time.time() - start

    eval_df = eval_results.to_pandas()

    # 5. Persist one detailed row per question. The genotype hash links these per-question
    #    records to the per-generation rows written by the optimization loop, enabling the later
    #    join in the analysis notebooks. Logging failures must never abort the multi-hour search.
    log_path = resolve_project_path(cfg["paths"]["computation_log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=LOG_FIELDNAMES)
            if csvfile.tell() == 0:
                writer.writeheader()
            for idx, row_eval in eval_df.iterrows():
                row_ref = reference_df.iloc[idx]
                row_rag = rag_results[idx]
                writer.writerow(
                    {
                        "Hash Id": ind_hash,
                        "chunk_size": decoded["chunk_size"],
                        "chunk_overlap": decoded["chunk_overlap"],
                        "top_k": decoded["top_k"],
                        "temperature": decoded["temperature"],
                        "model_name": decoded["model_name"],
                        "time_run_async_rag_chain": rag_time,
                        "time_aevaluate": eval_time,
                        "usage_metadata_evaluation": eval_results.total_tokens(),
                        "question": row_rag["question"],
                        "generated_answer": row_rag["answer"].content,
                        "reference_answer": row_ref["reference"],
                        "f1_score": row_eval.get(FACTUAL_CORRECTNESS_COLUMN, 0),
                        "reference_contexts": row_ref.get("reference_contexts", ""),
                        "retrieved_contexts": row_rag["context"],
                        "usage_metadata_question": getattr(row_rag["answer"], "usage_metadata", {}),
                        "response_metadata_question": getattr(row_rag["answer"], "response_metadata", {}),
                        "persona_name": row_ref.get("persona_name", ""),
                        "query_style": row_ref.get("query_style", ""),
                        "query_length": row_ref.get("query_length", ""),
                        "synthesizer_name": row_ref.get("synthesizer_name", ""),
                        "source_file": row_ref.get("source_file", ""),
                    }
                )
    except Exception as exc:  # noqa: BLE001 - logging must not crash the optimization run.
        logger.error("CRITICAL LOGGING ERROR: %s", exc)

    # 6. Fitness = mean factual-correctness F1 over the benchmark, skipping invalid scores.
    return (metric_mean(eval_df, eval_cfg["nan_policy"]),)


async def run_async_aeavluate(individual: list[int]) -> tuple[float]:
    """Deprecated alias preserving the original (misspelled) function name.

    The original experiment exposed this entry point as ``run_async_aeavluate``. The alias is
    kept so external scripts and notebooks keep working; new code should call
    :func:`run_async_evaluate`.
    """

    warnings.warn(
        "run_async_aeavluate is deprecated; use run_async_evaluate instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return await run_async_evaluate(individual)


def main() -> None:
    """Command-line entry point to evaluate a single genotype, e.g. for a smoke test.

    Example: ``python -m evo_rag_hpo.evaluate 1 2 6 0 3``
    """

    parser = argparse.ArgumentParser(description="Evaluate one RAG HPO genotype.")
    parser.add_argument("genes", nargs="*", type=int, default=[1, 2, 6, 0, 3])
    parser.add_argument("--config", default=None, help="Path to a YAML configuration file.")
    args = parser.parse_args()

    import asyncio

    config = load_config(args.config)
    configure_logging(config["logging"]["level"])
    result = asyncio.run(run_async_evaluate(args.genes, config))
    print(result)


if __name__ == "__main__":
    main()
