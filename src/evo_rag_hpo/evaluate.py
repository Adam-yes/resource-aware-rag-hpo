"""Ragas-based evaluation for candidate RAG configurations."""

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
from .runtime import configure_logging, ensure_equal_lengths, failure_fitness, metric_mean
from .schema import FACTUAL_CORRECTNESS_COLUMN, LOG_FIELDNAMES

logger = logging.getLogger(__name__)


async def run_async_evaluate(individual: list[int], config: dict[str, Any] | None = None) -> tuple[float]:
    """Evaluate one genotype and return its mean factual-correctness F1 score."""

    run_config = config or load_config()
    attempts = int(run_config["evaluation"]["failure_retries"]) + 1
    for attempt in range(1, attempts + 1):
        try:
            return await _run_async_evaluate_once(individual, run_config)
        except Exception as exc:  # noqa: BLE001 - candidate failures should not abort a multi-hour run.
            logger.exception("Candidate evaluation failed on attempt %s/%s for %s.", attempt, attempts, individual)
            if attempt >= attempts:
                _log_failed_candidate(individual, run_config, exc)
                return failure_fitness(run_config)
    return failure_fitness(run_config)


async def _run_async_evaluate_once(individual: list[int], config: dict[str, Any]) -> tuple[float]:
    """Evaluate one genotype once, raising on measurement-contract violations."""

    import pandas as pd
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from ragas import EvaluationDataset, aevaluate
    from ragas.metrics import FactualCorrectness
    from ragas.run_config import RunConfig

    decoded = decode_individual(individual, config)
    ind_hash = genotype_hash(individual)

    evaluation_path = resolve_project_path(config["paths"]["evaluation_dataset"])
    reference_df = pd.read_csv(evaluation_path)

    start = time.time()
    rag_results = await run_async_rag_chain(decoded, config)
    rag_time = time.time() - start
    ensure_equal_lengths(rag_results, reference_df["reference"])

    dataset = EvaluationDataset.from_list(
        [
            {"reference": ref, "response": result["answer"].content}
            for result, ref in zip(rag_results, reference_df["reference"])
        ]
    )

    judge_llm = ChatOllama(
        model=config["models"]["judge"],
        temperature=0,
        num_ctx=config["inference"]["max_num_ctx"],
        keep_alive=config["inference"]["llm_keep_alive"],
        seed=config["optimization"]["random_seed"],
    )
    _ = OllamaEmbeddings(model=config["models"]["embedding"], keep_alive=config["inference"]["embedding_keep_alive"])
    factual_correctness = FactualCorrectness(mode="f1", atomicity="low", coverage="high", llm=judge_llm)

    eval_cfg = config["evaluation"]
    eval_run_config = RunConfig(
        timeout=eval_cfg["timeout"],
        max_retries=eval_cfg["max_retries"],
        max_wait=eval_cfg["max_wait"],
        max_workers=eval_cfg["max_workers"],
        seed=config["optimization"]["random_seed"],
    )
    start = time.time()
    eval_results = await aevaluate(
        dataset,
        metrics=[factual_correctness],
        run_config=eval_run_config,
        token_usage_parser=simple_ollama_parser,
    )
    eval_time = time.time() - start

    eval_df = eval_results.to_pandas()
    log_path = resolve_project_path(config["paths"]["computation_log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

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

    return (metric_mean(eval_df, config["evaluation"]["nan_policy"]),)


async def run_async_aeavluate(individual: list[int]) -> tuple[float]:
    """Backward-compatible alias for the original misspelled function name."""

    warnings.warn(
        "run_async_aeavluate is deprecated; use run_async_evaluate instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return await run_async_evaluate(individual)


def _log_failed_candidate(individual: list[int], config: dict[str, Any], exc: Exception) -> None:
    decoded = decode_individual(individual, config)
    log_path = resolve_project_path(config["paths"]["computation_log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = {field: "" for field in LOG_FIELDNAMES}
    row.update(
        {
            "Hash Id": genotype_hash(individual),
            "chunk_size": decoded["chunk_size"],
            "chunk_overlap": decoded["chunk_overlap"],
            "top_k": decoded["top_k"],
            "temperature": decoded["temperature"],
            "model_name": decoded["model_name"],
            "f1_score": config["evaluation"]["failed_candidate_fitness"],
            "source_file": f"candidate_failure:{type(exc).__name__}",
        }
    )
    with log_path.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=LOG_FIELDNAMES)
        if csvfile.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
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
