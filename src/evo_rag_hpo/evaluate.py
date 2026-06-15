"""Ragas-based evaluation for candidate RAG configurations."""

from __future__ import annotations

import argparse
import csv
import time
from typing import Any

from .config import decode_individual, genotype_hash, load_config, resolve_project_path
from .logger import simple_ollama_parser
from .rag_chain_pipeline import run_async_rag_chain

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


async def run_async_evaluate(individual: list[int], config: dict[str, Any] | None = None) -> tuple[float]:
    """Evaluate one genotype and return its mean factual-correctness F1 score."""

    from langchain_ollama import ChatOllama, OllamaEmbeddings
    import pandas as pd
    from ragas import EvaluationDataset, aevaluate
    from ragas.metrics import FactualCorrectness
    from ragas.run_config import RunConfig

    run_config = config or load_config()
    decoded = decode_individual(individual, run_config)
    ind_hash = genotype_hash(individual)

    evaluation_path = resolve_project_path(run_config["paths"]["evaluation_dataset"])
    reference_df = pd.read_csv(evaluation_path)

    start = time.time()
    rag_results = await run_async_rag_chain(decoded, run_config)
    rag_time = time.time() - start

    dataset = EvaluationDataset.from_list(
        [
            {"reference": ref, "response": result["answer"].content}
            for result, ref in zip(rag_results, reference_df["reference"])
        ]
    )

    judge_llm = ChatOllama(
        model=run_config["models"]["judge"],
        temperature=0,
        num_ctx=5120,
        keep_alive="15m",
        seed=run_config["optimization"]["random_seed"],
    )
    _ = OllamaEmbeddings(model=run_config["models"]["embedding"])
    factual_correctness = FactualCorrectness(mode="f1", atomicity="low", coverage="high", llm=judge_llm)

    eval_run_config = RunConfig(timeout=720, max_retries=2, max_wait=30, max_workers=8, seed=42)
    start = time.time()
    eval_results = await aevaluate(
        dataset,
        metrics=[factual_correctness],
        run_config=eval_run_config,
        token_usage_parser=simple_ollama_parser,
    )
    eval_time = time.time() - start

    eval_df = eval_results.to_pandas()
    log_path = resolve_project_path(run_config["paths"]["computation_log"])
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
                    "f1_score": row_eval.get("factual_correctness(mode=f1)", 0),
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

    score = eval_df["factual_correctness(mode=f1)"].mean()
    return (float(score),)


async def run_async_aeavluate(individual: list[int]) -> tuple[float]:
    """Backward-compatible alias for the original misspelled function name."""

    return await run_async_evaluate(individual)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one RAG HPO genotype.")
    parser.add_argument("genes", nargs="*", type=int, default=[1, 2, 6, 0, 3])
    parser.add_argument("--config", default=None, help="Path to a YAML configuration file.")
    args = parser.parse_args()

    import asyncio

    result = asyncio.run(run_async_evaluate(args.genes, load_config(args.config)))
    print(result)


if __name__ == "__main__":
    main()
