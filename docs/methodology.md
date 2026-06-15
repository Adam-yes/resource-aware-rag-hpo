# Methodology

This project evaluates Naive-RAG pipelines for technical knowledge access in manufacturing documentation.

The optimization target is a full pipeline configuration, not a single isolated parameter. Each genotype encodes:

- chunk size;
- chunk overlap;
- retrieval depth (`top_k`);
- generation temperature;
- local model choice.

The phenotype is evaluated by executing the complete RAG pipeline over a fixed evaluation set and scoring generated answers against references with Ragas factual correctness in F1 mode.

## Optimization Loop

The genetic algorithm uses:

- a population of 42 individuals (verified against `hpo_history.csv`);
- tournament selection, uniform crossover, and elitism (a hall-of-fame elite carried forward);
- structure-aware mutation: single-step nudges for ordinal genes, random re-selection for the
  categorical model gene;
- early stopping when the relative improvement in *average* population fitness between two
  consecutive generations (Delta-mu) falls below 5%.

The published run terminates after six generations (0-5), when Delta-mu first reaches 4.57%.

## Inference And Evaluation

Generation and judging both run on local Ollama models with a **fixed** context window
(`num_ctx = 5120`), `num_predict = 1024`, and pinned seeds. The judge is `qwen3-coder:30b` at
temperature 0. Fitness is the mean RAGAS FactualCorrectness (F1) over the 50-question benchmark:
each answer and reference is decomposed into atomic facts, and the per-question F1 is the harmonic
mean of fact precision and recall. Invalid (NaN) scores are skipped from the mean (`nan_policy:
drop`), matching the original aggregation.

The implementation writes two linked logs:

- `hpo_history.csv` records generation-level candidates and fitness values.
- `evolution_computation_log.csv` records question-level generated answers, reference answers, retrieved contexts, timings, and evaluation metadata.

Both logs use the same genotype hash to support later joins.

