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

- tournament selection;
- uniform crossover;
- mixed mutation for ordinal and categorical dimensions;
- elitism;
- early stopping when average-fitness improvement falls below the configured threshold.

The implementation writes two linked logs:

- `hpo_history.csv` records generation-level candidates and fitness values.
- `evolution_computation_log.csv` records question-level generated answers, reference answers, retrieved contexts, timings, and evaluation metadata.

Both logs use the same genotype hash to support later joins.

