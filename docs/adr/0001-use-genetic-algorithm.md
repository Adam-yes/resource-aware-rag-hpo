# ADR 0001: Use A Genetic Algorithm For Mixed RAG Hyperparameter Search

## Context

The search space combines ordinal parameters, categorical model choices, and expensive end-to-end evaluations.

## Decision

Use a DEAP-based genetic algorithm with elitism, mutation, crossover, caching, and configurable early stopping.

## Consequences

The search is explainable and handles mixed discrete spaces naturally. It is not guaranteed to find a global optimum, so logs, seeds, and stopping criteria must remain transparent.

