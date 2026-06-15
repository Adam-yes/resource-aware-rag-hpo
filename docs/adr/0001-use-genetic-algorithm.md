# ADR 0001: Use A Genetic Algorithm For Mixed RAG Hyperparameter Search

## Context

The search space combines ordinal parameters, categorical model choices, and expensive end-to-end evaluations.

## Decision

Use a DEAP-based genetic algorithm with elitism, structure-aware mutation, uniform crossover, and average-fitness (Delta-mu) early stopping. No fitness cache is used: repeated genomes are re-evaluated, matching the original experiment's evaluation counts.

## Consequences

The search is explainable and handles mixed discrete spaces naturally. It is not guaranteed to find a global optimum, so logs, seeds, and stopping criteria must remain transparent.

