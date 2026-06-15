# ADR 0002: Use FactualCorrectness F1 As The Primary Fitness Signal

## Context

The task is technical-document question answering where factual grounding matters more than fluent but unsupported responses.

## Decision

Use Ragas `FactualCorrectness` in F1 mode as the primary automated quality metric.

## Consequences

The metric aligns with factual answer quality, but it is not a complete deployment objective. Latency, retrieval quality, failure rate, and compute cost must be tracked separately and may become a multi-objective score later.

