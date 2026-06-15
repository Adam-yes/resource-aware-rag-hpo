# ADR 0004: Prioritize Local Ollama Inference

## Context

Manufacturing deployments can face privacy, network, and data-residency constraints.

## Decision

Use local Ollama models for generation, embeddings, and judge-model execution.

## Consequences

The system better reflects on-prem constraints and cost predictability. It also requires explicit runtime configuration for context windows, timeouts, warm model lifetimes, and failure handling.

