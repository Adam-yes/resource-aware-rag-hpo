# ADR 0003: Bound The Search Space To Deployable Local-RAG Configurations

## Context

The study compares chunking, retrieval depth, decoding temperature, and local model choice under resource constraints.

## Decision

Represent each candidate as `[chunk_size, chunk_overlap, top_k, temperature, model_name]`, with each gene indexing a configured finite search-space list.

## Consequences

The genotype is compact and reproducible. Any expansion of the search space should be documented because it changes grid-search size, optimization cost, and comparability with prior results.

