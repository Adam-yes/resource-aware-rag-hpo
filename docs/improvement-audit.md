# Engineering Decisions and Reproduction-Fidelity Audit

This document is intentionally candid. The repository has one prime directive: it must enable a
**faithful 1:1 reproduction** of the accompanying manuscript. Every engineering decision is made
in service of that goal first, and presentation second.

A useful distinction runs through this audit: the difference between a *faithful research
reproduction* and a *production refactor*. Both are legitimate, but they optimize for different
things. A reproduction must preserve the exact computational behavior that produced the published
numbers. A production refactor is free to change behavior to improve robustness or efficiency.
Conflating the two is a common and costly mistake - a change that looks like a bug fix can
silently invalidate a result. This repository chooses fidelity, and documents where that choice
overrides otherwise reasonable "improvements".

## What The Engineering Layer Keeps

The repository wraps the original research code in a maintainable, reviewable package **without
changing what it computes**. These additions are behavior-preserving and are kept:

- A `src/`-layout installable package (`evo_rag_hpo`) with three CLI entry points.
- A single source of truth for configuration (`configs/default.yaml` + `DEFAULT_CONFIG`) with
  explicit validation and clear, early error messages.
- Lazy imports of the heavy stack (LangChain, RAGAS, DEAP) so the package imports cheaply and is
  unit-testable without a GPU or local models.
- Deterministic seeding, an explicit answer/reference length guard, and a metric-column contract
  check - none of which alter the success-path result, but all of which make failures legible.
- A test suite (config validation, genotype contracts, mutation bounds, metric policy, logging
  schema, and an early-stopping reproduction test), Ruff lint/format, and a CI gate that installs
  the real dependencies and runs `pytest` with coverage.
- English, didactic comments and docstrings throughout, translated and expanded from the original.

## What Was Reverted To Preserve Fidelity

An earlier refactor introduced several changes that, while defensible as production engineering,
**alter the computation** and therefore break a 1:1 reproduction. Each was reverted, and the
faithful behavior is now the default. They remain documented here so the decision is auditable.

| Area | Reverted change | Why it breaks reproduction | Faithful behavior (now default) |
| --- | --- | --- | --- |
| Chunking | Recursive character splitter | Changes every chunk boundary, hence all retrieval results | `MarkdownTextSplitter` (as published) |
| Context window | Dynamic `num_ctx` from `chunk_size * top_k`, judge at 16384 | Varies generation/judging behavior across candidates | Fixed `num_ctx = 5120` for generation and judging |
| Early stopping | Monitor `max` fitness with patience 2 | Stops on a different generation than the paper | Monitor `avg` (Delta-mu), stop on first < 5% (six generations) |
| Evaluation count | Global fitness-archive cache | Skips re-evaluation of repeated genomes (203 -> 152) | No cache; repeated genomes re-evaluated, matching the logs |
| NaN handling | `nan_policy: zero` (penalize failures with 0) | Changes the aggregate fitness when an evaluation is invalid | `nan_policy: drop` (pandas-default skip), matching the run |
| Prompt | Condensed/re-cased instruction template | Changes model output, hence the F1 scores | Original prompt reproduced verbatim |

The early-stopping behavior is locked in by an integration test that replays the manuscript's
average-fitness sequence and asserts the loop terminates after generation 5.

## Honest Provenance Notes

- **GA operator probabilities** (crossover 0.7, mutation 0.4, per-gene mutation 0.2, tournament
  size 3) are not separately tabulated in the manuscript. The population size (42) and the
  stopping criterion are verified against the committed `hpo_history.csv`. The operator
  probabilities follow the reference GA implementation cited in the paper (Wirsansky,
  *Hands-On Genetic Algorithms with Python*) and are exposed in configuration so a reproducer can
  pin or vary them deliberately. They are labeled as configuration choices, not paper claims.
- **Determinism** is best-effort: seeds are fixed everywhere they can be, but local LLM inference
  through Ollama is not bit-for-bit reproducible, so exact fitness values may vary slightly while
  the methodology and aggregate behavior reproduce.

## Robustness Extras That Were Deliberately Not Added Back

Graceful candidate-failure retries and a persistent fitness cache are reasonable for a long
production run, but they change the logged output and evaluation counts. They are intentionally
**out of scope** for the reproduction default. If reintroduced, they belong behind explicit,
off-by-default configuration flags, with their effect on the logs documented.
