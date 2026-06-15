# Design: Resource-Aware RAG Hyperparameter Optimization

## Leadership Framing

This project treats RAG configuration as a production engineering problem, not only a modeling exercise. In manufacturing settings, the best configuration is rarely the largest model or the deepest retrieval setting. The useful configuration is the one that balances factual answer quality with latency, memory footprint, local deployment constraints, and repeatability.

The central design choice is therefore to optimize a complete Naive-RAG pipeline under resource constraints. The search evaluates the interaction between chunking, retrieval depth, decoding, and model choice instead of tuning each parameter in isolation.

## Why Evolutionary Search

The search space mixes ordinal values (`chunk_size`, `top_k`, `temperature`) and categorical values (`model_name`). Exhaustive grid search is transparent but expensive. Bayesian optimization can be strong, but it adds modeling assumptions and operational complexity that are harder to explain and reproduce for this bounded research artifact.

A genetic algorithm is a pragmatic middle ground:

- it handles mixed discrete/categorical search spaces naturally;
- it exposes understandable trade-offs to stakeholders;
- it can stop early when marginal improvement no longer justifies compute.

## Reproduction Fidelity As A Design Constraint

This is a research artifact, so it reproduces the manuscript's computational behavior before it
optimizes for anything else. The chunking splitter, the prompt, the fixed context window, the
no-cache evaluation loop, and the average-fitness (Delta-mu) early-stopping rule are all pinned to
the published behavior, even where a production system might choose differently. Engineering
additions (packaging, central configuration, validation, tests, CI) are kept only because they
preserve behavior. The decisions — including changes reverted to protect reproduction — are listed
in [improvement-audit.md](improvement-audit.md).

## Why Local Ollama Models

Manufacturing environments often have constraints around data residency, network access, cost predictability, and operational privacy. Local Ollama inference makes those constraints visible. It also lets the study compare compact and larger open-weight models under a realistic deployment assumption: not every useful RAG system can rely on a remote proprietary API.

The trade-off is operational: local models require explicit management of context windows, VRAM, timeouts, and model warm-up behavior. Those constraints are not incidental; they are part of the resource-aware thesis.

## Quality, Latency, And Compute Cost

The system evaluates factual correctness as the quality signal, but quality alone is not the whole engineering decision. A Tech Lead deciding whether to deploy such a pipeline would also ask:

- How much latency does an incremental quality gain add?
- Does a larger model justify its memory and runtime cost?
- Does higher `top_k` improve factuality or only inflate context and truncation risk?
- Can the system recover from failed candidates during a long optimization run?

This repository should make those trade-offs explicit in configuration, logs, and documentation.

## Known Limitations

- Public artifacts do not include third-party manuals or full experiment logs.
- Full reproduction requires local Ollama models and licensed source documents.
- FactualCorrectness is useful but incomplete as a single metric; answer utility, latency, cost, and retrieval faithfulness should eventually be combined.
- Local LLM runtimes are not perfectly deterministic even with configured seeds.

## What I Would Do Next

These are explicitly *forward-looking* directions for a production successor, kept separate from
the reproduction default so they never silently alter the published results:

- Add a multi-objective fitness that explicitly combines quality, latency, and compute budget.
- Offer an opt-in, off-by-default fitness cache and candidate-failure policy for long runs, with
  their effect on the logs documented.
- Add richer observability for candidate failures, timeouts, and context truncation.
- Publish full artifacts through Zenodo after paper clearance.
- Render key notebooks into static documentation pages for review without notebook execution.

