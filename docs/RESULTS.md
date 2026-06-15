# Results

This page is the durable, paper-grounded summary of the study's outcomes. Every figure below is
taken directly from the manuscript and its committed experiment logs (`results/experiments/`),
not estimated. Full per-question logs are released separately (see
[data-availability.md](data-availability.md)).

## Headline Result

The evolutionary search increased **average population fitness by approximately 80%** over the
initial random population (0.2239 -> 0.4025 across six generations) while evaluating only
**152 unique configurations** - **more than 99% fewer** than an exhaustive grid search over the
defined space (which contains roughly 61,000 configurations). The complete run took about
**30 hours** on the reported Apple M4 Pro hardware.

The leadership takeaway: a guided, resource-aware search materially improves RAG answer quality
*and* surfaces the quality-versus-latency operating points a team needs before deploying locally.

## Convergence

Average fitness rises monotonically while maximum fitness stays flat through the early phase and
lifts only in the final two generations - the search concentrates the whole population on
stronger regions rather than locking onto one early winner. The run stops after generation 5,
when the average-fitness improvement (Delta-mu) first drops below the 5% threshold (4.57%).

| Generation | Evaluations | Avg. fitness | Max. fitness |
| ---: | ---: | ---: | ---: |
| 0 | 42 | 0.2239 | 0.4148 |
| 1 | 33 | 0.2987 | 0.4148 |
| 2 | 32 | 0.3185 | 0.4148 |
| 3 | 35 | 0.3460 | 0.4148 |
| 4 | 32 | 0.3849 | 0.4346 |
| 5 | 29 | 0.4025 | 0.4460 |

> Evaluations sum to 203 across the run for 152 *unique* configurations; repeated genomes are
> re-evaluated rather than cached, matching the original experiment.

## Best Configuration

| Field | Value |
| --- | --- |
| chunk_size | 1024 |
| chunk_overlap | 25% |
| top_k | 10 |
| temperature | 0.3 |
| model_name | qwen3-vl:30b-a3b-instruct |
| FactualCorrectness F1 (fitness) | 0.4460 |
| Latency | 75.93 s |
| population size | 42 |
| max generations | 7 (early-stopped at generation 5) |

## Quality-Latency Trade-off

The highest-scoring configuration is also the slowest. Crucially, a compact model
(`qwen3-vl:2b`) reaches **86% of the best observed quality at roughly 7% of its latency** - the
practically relevant result for resource-constrained deployment.

| Model | Fitness (F1) | Latency (s) |
| --- | ---: | ---: |
| qwen3-vl:30b-a3b-instruct | 0.4460 | 75.93 |
| qwen3-vl:2b | 0.3836 | 5.42 |
| qwen3-coder:30b | 0.3692 | 48.07 |
| gemma3n:e2b | 0.3628 | 52.44 |
| granite3.3:2b | 0.3550 | 22.09 |

## Parameter Influence

Feature-importance (Random Forest) and correlation analyses agree on the ranking of influence:

- **Top-k** is the strongest single factor (correlation with fitness r = 0.53).
- **Model architecture** and **chunk size** follow (chunk size r = 0.51).
- **Temperature** is weakly *negative* (r = -0.14); **chunk overlap** contributes little (r = 0.10).

Across generations the search concentrates on larger chunk sizes (768-1024 by the final
generation), moderate-to-high Top-k, and lower temperatures - consistent with the best
configuration above.

## Measurement Reliability

Out of approximately 7,600 evaluated question-answer pairs (152 configurations x 50 questions),
only two returned an invalid score - a failure rate below 0.03%, both on smaller models and
attributed to evaluator parsing errors. These NaN scores are excluded from the per-configuration
mean (`nan_policy: drop`), exactly as in the original run.

## Public Samples

The repository includes lightweight samples in `results/samples/` to document the log schema and
analysis shape. Full logs are excluded until artifact release because they are large and tied to
licensed source documents.

## Figures

Selected publication figures are in `results/figures/`. See the [README](../README.md#paper-figures)
for the figure index.
