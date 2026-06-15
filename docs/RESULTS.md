# Results

## Headline Result

The evolutionary search improved average fitness by approximately **80%** over the initial random population while evaluating **152 unique configurations**, corresponding to **more than 99% fewer evaluations** than exhaustive grid search over the defined space.

The manager-level takeaway: systematic resource-aware search can materially improve RAG quality while exposing deployment-relevant latency and model-size trade-offs.

## Publicly Documented Metrics

| Metric | Value | Source |
| --- | ---: | --- |
| Average fitness increase | ~80% | manuscript/README summary |
| Unique configurations evaluated | 152 | manuscript/README summary |
| Evaluation reduction vs. grid search | >99% | manuscript/README summary |
| Compact-model quality | 86% of maximum observed quality | manuscript/README summary |
| Compact-model inference time | 7% of best-performing configuration time | manuscript/README summary |

## Best Configuration

The full best-configuration table will be filled from the full artifact release.

| Field | Value |
| --- | --- |
| chunk_size | TODO: fill from full artifact |
| chunk_overlap | TODO: fill from full artifact |
| top_k | TODO: fill from full artifact |
| temperature | TODO: fill from full artifact |
| model_name | TODO: fill from full artifact |
| FactualCorrectness F1 | TODO: fill from full artifact |
| latency | TODO: fill from full artifact |
| baseline comparison | TODO: fill from full artifact |
| population size | 42 |
| max generations | 7 |

## Public Samples

The public repository includes lightweight samples in `results/samples/` to document schema and analysis shape. Full logs are intentionally excluded until artifact release because they are large and tied to licensed source documents.

