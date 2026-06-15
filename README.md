<p align="center">
  <img src="docs/assets/repository-header.svg" alt="Resource-Aware RAG HPO" width="100%">
</p>

<p align="center">
  <a href="https://github.com/Adam-yes/resource-aware-rag-hpo/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/Adam-yes/resource-aware-rag-hpo/ci.yml?branch=main&label=CI&style=flat-square"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.12-2F6BFF?style=flat-square">
  <img alt="License" src="https://img.shields.io/github/license/Adam-yes/resource-aware-rag-hpo?style=flat-square">
  <img alt="Ollama" src="https://img.shields.io/badge/Ollama-local%20LLMs-1F2933?style=flat-square">
  <img alt="Status" src="https://img.shields.io/badge/paper-under%20review-E0A13D?style=flat-square">
</p>

<p align="center">
  <b>Evolutionary hyperparameter optimization for resource-aware Naive-RAG pipelines in technical manufacturing knowledge access.</b>
</p>

<p align="center">
  <a href="#why-it-matters">Why</a> ·
  <a href="#results">Results</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#reproduce">Reproduce</a> ·
  <a href="docs/DESIGN.md">Design</a> ·
  <a href="docs/RESULTS.md">Full results</a>
</p>

---

## Why It Matters

Manufacturing teams depend on technical documentation for maintenance, troubleshooting, and operator support. Retrieval-augmented generation can make that knowledge accessible in natural language, but quality hinges on interacting choices across chunking, retrieval depth, model selection, and decoding — a mixed-integer space too large to tune by hand and too expensive to search exhaustively.

This project frames RAG configuration as an optimization problem. A genetic algorithm searches the joint hyperparameter space and scores each candidate end to end on factual correctness, surfacing the quality–latency trade-offs that matter before deploying a system on local hardware.

> Accompanies the under-review manuscript **"Resource-Aware Retrieval-Augmented Technical Knowledge Access for Smart Manufacturing: Evolutionary Configuration and Quality-Latency Trade-offs."**

Design rationale lives in [docs/DESIGN.md](docs/DESIGN.md); decision records in [docs/adr/](docs/adr/). The computational path is pinned to the published behavior — see the [reproduction-fidelity audit](docs/improvement-audit.md) for the exact decisions.

## Results

<p align="center">
  <img src="docs/assets/results-overview.svg" alt="Results overview" width="100%">
</p>

Evolutionary search raised average population fitness by **~80%** (0.224 → 0.403 over six generations) while evaluating only **152 unique configurations** — **>99% fewer** than an exhaustive grid. **Top-k**, **model architecture**, and **chunk size** dominate; a compact model reaches **86% of the best observed quality at ~7% of its latency**. Full numbers and tables: [docs/RESULTS.md](docs/RESULTS.md).

### Paper figures

| Figure | What it shows |
| --- | --- |
| [Convergence](results/figures/Konvergenz_Max_Avg.png) | Average and maximum fitness over generations |
| [Quality–latency](results/figures/max_fit_vs_latency.png) | Fitness versus inference time |
| [Feature importance](results/figures/Feature_Importance_Random_Forest.png) | Relative influence of core hyperparameters |
| [Interaction heatmap](results/figures/heatmap_f1_temp_chunk_top.png) | Retrieval depth × chunking × temperature |
| [Model clusters](results/figures/model_cluster.png) | Fitness distribution across model size groups |
| [Top-k distribution](results/figures/Top_k_distribution.png) | Concentration of retrieval depth across generations |
| [Method schematic](results/figures/Hyperparameter_Optimierung_Methode.png) | Paper-native pipeline schematic |

## Architecture

<p align="center">
  <img src="docs/assets/system-architecture.svg" alt="System architecture for resource-aware RAG HPO" width="100%">
</p>

- **Local inference** — Ollama models for generation, judging, and embeddings.
- **Retrieval** — Chroma vector stores over licensed technical documents.
- **Evaluation** — Ragas factual correctness (F1) as the fitness signal.
- **Search** — DEAP genetic optimization with elitism and average-fitness early stopping.
- **Analysis** — notebooks for convergence, latency, parameter importance, and model clusters.

## Repository Map

```text
configs/              Runtime configuration (single source of truth)
docs/                 Methodology, design, results, reproduction, ADRs
notebooks/            Analysis notebooks
results/figures/      Publication figures
results/samples/      Lightweight result samples
src/evo_rag_hpo/      Python package
tests/                Unit and contract tests
```

Raw manuals, local vector stores, and full experiment logs are excluded from public tracking.

## Quick Start

```bash
# Environment
conda env create -f environment.yml
conda activate evo-rag

# Models
ollama pull embeddinggemma:300m
ollama pull qwen3-coder:30b

# Workflow
python -m evo_rag_hpo.index    --config configs/default.yaml   # build Chroma vector stores
python -m evo_rag_hpo.optimize --config configs/default.yaml   # run evolutionary search
python -m evo_rag_hpo.evaluate 1 2 6 0 3 --config configs/default.yaml   # score one genotype
```

Add `--force` to `index` to rebuild an existing Chroma collection instead of resuming.

Runtime behavior is explicit in [configs/default.yaml](configs/default.yaml): model choices, the search space, the fixed context window (`num_ctx`), keep-alive values, NaN handling, the early-stopping criterion, and output paths.

## Reproduce

| Level | Goal | Command or artifact |
| --- | --- | --- |
| Smoke test | Validate imports and lightweight contracts | `python -m compileall src && python -m unittest discover -s tests` |
| Engineering gate | Run lint, format, compile, and tests | `make check` |
| Sample analysis | Inspect public samples and figures | `results/samples/`, `results/figures/` |
| Full run | Re-run the complete optimization loop | Requires licensed documents, evaluation set, Ollama, and configured models |

Full experimental artifacts will be published to Zenodo or a GitHub Release after paper clearance.

## Data and Model Policy

This repository does **not** redistribute third-party technical manuals, model weights, local vector databases, or large experiment logs.

- [Data availability](docs/data-availability.md)
- [Model attribution](docs/model-attribution.md)
- [Reproduction guide](docs/reproduction.md)
- [Reproduction-fidelity audit](docs/improvement-audit.md)
- [Roadmap](ROADMAP.md)

## Citation

Citation metadata will be finalized after publication. Until then, cite this repository as the research artifact for the under-review manuscript (see [CITATION.cff](CITATION.cff)).

## License

Code and documentation are released under the Apache License 2.0. Data and model weights remain subject to their upstream licenses.
