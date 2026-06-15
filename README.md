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
  <a href="#why-it-matters">Why it matters</a> |
  <a href="#results-at-a-glance">Results</a> |
  <a href="#quick-start">Quick start</a> |
  <a href="#reproduce-the-study">Reproduce</a> |
  <a href="docs/methodology.md">Methodology</a>
</p>

---

## Why It Matters

Manufacturing teams rely on technical documentation for maintenance, troubleshooting, process support, and operator assistance. Retrieval-augmented generation can make that knowledge accessible through natural language, but real-world quality depends on interacting choices across retrieval, chunking, model selection, and decoding.

This repository turns RAG configuration into an engineering optimization problem: a genetic algorithm searches the mixed hyperparameter space and evaluates each candidate end to end using factual correctness.

> This repository accompanies the under-review manuscript **"Evolutionary Hyperparameter Optimization of Resource-Aware Naive-RAG Pipelines for Technical Knowledge Access in Manufacturing"**.

## Results At A Glance

<p align="center">
  <img src="docs/assets/results-overview.svg" alt="Results overview" width="100%">
</p>

## Paper Figures

The README uses a custom results overview so the public landing page keeps consistent typography and visual scale. The original paper figures remain available as full-size artifacts:

| Figure | What it shows |
| --- | --- |
| [Optimization convergence](results/figures/Konvergenz_Max_Avg.png) | Average and maximum fitness over generations |
| [Quality-latency tradeoff](results/figures/max_fit_vs_latency.png) | Fitness versus inference-time behavior |
| [Feature importance](results/figures/Feature_Importance_Random_Forest.png) | Relative influence of core RAG hyperparameters |
| [Interaction heatmap](results/figures/heatmap_f1_temp_chunk_top.png) | Interaction between retrieval depth, chunking, and temperature |
| [Model clusters](results/figures/model_cluster.png) | Grouping of model behavior in the search results |
| [Top-k distribution](results/figures/Top_k_distribution.png) | Evolutionary concentration over retrieval depth |

## System Architecture

<p align="center">
  <img src="results/figures/Hyperparameter_Optimierung_Methode.png" alt="RAG optimization workflow" width="90%">
</p>

The workflow combines:

- **Local inference:** Ollama models for generation, judging, and embeddings.
- **Retrieval:** Chroma vector stores over licensed technical documents.
- **Evaluation:** Ragas factual correctness as the automated fitness signal.
- **Search:** DEAP-based genetic optimization with elitism and early stopping.
- **Analysis:** notebooks for convergence, latency, parameter importance, and model clusters.

## Repository Map

```text
configs/              Central runtime configuration
docs/                 Methodology, data, reproduction, and attribution notes
notebooks/            Curated analysis notebooks
results/figures/      Selected publication figures
results/samples/      Lightweight result samples
src/evo_rag_hpo/      Reusable Python package
tests/                Unit and import tests
```

Legacy working folders, submission files, raw manuals, local vector stores, and full experiment logs are intentionally excluded from public Git tracking.

## Quick Start

Create the environment:

```bash
conda env create -f environment.yml
conda activate evo-rag
```

Install Ollama and pre-pull the configured models:

```bash
ollama pull embeddinggemma:300m
ollama pull qwen3-coder:30b
```

Run the workflow:

```bash
python -m evo_rag_hpo.index --config configs/default.yaml
python -m evo_rag_hpo.optimize --config configs/default.yaml
python -m evo_rag_hpo.evaluate 1 2 6 0 3 --config configs/default.yaml
```

## Reproduce The Study

| Level | Goal | Command or artifact |
| --- | --- | --- |
| Smoke test | Validate package imports and lightweight contracts | `python -m compileall src && python -m unittest discover -s tests` |
| Sample analysis | Inspect public samples and selected figures | `results/samples/`, `results/figures/` |
| Full artifact | Re-run the complete optimization loop | Requires licensed source documents, evaluation set, Ollama, configured models, and full CSV artifacts |

Full experimental artifacts are intended for Zenodo or GitHub Release publication after paper clearance.

## Public Interfaces

```bash
python -m evo_rag_hpo.index      # build Chroma vector stores
python -m evo_rag_hpo.optimize   # run evolutionary optimization
python -m evo_rag_hpo.evaluate   # evaluate a single genotype
```

The central runtime configuration is [configs/default.yaml](configs/default.yaml).

## Data And Model Policy

This repository does **not** redistribute third-party technical manuals, model weights, local vector databases, or large experiment logs.

- Data availability: [docs/data-availability.md](docs/data-availability.md)
- Model attribution: [docs/model-attribution.md](docs/model-attribution.md)
- Reproduction guide: [docs/reproduction.md](docs/reproduction.md)

## Citation

Citation information will be added after publication. Until then, cite this repository as a research artifact for the under-review manuscript.

## License

Code and documentation are released under the Apache License 2.0. Data and model weights are subject to separate upstream licenses and availability constraints.
