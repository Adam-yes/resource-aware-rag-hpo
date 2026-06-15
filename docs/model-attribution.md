# Model Attribution

The study uses open-weight local models served through Ollama. This repository does not distribute model weights.

Users are responsible for:

- installing Ollama;
- downloading the configured models locally;
- checking upstream model licenses;
- validating whether the models are appropriate for their deployment context.

The default configuration lists the model identifiers used by the search space and the fixed evaluation components:

- embedding model: `embeddinggemma:300m`;
- judge model: `qwen3-coder:30b`;
- candidate generation models: see `configs/default.yaml`.

Model outputs generated during experiments are experiment artifacts. Model weights and their licenses remain governed by the respective upstream model providers.

