"""Configuration model, validation, and genotype decoding for the RAG HPO workflow.

This module is the single source of truth for the experiment's configuration. It exposes:

* :data:`DEFAULT_CONFIG` - the built-in defaults reproducing the published experiment, used
  whenever no YAML file is supplied (and as the base that a YAML file overrides).
* :func:`load_config` - load and validate a YAML config, deep-merged over the defaults.
* :func:`decode_individual` - map a genetic genotype (indices) to concrete hyperparameters.
* path/limit/hash helpers shared across the pipeline.

Keeping the search space and all tunables here - rather than scattered as literals across
modules - is what makes the study auditable: every number that influences the result is in one
place and is covered by :func:`validate_config`.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import md5
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"

# Built-in defaults. These reproduce the original published experiment exactly; a YAML file (see
# configs/default.yaml) may override any leaf value via load_config's deep merge.
DEFAULT_CONFIG: dict[str, Any] = {
    "paths": {
        "raw_documents": "data/raw",
        "persist_directory": "data/chroma_db",
        "evaluation_dataset": "data/evaluation/evaluation_testset_50_short_query_length.csv",
        "hpo_history": "results/experiments/hpo_history.csv",
        "computation_log": "results/experiments/evolution_computation_log.csv",
    },
    "models": {
        "embedding": "embeddinggemma:300m",
        "judge": "qwen3-coder:30b",
    },
    "optimization": {
        "population_size": 42,
        "max_generations": 7,
        "hall_of_fame_size": 1,
        "crossover_probability": 0.7,
        "mutation_probability": 0.4,
        "individual_mutation_probability": 0.2,
        "random_seed": 42,
        "tournament_size": 3,
        "early_stopping_min_improvement": 0.05,
        "early_stopping_patience": 1,
        "early_stopping_metric": "avg",
    },
    "inference": {
        # Fixed context window for both generation and judging, matching the original study.
        "num_ctx": 5120,
        "num_predict": 1024,
        "llm_keep_alive": "15m",
        "embedding_keep_alive": "15m",
    },
    "evaluation": {
        "timeout": 720,
        "max_retries": 2,
        "max_wait": 30,
        "max_workers": 8,
        # "drop" reproduces pandas' NaN-skipping mean used in the original fitness aggregation.
        "nan_policy": "drop",
    },
    "indexing": {
        "batch_size": 5000,
        "skip_existing": True,
    },
    "logging": {
        "level": "INFO",
    },
    "search_space": {
        "chunk_size": [128, 256, 384, 512, 640, 768, 896, 1024],
        "chunk_overlap": [0, 12.5, 25],
        "top_k": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "temperature": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "model_name": [
            "qwen2.5-coder:0.5b-instruct",
            "granite4:350m",
            "gemma3:1b",
            "qwen2.5-coder:1.5b-instruct",
            "deepseek-r1:1.5b",
            "qwen3:1.7b",
            "granite3.1-moe:1b",
            "granite3.3:2b",
            "qwen3-vl:2b-thinking",
            "qwen3-vl:2b-instruct",
            "qwen2.5-coder:3b-instruct",
            "granite3.1-moe:3b",
            "granite4:micro",
            "phi4-mini:3.8b-q4_K_M",
            "qwen2.5vl:3b",
            "gemma3:4b",
            "granite4:tiny-h",
            "gemma3n:e2b",
            "qwen3-coder:30b",
            "gpt-oss:20b",
            "qwen3-vl:30b-a3b-instruct",
        ],
    },
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config deep-merged over the defaults, validating the result.

    Heavy execution paths depend on PyYAML through the environment file, but tests and simple
    imports work without it: when no YAML file exists, the validated built-in defaults are
    returned and PyYAML is never imported.

    Args:
        path: Optional path to a YAML config. Defaults to ``configs/default.yaml``.

    Returns:
        A fully validated configuration dictionary.
    """

    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        config = deepcopy(DEFAULT_CONFIG)
        validate_config(config)
        return config

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load YAML configuration files.") from exc

    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}

    config = deepcopy(DEFAULT_CONFIG)
    _deep_update(config, loaded)
    validate_config(config)
    return config


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``updates`` into ``target`` (nested dicts merge; leaves overwrite)."""

    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def validate_config(config: dict[str, Any]) -> None:
    """Validate the configuration shape and ranges, raising early on misconfiguration.

    Fails fast with a precise message rather than letting an invalid value surface as an obscure
    error hours into a search.
    """

    required_sections = ["paths", "models", "optimization", "search_space", "inference", "evaluation", "indexing"]
    for section in required_sections:
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Missing or invalid config section: {section}")

    search_space = config["search_space"]
    for key in ["chunk_size", "chunk_overlap", "top_k", "temperature", "model_name"]:
        values = search_space.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"search_space.{key} must be a non-empty list.")

    opt = config["optimization"]
    for key in ["crossover_probability", "mutation_probability", "individual_mutation_probability"]:
        value = opt[key]
        if not 0 <= value <= 1:
            raise ValueError(f"optimization.{key} must be in [0, 1].")
    for key in ["population_size", "max_generations", "hall_of_fame_size", "tournament_size"]:
        if int(opt[key]) <= 0:
            raise ValueError(f"optimization.{key} must be positive.")
    if opt["early_stopping_metric"] not in {"avg", "max"}:
        raise ValueError("optimization.early_stopping_metric must be 'avg' or 'max'.")
    if int(opt["early_stopping_patience"]) < 1:
        raise ValueError("optimization.early_stopping_patience must be at least 1.")

    inference = config["inference"]
    for key in ["num_ctx", "num_predict"]:
        if int(inference[key]) <= 0:
            raise ValueError(f"inference.{key} must be positive.")

    evaluation = config["evaluation"]
    if evaluation["nan_policy"] not in {"zero", "drop", "raise"}:
        raise ValueError("evaluation.nan_policy must be one of: zero, drop, raise.")


def decode_individual(individual: list[int] | tuple[int, ...], config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Translate a genotype (list of indices) into concrete RAG hyperparameters (phenotype).

    The genome order is fixed: ``[chunk_size, chunk_overlap, top_k, temperature, model_name]``.
    Each gene is an index into the corresponding ``search_space`` list.

    Raises:
        ValueError: If the genome length does not match the number of search dimensions.
        IndexError: If any gene is out of bounds for its dimension.
    """

    cfg = config or load_config()
    search_space = cfg["search_space"]
    keys = ["chunk_size", "chunk_overlap", "top_k", "temperature", "model_name"]

    if len(individual) != len(keys):
        raise ValueError(f"Expected {len(keys)} genes, received {len(individual)}.")

    decoded: dict[str, Any] = {}
    for index, key in enumerate(keys):
        values = search_space[key]
        gene = individual[index]
        if gene < 0 or gene >= len(values):
            raise IndexError(f"Gene {index} for '{key}' is out of bounds: {gene}.")
        decoded[key] = values[gene]
    return decoded


def search_space_limits(config: dict[str, Any] | None = None) -> list[int]:
    """Return the maximum valid gene index for each search dimension (inclusive upper bounds)."""

    cfg = config or load_config()
    search_space = cfg["search_space"]
    return [
        len(search_space["chunk_size"]) - 1,
        len(search_space["chunk_overlap"]) - 1,
        len(search_space["top_k"]) - 1,
        len(search_space["temperature"]) - 1,
        len(search_space["model_name"]) - 1,
    ]


def genotype_hash(individual: list[int] | tuple[int, ...]) -> str:
    """Return the stable 8-char hash that joins per-generation and per-question logs."""

    return md5(str(list(individual)).encode("utf-8")).hexdigest()[:8]


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a repository-relative path to an absolute one, independent of the working dir."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).resolve().parents[2] / candidate


def ensure_output_directories(config: dict[str, Any] | None = None) -> None:
    """Create the parent directories for all configured output artifacts."""

    cfg = config or load_config()
    for key in ("hpo_history", "computation_log"):
        resolve_project_path(cfg["paths"][key]).parent.mkdir(parents=True, exist_ok=True)
    resolve_project_path(cfg["paths"]["persist_directory"]).mkdir(parents=True, exist_ok=True)
