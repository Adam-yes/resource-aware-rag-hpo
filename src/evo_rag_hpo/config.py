"""Configuration helpers for the resource-aware RAG HPO workflow."""

from __future__ import annotations

from copy import deepcopy
from hashlib import md5
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    "paths": {
        "raw_documents": "data/raw",
        "persist_directory": "data/chroma_db",
        "evaluation_dataset": "data/evaluation/evaluation_testset_50_short_query_length.csv",
        "hpo_history": "results/experiments/hpo_history.csv",
        "computation_log": "results/experiments/evolution_computation_log.csv",
        "fitness_archive": "results/experiments/fitness_archive.json",
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
        "early_stopping_patience": 2,
        "early_stopping_metric": "max",
    },
    "inference": {
        "prompt_token_headroom": 1024,
        "answer_token_budget": 1024,
        "min_num_ctx": 5120,
        "max_num_ctx": 16384,
        "num_predict": 1024,
        "llm_keep_alive": "15m",
        "embedding_keep_alive": "15m",
    },
    "evaluation": {
        "timeout": 720,
        "max_retries": 2,
        "max_wait": 30,
        "max_workers": 8,
        "nan_policy": "zero",
        "failed_candidate_fitness": 0.0,
        "failure_retries": 1,
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
    """Load the YAML config, falling back to the built-in defaults.

    Heavy execution paths depend on PyYAML through the environment file, but tests and
    simple imports should still work without it.
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
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def validate_config(config: dict[str, Any]) -> None:
    """Validate the public config shape with clear, early errors."""

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
    for key in ["prompt_token_headroom", "answer_token_budget", "min_num_ctx", "max_num_ctx", "num_predict"]:
        if int(inference[key]) <= 0:
            raise ValueError(f"inference.{key} must be positive.")
    if int(inference["max_num_ctx"]) < int(inference["min_num_ctx"]):
        raise ValueError("inference.max_num_ctx must be >= inference.min_num_ctx.")

    evaluation = config["evaluation"]
    if evaluation["nan_policy"] not in {"zero", "drop", "raise"}:
        raise ValueError("evaluation.nan_policy must be one of: zero, drop, raise.")
    if int(evaluation["failure_retries"]) < 0:
        raise ValueError("evaluation.failure_retries must be non-negative.")


def decode_individual(individual: list[int] | tuple[int, ...], config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Translate a genotype into concrete RAG hyperparameters."""

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
    """Return the maximum valid gene index for each search dimension."""

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
    """Create the stable short hash used to join HPO and computation logs."""

    return md5(str(list(individual)).encode("utf-8")).hexdigest()[:8]


def resolve_project_path(path: str | Path) -> Path:
    """Resolve repository-relative paths without requiring a specific working directory."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).resolve().parents[2] / candidate


def ensure_output_directories(config: dict[str, Any] | None = None) -> None:
    """Create parent directories for configured outputs."""

    cfg = config or load_config()
    for key in ("hpo_history", "computation_log", "fitness_archive"):
        resolve_project_path(cfg["paths"][key]).parent.mkdir(parents=True, exist_ok=True)
    resolve_project_path(cfg["paths"]["persist_directory"]).mkdir(parents=True, exist_ok=True)
