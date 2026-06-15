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
        return deepcopy(DEFAULT_CONFIG)

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load YAML configuration files.") from exc

    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}

    config = deepcopy(DEFAULT_CONFIG)
    _deep_update(config, loaded)
    return config


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


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
    for key in ("hpo_history", "computation_log"):
        resolve_project_path(cfg["paths"][key]).parent.mkdir(parents=True, exist_ok=True)
    resolve_project_path(cfg["paths"]["persist_directory"]).mkdir(parents=True, exist_ok=True)

