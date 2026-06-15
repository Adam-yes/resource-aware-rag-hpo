"""Driver for the evolutionary hyperparameter search over the Naive-RAG pipeline.

This module wires together the DEAP genetic algorithm: it builds the toolbox (gene
initializers, selection, crossover, mutation), connects the RAGAS-based fitness function, and
runs the elitist evolutionary loop defined in :mod:`evo_rag_hpo.elitism`.

Genome encoding
---------------
Each individual is a list of five integer indices into the search space, in the fixed order
``[chunk_size, chunk_overlap, top_k, temperature, model_name]``. Working in index space lets a
single, uniform set of integer operators handle a search space that mixes ordinal dimensions
(chunk size, Top-k, temperature) with a purely categorical one (model name);
:func:`evo_rag_hpo.config.decode_individual` maps indices back to concrete values.

Reproduction and provenance
---------------------------
Values taken directly from the paper / the experiment data: the population size (42, verified
against ``hpo_history.csv``), the early-stopping criterion, the search space, and the fitness
definition. The per-operator probabilities (crossover, mutation, per-gene mutation, tournament
size) are not separately tabulated in the manuscript; they are exposed in
:mod:`configs.default` so they are explicit and auditable, follow the conventions of the
reference GA implementation (Wirsansky, *Hands-On Genetic Algorithms with Python*) cited in the
paper, and are kept in configuration so a reproduction can pin or vary them deliberately.
"""

from __future__ import annotations

import argparse
import asyncio
import random
from typing import Sequence

from .config import decode_individual, ensure_output_directories, load_config, resolve_project_path, search_space_limits
from .elitism import ea_simple_with_elitism
from .evaluate import run_async_evaluate
from .runtime import configure_logging, set_deterministic_seed


def mutate_hpo_space(individual: list[int], indpb: float, limits: Sequence[int]) -> tuple[list[int]]:
    """Mutate a genome in place, respecting the structure of each search dimension.

    Two kinds of genes are handled differently:

    * **Ordinal genes** (chunk size, chunk overlap, Top-k, temperature) are nudged by a single
      step (-1 / +1) so that mutation explores *neighboring* values, preserving the ordering
      semantics of the dimension. At the boundaries the step is reflected inward to stay in range.
    * **The categorical gene** (model name, the last gene) has no meaningful ordering, so it is
      mutated by jumping to a different random index via modular arithmetic - guaranteeing the
      model actually changes rather than drifting to an adjacent, unrelated entry.

    Args:
        individual: The genome to mutate in place.
        indpb: Independent per-gene mutation probability.
        limits: The maximum valid index for each gene (inclusive).

    Returns:
        A one-tuple ``(individual,)`` as required by DEAP's mutation operator contract.
    """

    last_idx = len(limits) - 1
    for idx, max_val in enumerate(limits):
        if random.random() >= indpb:
            continue

        current_val = individual[idx]
        if idx == last_idx:
            # Categorical (model) gene: jump to a different index uniformly at random.
            if max_val <= 0:
                continue
            individual[idx] = (current_val + random.randint(1, max_val)) % (max_val + 1)
        elif current_val == 0:
            # At the lower bound: the only in-range step is upward.
            individual[idx] = 1
        elif current_val == max_val:
            # At the upper bound: the only in-range step is downward.
            individual[idx] = max_val - 1
        else:
            # Interior ordinal value: take one step in either direction.
            individual[idx] += random.choice([-1, 1])

    return (individual,)


def run_optimization(config_path: str | None = None) -> None:
    """Build the GA toolbox, run the evolutionary search, and report the best configuration.

    Args:
        config_path: Optional path to a YAML configuration file. When ``None`` the default
            configuration is used.
    """

    import numpy as np
    from deap import base, creator, tools

    config = load_config(config_path)
    ensure_output_directories(config)
    configure_logging(config["logging"]["level"])

    opt = config["optimization"]
    limits = search_space_limits(config)

    # Seed Python's (and NumPy's) RNG so population init, selection, crossover, and mutation are
    # reproducible. Local LLM inference is only as deterministic as the Ollama backend allows, so
    # exact end-to-end reproduction of fitness values is best-effort.
    set_deterministic_seed(opt["random_seed"])

    # Single-objective maximization: fitness is the mean factual-correctness F1 score. ``creator``
    # registers types on a module-global registry, so guard against re-creation on re-import.
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    # Per-gene initializers draw a uniform random index within each dimension's bounds.
    toolbox = base.Toolbox()
    toolbox.register("attr_size", random.randint, 0, limits[0])
    toolbox.register("attr_overlap", random.randint, 0, limits[1])
    toolbox.register("attr_k", random.randint, 0, limits[2])
    toolbox.register("attr_temp", random.randint, 0, limits[3])
    toolbox.register("attr_model", random.randint, 0, limits[4])
    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        (toolbox.attr_size, toolbox.attr_overlap, toolbox.attr_k, toolbox.attr_temp, toolbox.attr_model),
        n=1,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Evolutionary operators. Tournament selection balances selection pressure and diversity;
    # uniform crossover swaps genes independently; the custom structure-aware mutation above
    # handles the ordinal/categorical mix.
    toolbox.register("select", tools.selTournament, tournsize=opt["tournament_size"])
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", mutate_hpo_space, indpb=opt["individual_mutation_probability"], limits=limits)

    # The fitness function is asynchronous (it awaits batched LLM calls), but DEAP's ``map`` is
    # synchronous, so each evaluation is driven to completion on a dedicated event loop. There is
    # deliberately no memoization cache: the original experiment re-evaluated repeated genomes
    # across generations (203 evaluations for 152 unique configurations), and a cache would change
    # both the evaluation counts and the computation log.
    loop = asyncio.new_event_loop()

    def evaluate_individual(individual: list[int]) -> tuple[float]:
        return loop.run_until_complete(run_async_evaluate(list(individual), config))

    toolbox.register("evaluate", evaluate_individual)

    population = toolbox.population(n=opt["population_size"])
    halloffame = tools.HallOfFame(opt["hall_of_fame_size"])

    # Compile both average and maximum population fitness per generation; ``avg`` drives the
    # Delta-mu early-stopping criterion, ``max`` tracks the best-so-far for reporting.
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    try:
        _, logbook = ea_simple_with_elitism(
            population=population,
            toolbox=toolbox,
            cxpb=opt["crossover_probability"],
            mutpb=opt["mutation_probability"],
            ngen=opt["max_generations"],
            stats=stats,
            halloffame=halloffame,
            verbose=True,
            filename=str(resolve_project_path(config["paths"]["hpo_history"])),
            min_improvement=opt["early_stopping_min_improvement"],
            patience=opt["early_stopping_patience"],
            stopping_metric=opt["early_stopping_metric"],
        )
    finally:
        loop.close()

    best = halloffame[0]
    print(logbook)
    print("Best solution found")
    print(f"Fitness: {best.fitness.values[0]:.4f}")
    print(f"Genes: {list(best)}")
    print(f"Parameters: {decode_individual(best, config)}")


def main() -> None:
    """Command-line entry point: ``python -m evo_rag_hpo.optimize [--config ...]``."""

    parser = argparse.ArgumentParser(description="Run resource-aware RAG hyperparameter optimization.")
    parser.add_argument("--config", default=None, help="Path to a YAML configuration file.")
    args = parser.parse_args()
    run_optimization(args.config)


if __name__ == "__main__":
    main()
