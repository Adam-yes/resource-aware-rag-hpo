"""Genetic optimization entrypoint for the RAG hyperparameter search."""

from __future__ import annotations

import argparse
import asyncio
import random
from typing import Sequence

from .config import decode_individual, ensure_output_directories, load_config, resolve_project_path, search_space_limits
from .elitism import ea_simple_with_elitism
from .evaluate import run_async_evaluate


def mutate_hpo_space(individual: list[int], indpb: float, limits: Sequence[int]) -> tuple[list[int]]:
    """Mutate mixed ordinal/categorical HPO genes while staying in bounds."""

    last_idx = len(limits) - 1
    for idx, max_val in enumerate(limits):
        if random.random() >= indpb:
            continue

        current_val = individual[idx]
        if idx == last_idx:
            if max_val <= 0:
                continue
            individual[idx] = (current_val + random.randint(1, max_val)) % (max_val + 1)
        elif current_val == 0:
            individual[idx] = 1
        elif current_val == max_val:
            individual[idx] = max_val - 1
        else:
            individual[idx] += random.choice([-1, 1])

    return (individual,)


def run_optimization(config_path: str | None = None) -> None:
    """Run the full evolutionary optimization loop."""

    import numpy as np
    from deap import base, creator, tools

    config = load_config(config_path)
    ensure_output_directories(config)

    opt = config["optimization"]
    limits = search_space_limits(config)
    random.seed(opt["random_seed"])

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

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
    toolbox.register("select", tools.selTournament, tournsize=opt["tournament_size"])
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", mutate_hpo_space, indpb=opt["individual_mutation_probability"], limits=limits)

    fitness_archive: dict[tuple[int, ...], tuple[float]] = {}

    def fitness_wrapper(individual: list[int]) -> tuple[float]:
        key = tuple(individual)
        if key not in fitness_archive:
            fitness_archive[key] = asyncio.run(run_async_evaluate(list(individual), config))
        return fitness_archive[key]

    toolbox.register("evaluate", fitness_wrapper)

    population = toolbox.population(n=opt["population_size"])
    halloffame = tools.HallOfFame(opt["hall_of_fame_size"])
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

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
    )

    best = halloffame[0]
    print(logbook)
    print("Best solution found")
    print(f"Fitness: {best.fitness.values[0]:.4f}")
    print(f"Genes: {list(best)}")
    print(f"Parameters: {decode_individual(best, config)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run resource-aware RAG hyperparameter optimization.")
    parser.add_argument("--config", default=None, help="Path to a YAML configuration file.")
    args = parser.parse_args()
    run_optimization(args.config)


if __name__ == "__main__":
    main()

