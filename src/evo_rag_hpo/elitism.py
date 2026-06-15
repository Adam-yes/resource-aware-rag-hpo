"""Evolutionary loop with elitism, CSV logging, and early stopping."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from .config import genotype_hash
from .schema import HPO_HISTORY_FIELDNAMES

logger = logging.getLogger(__name__)


def ea_simple_with_elitism(
    population: list[Any],
    toolbox: Any,
    cxpb: float,
    mutpb: float,
    ngen: int,
    stats: Any | None = None,
    halloffame: Any | None = None,
    verbose: bool = True,
    filename: str = "results/experiments/hpo_history.csv",
    min_improvement: float = 0.05,
    patience: int = 2,
    stopping_metric: str = "max",
) -> tuple[list[Any], Any]:
    """Execute a DEAP-style simple evolutionary algorithm with elitism."""

    from deap import algorithms, tools

    if halloffame is None:
        raise ValueError("halloffame must be provided for elitism.")
    if stopping_metric not in {"avg", "max"}:
        raise ValueError("stopping_metric must be 'avg' or 'max'.")

    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    with open(filename, "w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(HPO_HISTORY_FIELDNAMES)

    def save_generation(gen: int, pop: list[Any]) -> None:
        with open(filename, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            for idx, ind in enumerate(pop):
                fitness = ind.fitness.values[0] if ind.fitness.valid else None
                writer.writerow([genotype_hash(ind), gen, idx, fitness, str(list(ind))])

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    halloffame.update(population)
    hof_size = len(halloffame.items) if halloffame.items else 0

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        logger.info(logbook.stream)
    save_generation(0, population)
    best_metric = record.get(stopping_metric, 0)
    stale_generations = 0

    for gen in range(1, ngen + 1):
        offspring = toolbox.select(population, len(population) - hof_size)
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        offspring.extend(halloffame.items)
        halloffame.update(offspring)
        population[:] = offspring

        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            logger.info(logbook.stream)
        save_generation(gen, population)

        current_metric = record.get(stopping_metric, 0)
        if best_metric != 0:
            improvement = (current_metric - best_metric) / abs(best_metric)
            if improvement < min_improvement:
                stale_generations += 1
                if stale_generations >= patience:
                    message = (
                        f"Early stop at generation {gen}: {stopping_metric} improvement "
                        f"{improvement:.2%} < {min_improvement:.0%} for {patience} generation(s)"
                    )
                    logger.info(message)
                    break
            else:
                stale_generations = 0
                best_metric = current_metric
        else:
            best_metric = current_metric

    return population, logbook
