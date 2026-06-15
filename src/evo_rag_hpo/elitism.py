"""Simple evolutionary algorithm with elitism, per-generation logging, and early stopping.

Attribution
-----------
The elitist evolutionary loop is based on the ``eaSimpleWithElitism`` implementation from
*Hands-On Genetic Algorithms with Python (Second Edition)* by Eyal Wirsansky
(https://github.com/PacktPublishing/Hands-On-Genetic-Algorithms-with-Python-Second-Edition,
MIT License). The original algorithm is extended here with two project-specific additions:

1. **Per-generation CSV logging** of every individual (genotype hash, generation, fitness, and
   parameter list), later joined with the detailed per-question computation log.
2. **An early-stopping criterion** that terminates the search once the relative improvement in
   average population fitness (Delta-mu) falls below a threshold.

Reproduction note
-----------------
The published run stops on the first generation whose average fitness improves by less than 5%
over the *immediately preceding* generation (Delta-mu < 5%). With the default
``stopping_metric="avg"`` and ``patience=1`` this loop reproduces that behavior exactly,
yielding the six generations (0-5) reported in the paper. ``patience`` and ``stopping_metric``
are exposed for experimentation but must keep their defaults for a faithful reproduction.
"""

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
    patience: int = 1,
    stopping_metric: str = "avg",
) -> tuple[list[Any], Any]:
    """Run an elitist (mu + elite) evolutionary algorithm and return ``(population, logbook)``.

    Args:
        population: The initial population of individuals.
        toolbox: A DEAP toolbox providing ``evaluate``, ``select``, ``mate``, ``mutate``, ``map``.
        cxpb: Crossover (mating) probability passed to :func:`deap.algorithms.varAnd`.
        mutpb: Mutation probability passed to :func:`deap.algorithms.varAnd`.
        ngen: Maximum number of generations. The search may stop earlier via early stopping.
        stats: A :class:`deap.tools.Statistics` object compiling ``avg``/``max`` records.
        halloffame: A :class:`deap.tools.HallOfFame` holding the elite individuals carried over
            unchanged into each new generation. Required - elitism is undefined without it.
        verbose: When ``True``, log the generation statistics stream.
        filename: Destination CSV for the per-generation history.
        min_improvement: Relative-improvement threshold for early stopping (0.05 = 5%).
        patience: Number of consecutive sub-threshold generations tolerated before stopping. ``1``
            reproduces the paper's "stop on first Delta-mu < 5%" behavior.
        stopping_metric: Which compiled statistic to monitor, ``"avg"`` (the paper's Delta-mu) or
            ``"max"``.

    Returns:
        The final population and the DEAP logbook of per-generation statistics.

    Raises:
        ValueError: If ``halloffame`` is ``None`` or ``stopping_metric`` is unsupported.
    """

    from deap import algorithms, tools

    if halloffame is None:
        raise ValueError("halloffame must be provided for elitism.")
    if stopping_metric not in {"avg", "max"}:
        raise ValueError("stopping_metric must be 'avg' or 'max'.")

    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    # (Re)create the history file with a fresh header so each run starts clean.
    with open(filename, "w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(HPO_HISTORY_FIELDNAMES)

    def save_generation(gen: int, pop: list[Any]) -> None:
        """Append one row per individual for the given generation.

        The genotype hash is the join key that links these coarse per-generation records to the
        fine-grained per-question rows written by the evaluation module.
        """

        with open(filename, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            for idx, ind in enumerate(pop):
                fitness = ind.fitness.values[0] if ind.fitness.valid else None
                writer.writerow([genotype_hash(ind), gen, idx, fitness, str(list(ind))])

    # --- Generation 0: evaluate the initial population ---
    # Only individuals with invalid fitness are evaluated. DEAP invalidates fitness whenever an
    # individual is created or modified, so this naturally evaluates exactly the new genomes.
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

    # Track the previous generation's metric to compute the relative improvement Delta-mu.
    last_metric = record.get(stopping_metric, 0)
    stale_generations = 0

    # --- Generational loop ---
    for gen in range(1, ngen + 1):
        # Select parents, leaving room for the elite individuals re-inserted below.
        offspring = toolbox.select(population, len(population) - hof_size)

        # Apply crossover and mutation. ``varAnd`` invalidates the fitness of any individual it
        # actually modifies, so unchanged individuals keep their score and are not re-evaluated.
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Elitism: carry the hall-of-fame individuals over unchanged, then refresh the HoF.
        offspring.extend(halloffame.items)
        halloffame.update(offspring)
        population[:] = offspring

        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            logger.info(logbook.stream)
        save_generation(gen, population)

        # --- Early stopping on relative improvement over the previous generation ---
        current_metric = record.get(stopping_metric, 0)
        if last_metric != 0:
            improvement = (current_metric - last_metric) / abs(last_metric)
            if improvement < min_improvement:
                stale_generations += 1
                if stale_generations >= patience:
                    logger.info(
                        "Early stop at generation %s: %s improvement %.2f%% < %.0f%%",
                        gen,
                        stopping_metric,
                        improvement * 100,
                        min_improvement * 100,
                    )
                    break
            else:
                stale_generations = 0
        # Always advance the reference point to the current generation (Delta-mu is measured
        # between consecutive generations, not against the best-so-far).
        last_metric = current_metric

    return population, logbook
