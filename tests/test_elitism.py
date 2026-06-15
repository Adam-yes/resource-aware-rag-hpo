"""Integration test: the elitist loop reproduces the paper's six-generation early stop.

Skipped automatically when DEAP/NumPy are unavailable (both are declared project dependencies
and present in CI). The test drives the real ``ea_simple_with_elitism`` while scripting the
per-generation average fitness to the published convergence sequence, and asserts the search
terminates exactly when Delta-mu first drops below 5% (i.e. after generation 5).
"""

import tempfile
from pathlib import Path

import pytest

pytest.importorskip("deap")
from deap import base, creator, tools  # noqa: E402

from evo_rag_hpo.elitism import ea_simple_with_elitism  # noqa: E402

# Average fitness per generation reported in the paper (generations 0-5).
PAPER_AVG = [0.2239, 0.2987, 0.3185, 0.3460, 0.3849, 0.4025]

if not hasattr(creator, "FitnessMaxElitismTest"):
    creator.create("FitnessMaxElitismTest", base.Fitness, weights=(1.0,))
if not hasattr(creator, "IndividualElitismTest"):
    creator.create("IndividualElitismTest", list, fitness=creator.FitnessMaxElitismTest)


class ScriptedStats:
    """Returns a scripted average per compile() call so the trajectory matches the paper."""

    fields = ["avg", "max"]

    def __init__(self, sequence):
        self._sequence = list(sequence)
        self._call = 0

    def compile(self, _population):
        idx = min(self._call, len(self._sequence) - 1)
        self._call += 1
        avg = self._sequence[idx]
        return {"avg": avg, "max": avg}


def _toolbox_and_population():
    toolbox = base.Toolbox()
    toolbox.register("evaluate", lambda _ind: (0.5,))
    toolbox.register(
        "select",
        lambda pop, k: [creator.IndividualElitismTest(list(ind)) for ind in pop[:k]],
    )
    toolbox.register("mate", lambda a, b: (a, b))

    def mutate(ind):
        del ind.fitness.values  # ensure each generation has individuals to (re)evaluate
        return (ind,)

    toolbox.register("mutate", mutate)
    population = [creator.IndividualElitismTest([i]) for i in range(6)]
    return toolbox, population


def test_elitism_stops_at_generation_five():
    with tempfile.TemporaryDirectory() as tmpdir:
        history = Path(tmpdir) / "hpo_history.csv"
        toolbox, population = _toolbox_and_population()
        _, logbook = ea_simple_with_elitism(
            population=population,
            toolbox=toolbox,
            cxpb=0.0,
            mutpb=1.0,
            ngen=7,
            stats=ScriptedStats(PAPER_AVG),
            halloffame=tools.HallOfFame(1),
            verbose=False,
            filename=str(history),
            min_improvement=0.05,
            patience=1,
            stopping_metric="avg",
        )

    generations = [record["gen"] for record in logbook]
    assert generations == [0, 1, 2, 3, 4, 5]


def test_elitism_history_logs_every_generation():
    import csv

    with tempfile.TemporaryDirectory() as tmpdir:
        history = Path(tmpdir) / "hpo_history.csv"
        toolbox, population = _toolbox_and_population()
        ea_simple_with_elitism(
            population=population,
            toolbox=toolbox,
            cxpb=0.0,
            mutpb=1.0,
            ngen=7,
            stats=ScriptedStats(PAPER_AVG),
            halloffame=tools.HallOfFame(1),
            verbose=False,
            filename=str(history),
            min_improvement=0.05,
            patience=1,
            stopping_metric="avg",
        )
        with history.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

    assert sorted({int(r["gen"]) for r in rows}) == [0, 1, 2, 3, 4, 5]
    # Each logged row carries the join key used to link with the per-question computation log.
    assert all(r["Hash Id"] for r in rows)
