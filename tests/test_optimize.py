import random
import unittest

from evo_rag_hpo.config import DEFAULT_CONFIG, search_space_limits
from evo_rag_hpo.optimize import mutate_hpo_space


class OptimizeTests(unittest.TestCase):
    def test_mutation_stays_within_search_space_bounds(self):
        random.seed(42)
        limits = search_space_limits(DEFAULT_CONFIG)
        individual = [0, limits[1], 5, 5, 0]

        mutated = mutate_hpo_space(individual, indpb=1.0, limits=limits)[0]

        self.assertEqual(len(mutated), len(limits))
        self.assertTrue(all(0 <= gene <= limit for gene, limit in zip(mutated, limits)))

    def test_categorical_model_mutation_changes_model_when_possible(self):
        random.seed(1)
        limits = search_space_limits(DEFAULT_CONFIG)
        individual = [0, 0, 0, 0, 0]

        mutated = mutate_hpo_space(individual, indpb=1.0, limits=limits)[0]

        self.assertNotEqual(mutated[-1], 0)


if __name__ == "__main__":
    unittest.main()

