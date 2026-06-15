import copy
import unittest

from evo_rag_hpo.config import DEFAULT_CONFIG
from evo_rag_hpo.runtime import calculate_num_ctx, ensure_equal_lengths, failure_fitness, metric_mean


class FakeSeries:
    def __init__(self, values):
        self.values = values

    def fillna(self, value):
        return FakeSeries([value if item is None else item for item in self.values])

    def dropna(self):
        return FakeSeries([item for item in self.values if item is not None])

    def isna(self):
        return FakeSeries([item is None for item in self.values])

    def any(self):
        return any(self.values)

    def mean(self):
        return sum(self.values) / len(self.values)

    def __len__(self):
        return len(self.values)


class FakeFrame:
    columns = ["factual_correctness(mode=f1)"]

    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        if key not in self.columns:
            raise KeyError(key)
        return FakeSeries(self.values)


class RuntimeTests(unittest.TestCase):
    def test_calculate_num_ctx_uses_retrieval_size_and_caps(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["inference"]["max_num_ctx"] = 6000
        params = {"chunk_size": 1024, "top_k": 10}

        self.assertEqual(calculate_num_ctx(params, config), 6000)

    def test_calculate_num_ctx_respects_minimum(self):
        params = {"chunk_size": 128, "top_k": 1}

        self.assertEqual(calculate_num_ctx(params, DEFAULT_CONFIG), DEFAULT_CONFIG["inference"]["min_num_ctx"])

    def test_ensure_equal_lengths_rejects_mismatch(self):
        with self.assertRaises(ValueError):
            ensure_equal_lengths([1], [1, 2])

    def test_metric_mean_zero_nan_policy(self):
        self.assertEqual(metric_mean(FakeFrame([1.0, None, 0.5]), "zero"), 0.5)

    def test_metric_mean_raise_nan_policy(self):
        with self.assertRaises(ValueError):
            metric_mean(FakeFrame([1.0, None]), "raise")

    def test_failure_fitness_is_configured(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["evaluation"]["failed_candidate_fitness"] = -1.0

        self.assertEqual(failure_fitness(config), (-1.0,))


if __name__ == "__main__":
    unittest.main()
