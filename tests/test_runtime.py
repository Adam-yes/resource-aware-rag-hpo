"""Unit tests for the dependency-light runtime helpers.

These tests use lightweight fakes instead of pandas/numpy so they run fast and without the heavy
inference stack, exercising the metric-aggregation contract and the length guard.
"""

import unittest

from evo_rag_hpo.runtime import ensure_equal_lengths, metric_mean


class FakeSeries:
    """Minimal stand-in for a pandas Series supporting the operations metric_mean uses."""

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
    """Minimal stand-in for the RAGAS results DataFrame."""

    columns = ["factual_correctness(mode=f1)"]

    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        if key not in self.columns:
            raise KeyError(key)
        return FakeSeries(self.values)


class RuntimeTests(unittest.TestCase):
    def test_ensure_equal_lengths_rejects_mismatch(self):
        with self.assertRaises(ValueError):
            ensure_equal_lengths([1], [1, 2])

    def test_ensure_equal_lengths_accepts_matching(self):
        ensure_equal_lengths([1, 2], [3, 4])

    def test_metric_mean_drop_nan_policy_matches_pandas_default(self):
        # "drop" excludes NaNs, reproducing the original aggregation: mean of [1.0, 0.5].
        self.assertEqual(metric_mean(FakeFrame([1.0, None, 0.5]), "drop"), 0.75)

    def test_metric_mean_zero_nan_policy_penalizes_failures(self):
        # "zero" replaces the NaN with 0: mean of [1.0, 0.0, 0.5].
        self.assertEqual(metric_mean(FakeFrame([1.0, None, 0.5]), "zero"), 0.5)

    def test_metric_mean_raise_nan_policy(self):
        with self.assertRaises(ValueError):
            metric_mean(FakeFrame([1.0, None]), "raise")

    def test_metric_mean_missing_column_raises_keyerror(self):
        class EmptyFrame:
            columns = []

        with self.assertRaises(KeyError):
            metric_mean(EmptyFrame(), "drop")

    def test_metric_mean_all_nan_dropped_raises(self):
        with self.assertRaises(ValueError):
            metric_mean(FakeFrame([None, None]), "drop")


if __name__ == "__main__":
    unittest.main()
