import unittest
import tempfile
from pathlib import Path

import numpy as np

from rq1_harness.aggregation import weighted_fedavg
from rq1_harness.fedshe import (
    fedshe_plain_weighted_fedavg,
    load_fedshe_ckks_parameters,
)
from rq1_harness.metrics import aggregation_error, targeted_attack_success_rate
from rq1_harness.training import iid_partitions, load_or_create_iid_partitions
from scripts.run_e0_matrix import run_one


class AggregationTests(unittest.TestCase):
    def test_weighted_average(self):
        updates = [{"x": np.array([1.0, 3.0])}, {"x": np.array([3.0, 7.0])}]
        result = weighted_fedavg(updates, [1, 3])
        np.testing.assert_allclose(result["x"], [2.5, 6.0])

    def test_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            weighted_fedavg(
                [{"x": np.zeros(2)}, {"x": np.zeros(3)}], [1, 1]
            )

    def test_fedshe_plain_matches_weighted_fedavg(self):
        updates = [{"x": np.array([1.0, 3.0])}, {"x": np.array([3.0, 7.0])}]
        expected = weighted_fedavg(updates, [1, 3])
        actual = fedshe_plain_weighted_fedavg(updates, [1, 3])
        np.testing.assert_allclose(actual["x"], expected["x"])

    def test_fedshe_ckks_parameters_are_loaded_from_pinned_submodule(self):
        parameters = load_fedshe_ckks_parameters("128", "0", "8192")
        self.assertEqual(parameters["scheme"], "CKKS")
        self.assertEqual(parameters["n"], 8192)

    def test_fedshe_plain_matrix_row_passes(self):
        row = run_one("fedshe_plain", clients=2, seed=1)
        self.assertTrue(row["passes_acceptance"])
        self.assertEqual(row["relative_l2_error"], 0.0)

    def test_zero_error_metrics(self):
        value = {"x": np.array([1.0, 2.0])}
        metrics = aggregation_error(value, value)
        self.assertEqual(metrics["max_absolute_error"], 0.0)
        self.assertAlmostEqual(metrics["cosine_similarity"], 1.0)

    def test_targeted_asr(self):
        labels = np.array([1, 1, 2, 1])
        predictions = np.array([7, 3, 7, 7])
        self.assertAlmostEqual(
            targeted_attack_success_rate(labels, predictions, 1, 7), 2 / 3
        )

    def test_iid_partitions_are_reproducible_and_disjoint(self):
        first = iid_partitions(100, client_count=4, samples_per_client=10, seed=7)
        second = iid_partitions(100, client_count=4, samples_per_client=10, seed=7)
        self.assertEqual(first, second)
        flattened = [index for client in first for index in client]
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_partition_manifest_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partition.json"
            created = load_or_create_iid_partitions(
                path, "test", 100, 4, 10, 7
            )
            loaded = load_or_create_iid_partitions(
                path, "test", 100, 4, 10, 7
            )
            self.assertEqual(created, loaded)
            with self.assertRaises(ValueError):
                load_or_create_iid_partitions(path, "test", 100, 5, 10, 7)


if __name__ == "__main__":
    unittest.main()
