import unittest

import numpy as np

from rq1_harness.aggregation import weighted_fedavg
from rq1_harness.metrics import aggregation_error, targeted_attack_success_rate


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


if __name__ == "__main__":
    unittest.main()

