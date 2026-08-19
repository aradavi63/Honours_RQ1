import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch

from rq1_harness.aggregation import weighted_fedavg
from rq1_harness.fedshe import (
    fedshe_plain_weighted_fedavg,
    load_fedshe_ckks_parameters,
)
from rq1_harness.metrics import aggregation_error, targeted_attack_success_rate
from rq1_harness.inversion import (
    average_gradients,
    ciphertext_only_result,
    infer_single_label,
    parameter_gradients,
    reconstruction_metrics,
)
from rq1_harness.poisoning import (
    attack_is_active,
    attack_metrics,
    flip_source_labels,
    select_malicious_clients,
)
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

    def test_label_flipping_does_not_mutate_clean_targets(self):
        clean = torch.tensor([1, 2, 1, 7])
        poisoned = flip_source_labels(clean, 1, 7)
        self.assertEqual(clean.tolist(), [1, 2, 1, 7])
        self.assertEqual(poisoned.tolist(), [7, 2, 7, 7])

    def test_malicious_client_selection_is_reproducible(self):
        first = select_malicious_clients(10, 0.2, 4)
        second = select_malicious_clients(10, 0.2, 4)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        with self.assertRaises(ValueError):
            select_malicious_clients(5, 0.1, 4)

    def test_attack_schedules(self):
        self.assertEqual(
            [attack_is_active(i, 5, "first_third") for i in range(5)],
            [True, True, False, False, False],
        )
        self.assertEqual(
            [attack_is_active(i, 5, "final_third") for i in range(5)],
            [False, False, False, True, True],
        )

    def test_attack_metrics_separate_source_and_unaffected_classes(self):
        values = attack_metrics([1, 1, 2, 2, 7, 7], [7, 1, 2, 3, 7, 7], 1, 7)
        self.assertEqual(values["source_recall"], 0.5)
        self.assertEqual(values["targeted_attack_success_rate"], 0.5)
        self.assertEqual(values["unaffected_macro_recall"], 0.75)

    def test_individual_gradient_recovers_label(self):
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 3))
        gradients = parameter_gradients(
            model, torch.tensor([[[[0.1, 0.2], [0.3, 0.4]]]]), torch.tensor([2])
        )
        self.assertEqual(infer_single_label(gradients), 2)

    def test_average_gradients_matches_mean_batch_gradient(self):
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 3))
        inputs = torch.tensor(
            [[[[0.1, 0.2], [0.3, 0.4]]], [[[0.9, 0.8], [0.7, 0.6]]]]
        )
        labels = torch.tensor([0, 2])
        individuals = [
            parameter_gradients(model, inputs[index : index + 1], labels[index : index + 1])
            for index in range(2)
        ]
        averaged = average_gradients(individuals)
        batch = parameter_gradients(model, inputs, labels)
        for actual, expected in zip(averaged, batch):
            torch.testing.assert_close(actual, expected)

    def test_reconstruction_metrics_are_exact_for_identical_images(self):
        image = torch.tensor([[[[0.0, 0.5], [0.75, 1.0]]]])
        values = reconstruction_metrics(image, image)
        self.assertEqual(values["mse"], 0.0)
        self.assertEqual(values["psnr"], float("inf"))
        self.assertAlmostEqual(values["ssim"], 1.0)

    def test_ciphertext_observation_is_not_attackable_as_plaintext(self):
        result = ciphertext_only_result()
        self.assertEqual(result["applicability"], "not_applicable")
        self.assertTrue(np.isnan(result["mse"]))


if __name__ == "__main__":
    unittest.main()
