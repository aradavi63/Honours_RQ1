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
from rq1_harness.membership import (
    gaussian_out_cdf_scores,
    membership_metrics,
    spatial_temporal_scores,
)
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
from rq1_harness.training import (
    class_matched_indices,
    dirichlet_partitions,
    iid_partitions,
    load_or_create_iid_partitions,
)
from scripts.run_e0_matrix import run_one
from scripts.run_e2_matrix import result_filename as e2_result_filename
from scripts.run_e4a_matrix import result_filename


class AggregationTests(unittest.TestCase):
    def test_e2_result_names_isolate_non_iid_runs(self):
        self.assertEqual(
            e2_result_filename("plaintext", "all_rounds", 0.2, 1),
            "plaintext_all_rounds_fraction-0p2_seed-1.csv",
        )
        self.assertEqual(
            e2_result_filename(
                "plaintext", "all_rounds", 0.2, 1, "dirichlet", 1.0,
            ),
            "dirichlet-alpha-1_plaintext_all_rounds_fraction-0p2_seed-1.csv",
        )

    def test_e4a_result_names_separate_client_counts_and_score_methods(self):
        self.assertEqual(
            result_filename("margin", "individual_plaintext", 1, 5),
            "individual_plaintext_seed-1.csv",
        )
        self.assertEqual(
            result_filename("gaussian_cdf", "individual_plaintext", 1, 10),
            "clients-10_gaussian_cdf_individual_plaintext_seed-1.csv",
        )
        self.assertEqual(
            result_filename("margin", "individual_plaintext", 1, 5, "dirichlet", 1.0),
            "dirichlet-alpha-1_individual_plaintext_seed-1.csv",
        )
        self.assertEqual(
            result_filename(
                "margin", "individual_plaintext", 1, 5, "dirichlet", 1.0,
                "class_matched",
            ),
            "dirichlet-alpha-1_class-matched_individual_plaintext_seed-1.csv",
        )

    def test_dirichlet_partitions_are_reproducible_disjoint_and_balanced(self):
        labels = np.repeat(np.arange(4), 50)
        first = dirichlet_partitions(labels, 5, 20, alpha=0.5, seed=3)
        second = dirichlet_partitions(labels, 5, 20, alpha=0.5, seed=3)
        self.assertEqual(first, second)
        self.assertTrue(all(len(client) == 20 for client in first))
        flattened = [index for client in first for index in client]
        self.assertEqual(len(flattened), len(set(flattened)))
        histograms = [tuple(np.bincount(labels[client], minlength=4)) for client in first]
        self.assertGreater(len(set(histograms)), 1)

    def test_class_matched_indices_reproduce_reference_histogram(self):
        reference = np.array([0, 0, 1, 3, 3, 3])
        candidates = np.repeat(np.arange(4), 10)
        selected = class_matched_indices(reference, candidates, seed=4)
        np.testing.assert_array_equal(
            np.bincount(candidates[selected], minlength=4),
            np.bincount(reference, minlength=4),
        )
        self.assertEqual(len(selected), len(set(selected)))

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

    def test_membership_metrics_identify_perfect_separation(self):
        values = membership_metrics([0.9, 0.8, 0.7], [0.3, 0.2, 0.1])
        self.assertEqual(values["roc_auc"], 1.0)
        self.assertEqual(values["tpr_at_fpr_01"], 1.0)
        self.assertEqual(values["tpr_at_fpr_001"], 1.0)
        self.assertEqual(values["membership_advantage"], 1.0)

    def test_membership_metrics_reject_empty_scores(self):
        with self.assertRaises(ValueError):
            membership_metrics([], [0.1])

    def test_spatial_temporal_score_uses_target_minus_other_clients(self):
        rounds = [
            np.array([[0.8, 0.2, 0.4], [0.3, 0.1, 0.1]]),
            np.array([[0.6, 0.2, 0.2], [0.5, 0.3, 0.3]]),
        ]
        np.testing.assert_allclose(
            spatial_temporal_scores(rounds, "individual_plaintext"), [0.45, 0.2]
        )
        np.testing.assert_allclose(
            spatial_temporal_scores(rounds, "colluding_clients"), [0.45, 0.2]
        )

    def test_route_aggregate_score_averages_rounds(self):
        rounds = [np.array([[0.2], [0.8]]), np.array([[0.6], [0.4]])]
        np.testing.assert_allclose(
            spatial_temporal_scores(rounds, "route_aggregate"), [0.4, 0.6]
        )

    def test_gaussian_out_cdf_scores_target_against_other_clients(self):
        rounds = [
            np.array([[0.0, -1.0, 1.0], [2.0, 0.0, 0.0]]),
            np.array([[0.0, -1.0, 1.0], [2.0, 0.0, 0.0]]),
        ]
        scores = gaussian_out_cdf_scores(rounds)
        self.assertAlmostEqual(scores[0], 0.5)
        self.assertGreater(scores[1], 0.999)

    def test_gaussian_out_cdf_requires_two_shadow_clients(self):
        with self.assertRaises(ValueError):
            gaussian_out_cdf_scores([np.array([[0.5, 0.2]])])


if __name__ == "__main__":
    unittest.main()
