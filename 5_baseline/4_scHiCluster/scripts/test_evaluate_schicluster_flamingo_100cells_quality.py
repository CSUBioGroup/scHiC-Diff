import unittest
import tempfile
from pathlib import Path

import numpy as np

from evaluate_schicluster_flamingo_100cells_quality import (
    broadcast_truth,
    discover_datasets,
    evaluate_cell,
    metric_block,
)


class EvaluateSchiclusterQualityTests(unittest.TestCase):
    def test_metric_block_reports_pcc_and_spearman_on_masked_values(self):
        pred = np.array([1.0, 2.0, 4.0, 8.0])
        truth = np.array([1.0, 3.0, 2.0, 7.0])
        mask = np.array([True, False, True, True])

        metrics = metric_block("heldout", pred, truth, mask)

        self.assertEqual(metrics["n_heldout"], 3)
        self.assertAlmostEqual(metrics["pcc_heldout"], np.corrcoef([1.0, 4.0, 8.0], [1.0, 2.0, 7.0])[0, 1])
        self.assertAlmostEqual(metrics["spearman_heldout"], 1.0)

    def test_evaluate_cell_splits_all_observed_and_heldout(self):
        pred = np.array([1.0, 2.0, 3.0, 4.0])
        truth = np.array([1.0, 2.0, 3.0, 4.0])
        observed = np.array([0.0, 5.0, 0.0, 7.0])

        metrics = evaluate_cell(pred, truth, observed)

        self.assertEqual(metrics["n_all"], 4)
        self.assertEqual(metrics["n_observed"], 2)
        self.assertEqual(metrics["n_heldout"], 2)
        self.assertAlmostEqual(metrics["pcc_all"], 1.0)
        self.assertAlmostEqual(metrics["spearman_all"], 1.0)
        self.assertAlmostEqual(metrics["pcc_observed"], 1.0)
        self.assertAlmostEqual(metrics["spearman_heldout"], 1.0)

    def test_broadcast_truth_repeats_single_consensus_row(self):
        truth = np.array([[10.0, 20.0, 30.0]])
        pred = np.zeros((3, 3))

        out = broadcast_truth(truth, pred)

        np.testing.assert_array_equal(
            out,
            np.array(
                [
                    [10.0, 20.0, 30.0],
                    [10.0, 20.0, 30.0],
                    [10.0, 20.0, 30.0],
                ]
            ),
        )

    def test_discover_datasets_is_python38_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "beads_300_W_0.5_level_1_T1.npz").write_text("")
            (root / "ignore.txt").write_text("")

            self.assertEqual(discover_datasets(root), ["beads_300_W_0.5_level_1_T1"])


if __name__ == "__main__":
    unittest.main()
