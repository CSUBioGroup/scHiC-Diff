from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.sparse import csr_matrix, save_npz

from process_flamingo_100cells_schicluster import (
    dataset_from_stem,
    lower_triangle_rows,
    square_to_schicluster_rows,
)


class ProcessFlamingo100CellsTests(unittest.TestCase):
    def test_dataset_from_stem_groups_condition_without_slice_id(self):
        stem = "beads_300_W_0.5_level_1_consensus_2_slice_17"

        self.assertEqual(dataset_from_stem(stem), "beads_300_W_0.5_level_1_T2")

    def test_square_to_schicluster_rows_keeps_upper_triangle_nonzero_contacts(self):
        matrix = np.array(
            [
                [0.0, 2.5, 0.0],
                [2.5, 0.0, 3.5],
                [0.0, 3.5, 0.0],
            ]
        )

        rows = square_to_schicluster_rows(matrix)

        np.testing.assert_allclose(rows, np.array([[0.0, 1.0, 2.5], [1.0, 2.0, 3.5]]))

    def test_lower_triangle_rows_reads_upper_triangle_in_lower_triangle_feature_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            matrix = csr_matrix(
                np.array(
                    [
                        [0.0, 10.0, 20.0],
                        [0.0, 0.0, 30.0],
                        [0.0, 0.0, 0.0],
                    ],
                    dtype=np.float32,
                )
            )
            save_npz(tmp_path / "cell_1_chr19_mode.npz", matrix)

            rows = lower_triangle_rows([tmp_path / "cell_1_chr19_mode.npz"], n_bins=3)

            self.assertEqual(rows.shape, (1, 3))
            np.testing.assert_allclose(rows.toarray(), np.array([[10.0, 20.0, 30.0]]))


if __name__ == "__main__":
    unittest.main()
