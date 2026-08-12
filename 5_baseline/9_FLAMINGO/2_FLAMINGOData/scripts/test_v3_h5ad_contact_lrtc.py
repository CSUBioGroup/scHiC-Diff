from __future__ import annotations

import unittest

import numpy as np

from v3_h5ad_contact_lrtc import feature_row_to_matrix, matrix_to_feature_row


class CanonicalFeatureOrderTests(unittest.TestCase):
    def test_triu_feature_row_round_trip_preserves_coordinates(self):
        expected = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [1.0, 0.0, 4.0, 5.0],
                [2.0, 4.0, 0.0, 6.0],
                [3.0, 5.0, 6.0, 0.0],
            ]
        )
        triu = expected[np.triu_indices(4, k=1)]

        reconstructed = feature_row_to_matrix(triu, 4, feature_order="triu")
        np.testing.assert_array_equal(reconstructed, expected)
        np.testing.assert_array_equal(
            matrix_to_feature_row(reconstructed, 4, feature_order="triu"),
            triu,
        )

    def test_rejects_unsupported_feature_order(self):
        with self.assertRaisesRegex(ValueError, "Unsupported feature order"):
            feature_row_to_matrix(np.zeros(6), 4, feature_order="diagonal")


if __name__ == "__main__":
    unittest.main()
