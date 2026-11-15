"""Pure comparison/selection checks. No real-data parity or model execution."""
import unittest

import numpy as np
import pandas as pd

from scripts.validation_parity import compare_predictions, require_target_identity, selected_indices


class ValidationParityTests(unittest.TestCase):
    def test_samples_span_the_cohort_without_duplicates(self):
        np.testing.assert_array_equal(selected_indices(11, 3), [0, 5, 10])
        np.testing.assert_array_equal(selected_indices(3, 20), [0, 1, 2])
        np.testing.assert_array_equal(selected_indices(3, 'all'), [0, 1, 2])
        for bad in (0, -1, True):
            with self.assertRaises(ValueError):
                selected_indices(3, bad)

    def test_equal_probabilities_preserve_every_metric(self):
        values = np.eye(5) * .8 + .04
        result = compare_predictions(values, values.copy(), np.arange(5))
        self.assertTrue(result['passed'])
        self.assertEqual(result['maximum_probability_difference'], 0)

    def test_small_probability_change_cannot_hide_an_argmax_flip(self):
        first = np.array([[.4999999, .4999998, .0000001, .0000001, .0000001]])
        second = first[:, [1, 0, 2, 3, 4]]
        result = compare_predictions(first, second, [0])
        self.assertTrue(result['probabilities_within_tolerance'])
        self.assertFalse(result['passed'])
        self.assertEqual(result['argmax_mismatches'], 1)

    def test_invalid_probabilities_and_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            compare_predictions(np.ones((1, 5)), np.ones((1, 5)), [0])
        with self.assertRaises(ValueError):
            compare_predictions(np.eye(5), np.eye(4), np.arange(5))

    def test_identity_includes_labels_and_order(self):
        frame = pd.DataFrame({'dataset': ['sleepedf'] * 2, 'participant_id': ['SC401'] * 2,
                              'recording_id': ['SC4011'] * 2, 'epoch_id': ['id0', 'id1'],
                              'epoch_index': [0, 1], 'true_stage': ['W', 'N1']})
        require_target_identity(frame, frame.copy())
        with self.assertRaises(ValueError):
            require_target_identity(frame, frame.iloc[::-1])
        changed = frame.copy()
        changed.loc[0, 'true_stage'] = 'N2'
        with self.assertRaises(ValueError):
            require_target_identity(frame, changed)


if __name__ == '__main__':
    unittest.main()
