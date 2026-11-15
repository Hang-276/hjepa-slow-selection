import math
import unittest

import numpy as np

if __package__:
    from .metrics import risk, score_windows
else:
    from metrics import risk, score_windows


class WindowMetricTests(unittest.TestCase):
    def setUp(self):
        time = np.arange(4, dtype=float)[None, :, None]
        self.states = time * np.array([[1, 2, 3, 4], [2, 3, 1, 4]])[:, None, :]
        self.variance = np.array([1.0, 4.0, 9.0, 4.0])

    def test_oracle_uses_reference_variance_and_stable_axis_order(self):
        values = score_windows(self.states, self.states, self.variance, self.variance, [1, 2])
        np.testing.assert_allclose(values["true_scores"][0], [2.5, 2.5, 2.5, 10.0])
        np.testing.assert_array_equal(values["true_top2"], [[0, 1], [1, 2]])
        np.testing.assert_array_equal(values["true_top2"], values["predicted_top2"])
        self.assertEqual(values["true_gap"][0], 0)
        np.testing.assert_array_equal(values["wrong"], [False, False])
        np.testing.assert_array_equal(values["cross_term"], np.zeros((2, 4)))
        np.testing.assert_array_equal(values["error_energy"], np.zeros((2, 4)))

    def test_scoring_invariant_to_common_change_of_units(self):
        prediction = self.states + self.states[:, :, [2, 0, 3, 1]] * 0.1
        predicted_variance = self.variance + 0.3
        before = score_windows(prediction, self.states, self.variance, predicted_variance, [1, 2])
        scale, shift = np.array([2, -3, 0.5, 5]), np.array([-1, 1, 2, 9])
        after = score_windows(prediction * scale + shift, self.states * scale + shift,
                              self.variance * scale**2, predicted_variance * scale**2, [1, 2])
        for key in ("true_scores", "predicted_scores", "gap", "true_gap", "true_top2", "predicted_top2", "wrong"):
            np.testing.assert_allclose(before[key], after[key])

    def test_signed_terms_reconstruct_unnormalized_numerator(self):
        prediction = self.states * np.array([0.5, 2.0, -0.5, 1.0])
        predicted_variance = self.variance + 1
        values = score_windows(prediction, self.states, self.variance, predicted_variance, [1, 2])
        np.testing.assert_allclose(values["cross_term"][0], [-2.5, 20, -67.5, 0])
        np.testing.assert_allclose(values["error_energy"][0], [0.625, 10, 50.625, 0])
        np.testing.assert_allclose(values["predicted_scores"] * predicted_variance,
                                   values["true_scores"] * self.variance + values["cross_term"] + values["error_energy"])

    def test_invalid_reference_variance_and_lag(self):
        for invalid in ([1, 1, 1], [1, np.nan, 1, 1], [1, -1, 1, 1], [1, 1, 1, 0]):
            for physical, predicted in ((invalid, self.variance), (self.variance, invalid)):
                with self.subTest(physical=physical, predicted=predicted), self.assertRaises(ValueError):
                    score_windows(self.states, self.states, physical, predicted, [1])
        with self.assertRaises(ValueError):
            score_windows(self.states, self.states, self.variance, self.variance, [4])


class RiskTests(unittest.TestCase):
    def test_zero_errors_has_positive_bound_and_inclusive_threshold(self):
        result = risk([0.5, 0.6, 0.7, 0.4], [False, False, False, True], 0.5, 0.05, max_error=0.7)
        self.assertEqual((result["accepted"], result["errors"], result["coverage"]), (3, 0, 0.75))
        self.assertAlmostEqual(result["error_upper"], 1 - 0.05**(1 / 3))
        self.assertTrue(result["passes"])

    def test_one_error_in_two_has_known_exact_bound(self):
        result = risk([1, 2], [False, True], 0, 0.05)
        self.assertEqual(result["selective_error"], 0.5)
        self.assertAlmostEqual(result["error_upper"], math.sqrt(0.95))
        self.assertFalse(result["passes"])

    def test_all_wrong_and_no_accepted_bounds(self):
        full = risk([1, 2], [True, True], 0, 0.05)
        empty = risk([1, 2], [True, True], 3, 0.05)
        self.assertEqual((full["errors"], full["selective_error"], full["error_upper"]), (2, 1.0, 1.0))
        self.assertEqual((empty["accepted"], empty["selective_error"], empty["error_upper"]), (0, None, 1.0))
        self.assertFalse(empty["passes"])


if __name__ == "__main__":
    unittest.main()
