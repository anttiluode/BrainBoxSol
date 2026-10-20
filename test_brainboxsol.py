import unittest

import numpy as np

from brainboxsol import Medium, run_gate


class MediumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.box = Medium()
        cls.receipt = run_gate()

    def test_spectrum_is_hierarchical(self):
        eig = self.box.spectrum(self.box.initial)
        self.assertGreater(eig[4] / eig[3], 10)
        self.assertTrue(np.all(eig > 0))

    def test_local_adjoint_matches_finite_difference(self):
        self.assertLess(self.receipt['finite_difference_max_abs_error'], 1e-4)

    def test_learning_does_not_mutate_bridges(self):
        self.assertTrue(self.receipt['bridge_unchanged_during_training'])

    def test_supervised_local_fit_for_frozen_toy(self):
        errors = self.receipt['final_mean_abs_relative_error']
        self.assertLess(errors['local'], errors['frozen'])
        self.assertLess(errors['local'], errors['global'])

    def test_ungauged_bridge_drift_is_detectable(self):
        e0 = np.linalg.norm(self.receipt['pre_drift_error'])
        e1 = np.linalg.norm(self.receipt['post_local_bridge_drift_error'])
        self.assertGreater(e1, e0 * 5)

    def test_impulse_has_no_hidden_fast_memory_between_queries(self):
        t = np.linspace(0., 10., 30)
        a = self.box.impulse(self.box.initial, 1, t)
        b = self.box.impulse(self.box.initial, 1, t)
        np.testing.assert_array_equal(a, b)
        self.assertAlmostEqual(a[0], 0., places=10)


if __name__ == '__main__':
    unittest.main()
