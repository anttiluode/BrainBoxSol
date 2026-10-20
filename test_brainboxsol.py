import unittest

import numpy as np

from brainboxsol import Medium, run_gate, teacher_medium, write_one, write_protected


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

    def test_projected_update_disturbs_old_readout_less_than_global(self):
        box = self.box
        baseline = box.values(box.initial)
        target = box.values(teacher_medium(box))
        learned = box.initial.copy()
        for _ in range(80):
            write_one(box, learned, 0, target, baseline, 'global')
        old_response = box.read(learned, 0)[0]
        protected, global_writer = learned.copy(), learned.copy()
        write_protected(box, protected, 1, target, baseline, [0])
        write_one(box, global_writer, 1, target, baseline, 'global')
        protected_drift = abs(box.read(protected, 0)[0] - old_response)
        global_drift = abs(box.read(global_writer, 0)[0] - old_response)
        self.assertLess(protected_drift, global_drift)

    def test_impulse_has_no_hidden_fast_memory_between_queries(self):
        t = np.linspace(0., 10., 30)
        a = self.box.impulse(self.box.initial, 1, t)
        b = self.box.impulse(self.box.initial, 1, t)
        np.testing.assert_array_equal(a, b)
        self.assertAlmostEqual(a[0], 0., places=10)


if __name__ == '__main__':
    unittest.main()
