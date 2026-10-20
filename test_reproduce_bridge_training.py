"""Independent training-audit regression controls."""
import copy
import importlib.util
import unittest
import numpy as np
import bridge_robustness as br

AVAILABLE = importlib.util.find_spec("reproduce_bridge_training") is not None
if AVAILABLE:
    import reproduce_bridge_training as audit


class AvailabilityTests(unittest.TestCase):
    def test_training_reproduction_auditor_exists(self):
        self.assertTrue(AVAILABLE, "training reproduction auditor is missing")


@unittest.skipUnless(AVAILABLE, "auditor not implemented")
class TrainingAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receipt = br.run_experiment("smoke")
        cls.observed = {}
        for d in cls.receipt["devices"]:
            cls.observed[str(d["seed"])] = {}
            for writer in ("standard", "noise_aware"):
                info = d["training"][writer]
                cls.observed[str(d["seed"])][writer] = {
                    "read_calls": info["read_calls"],
                    "node_system_solves": info["forward_solves"] + info["adjoint_solves"],
                    "update_solves": info["linear_update_solves"],
                }

    def test_identical_training_passes_without_mutating_receipt(self):
        before = copy.deepcopy(self.receipt)
        result = audit.compare_training(self.receipt, self.receipt, self.observed)
        self.assertTrue(result["passed"])
        self.assertEqual(result["writers_retrained"], 2)
        self.assertEqual(self.receipt, before)

    def test_rejects_intermediate_loss_tampering_even_with_identical_endpoints(self):
        bad = copy.deepcopy(self.receipt)
        bad["devices"][0]["training"]["standard"]["loss_before_each_update"][1] += .001
        with self.assertRaises(ValueError):
            audit.compare_training(bad, self.receipt, self.observed)

    def test_rejects_changed_conductance_and_nonfinite_values(self):
        for value in (.001, float("nan")):
            bad = copy.deepcopy(self.receipt)
            bad["devices"][0]["materials"]["standard"][0] += value
            with self.assertRaises(ValueError):
                audit.compare_training(bad, self.receipt, self.observed)

    def test_rejects_forged_count_even_if_both_receipts_agree(self):
        bad = copy.deepcopy(self.receipt)
        bad["devices"][0]["training"]["standard"]["forward_solves"] += 1
        with self.assertRaises(ValueError):
            audit.compare_training(bad, bad, self.observed)

    def test_counts_actual_linear_solves_and_restores_original_function(self):
        box = br.PhaseMedium(blocks=3, seed=18)
        base = box.values(box.initial)
        targets = box.values(br.teacher(box, [1,-1,1]))
        factors = np.ones((2, len(box.initial)))
        cfg = br.load_protocol()["training"] | {"steps":3, "worlds":2}
        original = np.linalg.solve
        state, info, counts = audit.fit_counted(br.fit_material, box, targets, base, factors, cfg)
        self.assertIs(np.linalg.solve, original)
        self.assertEqual(counts, {"read_calls":24,"node_system_solves":48,"update_solves":3})
        expected, expected_info = br.fit_material(box, targets, base, factors, cfg)
        np.testing.assert_array_equal(state, expected)
        self.assertEqual(info, expected_info)

    def test_rejects_wrong_seed_order(self):
        bad = copy.deepcopy(self.receipt)
        bad["devices"][0]["seed"] += 1
        with self.assertRaises(ValueError):
            audit.compare_training(bad, self.receipt, self.observed)


if __name__ == "__main__":
    unittest.main()
