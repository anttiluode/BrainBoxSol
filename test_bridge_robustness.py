"""Behavioral checks for bridge-noise writing and repair, written before implementation."""
import copy
import importlib.util
import unittest
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np
from phase_bridge_gate import PhaseMedium, teacher, homeostat

AVAILABLE = importlib.util.find_spec("bridge_robustness") is not None
if AVAILABLE:
    import bridge_robustness as br

class AvailabilityTests(unittest.TestCase):
    def test_bridge_robustness_feature_exists(self):
        self.assertTrue(AVAILABLE, "bridge_robustness implementation is missing")

@unittest.skipUnless(AVAILABLE, "implementation not present yet")
class BridgeRobustnessTests(unittest.TestCase):
    def setUp(self):
        self.box = PhaseMedium(blocks=3, seed=17)
        self.base = self.box.values(self.box.initial)
        self.target = self.box.values(teacher(self.box, [1,-1,1]))

    def test_bridge_corruption_leaves_interior_and_input_unchanged(self):
        initial = self.box.initial.copy()
        f = br.noise_factors(self.box, np.random.default_rng(5), "mixed", .2, 4)
        np.testing.assert_array_equal(f[:, self.box.owner >= 0], 1.)
        self.assertTrue(np.all(f > 0))
        self.assertGreater(np.std(f[:, self.box.owner == -1]), .01)
        np.testing.assert_array_equal(self.box.initial, initial)

    def test_zero_noise_is_exact_identity(self):
        f = br.noise_factors(self.box, np.random.default_rng(5), "mixed", 0., 4)
        np.testing.assert_array_equal(f, np.ones_like(f))

    def test_interior_challenge_does_not_change_bridges(self):
        f = br.noise_factors(self.box, np.random.default_rng(5), "interior", .2, 4)
        np.testing.assert_array_equal(f[:, self.box.owner == -1], 1.)
        self.assertGreater(np.std(f[:, self.box.owner >= 0]), .01)

    def test_corrupted_objective_jacobian_matches_finite_difference(self):
        f = (br.noise_factors(self.box, np.random.default_rng(9), "mixed", .25, 3)
             * br.noise_factors(self.box, np.random.default_rng(19), "interior", .2, 3))
        state = self.box.initial.copy()
        residual, jac = br.residual_jacobian(self.box, state, self.target, self.base, f)
        for col, edge in enumerate(np.flatnonzero(self.box.owner >= 0)):
            plus, minus = state.copy(), state.copy()
            plus[edge] += 1e-5
            minus[edge] -= 1e-5
            rp = br.residual_jacobian(self.box, plus, self.target, self.base, f)[0]
            rm = br.residual_jacobian(self.box, minus, self.target, self.base, f)[0]
            np.testing.assert_allclose(jac[:,col], (rp-rm)/2e-5, rtol=2e-5, atol=2e-8)

    def test_clean_writer_reduces_target_error_without_bridge_writes(self):
        f = np.ones((4, len(self.box.initial)))
        cfg = br.load_protocol()["training"] | {"steps":16, "worlds":4}
        trained, info = br.fit_material(self.box, self.target, self.base, f, cfg)
        self.assertLess(info["final_loss"], info["initial_loss"] * .7)
        np.testing.assert_array_equal(trained[self.box.owner == -1], self.box.initial[self.box.owner == -1])
        self.assertTrue(np.all((trained[self.box.owner >= 0] >= .1) & (trained[self.box.owner >= 0] <= 2.)))

    def test_zero_noise_training_matches_clean_with_equal_cost(self):
        cfg = br.load_protocol()["training"] | {"steps":4, "worlds":3}
        clean = np.ones((3, len(self.box.initial)))
        noised = br.noise_factors(self.box, np.random.default_rng(9), "mixed", 0, 3)
        a, ai = br.fit_material(self.box, self.target, self.base, clean, cfg)
        b, bi = br.fit_material(self.box, self.target, self.base, noised, cfg)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(ai["read_calls"], bi["read_calls"])
        self.assertEqual(ai["read_calls"], (cfg["steps"]+1)*3*self.box.blocks)

    def test_repair_preserves_each_interface_and_does_not_mutate_source(self):
        factors = br.noise_factors(self.box, np.random.default_rng(3), "mixed", .6, 1)[0]
        damaged = self.box.initial * factors
        before = damaged.copy()
        fixed = homeostat(self.box, damaged, self.box.initial, "interface")
        np.testing.assert_array_equal(damaged,before)
        np.testing.assert_array_equal(fixed[self.box.owner >= 0],damaged[self.box.owner >= 0])
        for k in range(self.box.blocks-1):
            mask = self.box.interface == k
            self.assertAlmostEqual(fixed[mask].sum(), self.box.initial[mask].sum(), places=12)

    def test_development_and_full_devices_are_disjoint(self):
        p = br.load_protocol()
        used = []
        for part in ["smoke","development","full"]:
            values = p["seeds"][part]
            self.assertFalse(set(used) & set(values))
            used.extend(values)


    def test_relative_reduction_is_unresolved_at_numerical_floor(self):
        receipt = br.run_experiment("smoke")
        cfg = receipt["effective_config"]
        for device in receipt["devices"]:
            for row in device["evaluation"]:
                if (row["family"] == cfg["evaluation"]["primary_family"]
                        and row["strength"] == cfg["evaluation"]["primary_strength"]):
                    row["target_errors"]["standard_repaired"] = [0.] * len(row["factors"])
        summary = br.summarize(receipt["devices"], cfg, False)
        claim = summary["claims"]["combined_vs_repair"]
        self.assertIsNone(claim["relative_error_reduction"])
        self.assertFalse(claim["criteria"]["relative_error_reduction"])


    def test_cost_audit_rejects_forged_update_count(self):
        receipt = br.run_experiment("smoke")
        receipt["devices"][0]["training"]["standard"]["linear_update_solves"] += 1
        with self.assertRaises(ValueError):
            br.audit_receipt(receipt)

    def test_full_audit_rejects_invalid_source_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in br.SOURCE_FILES:
                dest = root/name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(br.ROOT/name, dest)
            protocol = br.load_protocol()
            protocol["seeds"]["full"] = [9001]
            protocol["training"].update(steps=2, worlds=2)
            protocol["evaluation"].update(draws=1)
            (root/"protocols/bridge_robustness.json").write_text(json.dumps(protocol))
            def git(*args):
                return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
            git("init")
            git("add", ".")
            git("-c","user.name=Test","-c","user.email=test@example.invalid",
                "commit","-m","test lock")
            lock = git("rev-parse","HEAD")
            subprocess.run([sys.executable,"bridge_robustness.py","--mode","full",
                            "--lock-commit",lock,"--out","receipt.json"],
                           cwd=root, check=True, stdout=subprocess.DEVNULL)
            changed = json.loads((root/"receipt.json").read_text())
            changed["lock_commit"] = "bogus"
            (root/"receipt.json").write_text(json.dumps(changed))
            result = subprocess.run(
                [sys.executable,"-c",
                 "import json; import bridge_robustness as b; b.audit_receipt(json.load(open('receipt.json')))"],
                cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertNotEqual(result.returncode, 0, "forged full-run lock incorrectly passed audit")

    def test_full_run_rejects_missing_lock_before_evaluation(self):
        with self.assertRaisesRegex(ValueError, "lock"):
            br.run_experiment("full")

    def test_smoke_receipt_is_recomputable_and_tampering_is_rejected(self):
        receipt = br.run_experiment("smoke")
        audit = br.audit_receipt(receipt)
        self.assertTrue(audit["passed"])
        self.assertFalse(receipt["held_out"])
        self.assertFalse(receipt["scientific_claims_evaluated"])
        for row in receipt["devices"]:
            self.assertEqual(row["training"]["standard"]["read_calls"], row["training"]["noise_aware"]["read_calls"])
        bad = copy.deepcopy(receipt)
        bad["devices"][0]["clean_errors"]["standard"] += .01
        with self.assertRaises(ValueError):
            br.audit_receipt(bad)

if __name__ == "__main__":
    unittest.main()
