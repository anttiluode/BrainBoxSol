"""Deterministic invariance and negative-control tests for phase_bridge_gate."""
import unittest
import numpy as np
from phase_bridge_gate import PhaseMedium, fit, drift, homeostat, templates, trial


class PhaseBridgeTests(unittest.TestCase):
    def setUp(self):
        self.box = PhaseMedium(seed=2)
        self.trained,self.target = fit(self.box,[1,-1,1,-1,1,-1],steps=20)
        self.omega = .012
        self.ts = np.linspace(0,2*np.pi/self.omega+500,350)

    def test_protected_writer_learns_six_contents(self):
        base = self.box.values(self.box.initial)
        error = np.mean(np.abs(self.box.values(self.trained)-self.target)/base)
        self.assertLess(error,.005)

    def test_travelling_phase(self):
        phase = np.unwrap(self.box.phases(self.trained,self.omega))
        self.assertTrue(np.all(np.diff(phase)<-.20))

    def test_interface_total_preserved_but_microscopic_edges_change(self):
        damaged = drift(self.box,self.trained,.8,np.random.default_rng(8))
        repaired = homeostat(self.box,damaged,self.trained,'interface')
        for k in range(self.box.blocks-1):
            selected = self.box.interface==k
            self.assertAlmostEqual(repaired[selected].sum(),
                                   self.trained[selected].sum(),places=12)
            self.assertGreater(np.linalg.norm(repaired[selected]-self.trained[selected]),1e-6)

    def test_interface_homeostasis_preserves_response_and_phase_better(self):
        damaged = drift(self.box,self.trained,.8,np.random.default_rng(8))
        repaired = homeostat(self.box,damaged,self.trained,'interface')
        base = self.box.values(self.box.initial)
        bad = np.mean(np.abs(self.box.values(damaged)-self.target)/base)
        good = np.mean(np.abs(self.box.values(repaired)-self.target)/base)
        self.assertLess(good,bad)
        old_phase = self.box.phases(self.trained,self.omega)
        d_bad = np.abs(np.angle(np.exp(1j*(self.box.phases(damaged,self.omega)-old_phase))))
        d_good = np.abs(np.angle(np.exp(1j*(self.box.phases(repaired,self.omega)-old_phase))))
        self.assertLess(d_good.mean(),d_bad.mean())

    def test_templates_reproducible(self):
        for mode in ('linear','phase','phase_frozen','timed'):
            a,_=templates(self.box,self.trained,self.omega,self.ts,mode,gate_g=self.trained)
            b,_=templates(self.box,self.trained,self.omega,self.ts,mode,gate_g=self.trained)
            np.testing.assert_array_equal(a,b)

    def test_fixed_time_can_beat_phase_after_drift(self):
        damaged = drift(self.box,self.trained,1.,np.random.default_rng(8))
        _,phase = trial(self.box,self.trained,damaged,self.omega,self.ts,
                        'phase',np.random.default_rng(31),30)
        _,fixed = trial(self.box,self.trained,damaged,self.omega,self.ts,
                        'timed',np.random.default_rng(31),30)
        self.assertLess(fixed,phase)


if __name__=='__main__':
    unittest.main()
