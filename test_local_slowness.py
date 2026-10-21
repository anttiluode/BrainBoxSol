"""Behavior and controls for the local-slowness model."""
import importlib.util
import unittest
import numpy as np
from adaptive_medium import make_grid, operator

AVAILABLE = importlib.util.find_spec("local_slowness") is not None
if AVAILABLE:
    import local_slowness as ls


class AvailabilityTests(unittest.TestCase):
    def test_model_exists(self):
        self.assertTrue(AVAILABLE, "local-slowness model missing")


@unittest.skipUnless(AVAILABLE, "model not implemented")
class LocalSlownessTests(unittest.TestCase):
    def setUp(self):
        self.grid=make_grid(3)
        self.A=operator(self.grid,np.full(len(self.grid.edges),.12),leak=.04)
        self.m=np.ones((2,9))

    def test_uniform_local_and_trace_dynamics_are_identical(self):
        drive=np.ones((2,5,9))
        a, av, ac=ls.evolve(self.A,self.m,drive,dt=.1,rate=0.,local=True)
        b, bv, bc=ls.evolve(self.A,self.m,drive,dt=.1,rate=0.,local=False)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(av,bv)
        self.assertEqual(ac,bc)

    def test_activity_writes_positive_fixed_budget_material_without_mutation(self):
        old=self.m.copy();drive=np.zeros((2,40,9));drive[:,:,0]=1
        material,voltage,cost=ls.evolve(self.A,self.m,drive,dt=.1,rate=.4,local=True)
        self.assertGreater(material[0,0],material[0,-1])
        self.assertTrue(np.all(material>0))
        np.testing.assert_allclose(material.mean(axis=1),1.,atol=1e-13)
        np.testing.assert_array_equal(old,self.m)
        self.assertEqual(cost,{"voltage_vector_steps":80,"material_vector_steps":80})

    def test_capacitance_analogy_does_not_change_dc_equilibrium(self):
        u=np.arange(1,10,dtype=float)
        equilibrium=np.linalg.solve(self.A,u)
        material=np.arange(1,10,dtype=float)[None,:]/5
        after=ls.voltage_step(self.A,equilibrium[None,:],material,u[None,:],.1)
        np.testing.assert_allclose(after[0],equilibrium,rtol=0,atol=1e-13)

    def test_waiting_preserves_budget_and_erodes_material_contrast(self):
        m=self.m.copy();m[:,0]=5;m[:,1:]=.5
        later,cost=ls.wait_material(m,20,dt=.1,rate=.04,local=True)
        self.assertLess(np.std(later),np.std(m))
        np.testing.assert_allclose(later.mean(axis=1),1.,atol=1e-13)
        self.assertEqual(cost,400)

    def test_probe_is_frozen_and_starts_from_zero(self):
        cfg={"dt":.1,"probe_duration":2.,"probe_pulse":1.,"probe_amplitude":1.,"sample_times":[1.,2.]}
        old=self.m.copy()
        values,cost=ls.probe(self.A,self.m,[0,4,8],cfg)
        repeated,_=ls.probe(self.A,self.m,[0,4,8],cfg)
        np.testing.assert_array_equal(values,repeated)
        np.testing.assert_array_equal(self.m,old)
        self.assertEqual(values.shape,(2,6))
        self.assertEqual(cost,40)

    def test_refining_time_step_converges_on_written_field_and_probe(self):
        drive=np.zeros((2,80,9));drive[:,:40,0]=1
        coarse,_,_=ls.evolve(self.A,self.m,drive,dt=.1,rate=.4,local=True)
        fine,_,_=ls.evolve(self.A,self.m,np.repeat(drive,2,axis=1),dt=.05,rate=.4,local=True)
        finer,_,_=ls.evolve(self.A,self.m,np.repeat(drive,4,axis=1),dt=.025,rate=.4,local=True)
        self.assertLess(np.max(np.abs(fine-finer)),np.max(np.abs(coarse-fine)))
        self.assertLess(np.max(np.abs(coarse-fine)),.04)

    def test_invalid_material_and_unstable_step_are_rejected(self):
        for value in (0.,-1.,float('nan')):
            m=self.m.copy();m[0,0]=value
            with self.assertRaises(ValueError):ls.voltage_step(self.A,np.zeros_like(m),m,np.zeros_like(m),.1)
        with self.assertRaises(ValueError):ls.evolve(self.A,self.m,np.ones((2,3,9)),dt=10.,rate=.4,local=True)

    def test_symmetric_but_non_rc_operators_are_rejected(self):
        for A in (np.array([[1.,2.],[2.,1.]]),np.array([[1.,-2.],[-2.,1.]])):
            with self.assertRaises(ValueError):
                ls.voltage_step(A,np.zeros((1,2)),np.ones((1,2)),np.zeros((1,2)),.1)


if __name__=="__main__":unittest.main()
