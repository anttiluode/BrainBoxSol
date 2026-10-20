"""Tests for G3: dynamic observability without a static boundary difference."""
import unittest
import numpy as np
from outer_lens import (
    paths, sample_response, dc_conductance, feedthrough, run_gate,
    train_arx, forecast_arx, probes,
)


class OuterResponseTests(unittest.TestCase):
    def test_dc_and_immediate_indistinguishability(self):
        self.assertEqual(dc_conductance('A'),dc_conductance('B'))
        self.assertEqual(feedthrough('A'),feedthrough('B'))
        self.assertEqual(dc_conductance('A'),1.)
        self.assertEqual(feedthrough('A'),2.)
        self.assertNotEqual(paths('A'),paths('B'))

    def test_early_and_late_agree_but_middle_differs(self):
        u=np.ones(5000)
        a,b=sample_response(u,'A'),sample_response(u,'B')
        self.assertAlmostEqual(a[0],b[0],places=12)
        self.assertAlmostEqual(a[-1],b[-1],places=5)
        self.assertGreater(float(np.max(abs(a-b))),.01)

    def test_reduced_order_of_hidden_dynamics(self):
        u=probes(10,2500,17)
        v=probes(11,800,6)
        for h in 'AB':
            y=sample_response(u,h)
            true=sample_response(v,h)
            pred1=forecast_arx(v,train_arx(u,y,1),1)
            pred2=forecast_arx(v,train_arx(u,y,2),2)
            err1=np.sqrt(np.mean((true-pred1)**2))
            err2=np.sqrt(np.mean((true-pred2)**2))
            self.assertLess(err2,1e-8)
            if h=='A':
                self.assertLess(err1,1e-8)
            else:
                self.assertGreater(err1,1e-3)

    def test_noisy_measurements_and_matched_coefficient_fir(self):
        result=run_gate(seed=22,train_noise=.01)
        for h in 'AB':
            for drive in ('rapid','slow','wave'):
                row=result['test'][h+':'+drive]
                self.assertLess(row['arx2_rmse'],row['fir5_rmse'])
        accuracy=result['classification_accuracy']['0.05']
        self.assertEqual(accuracy['dc'],.5)
        self.assertGreater(accuracy['arx2'],.95)


if __name__=='__main__':
    unittest.main()
