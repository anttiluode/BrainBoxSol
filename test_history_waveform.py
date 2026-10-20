import unittest
import numpy as np
from adaptive_medium import make_grid,initial_material
from adaptive_medium_task import TaskConfig,sample_cue,make_drive


class WaveformTests(unittest.TestCase):
    def test_conditioner_matches_charge_and_energy_without_using_labels(self):
        import history_waveform as h
        t=np.arange(160)*.1
        a=h.match_waveform(np.exp(-t/2),.1)
        b=h.match_waveform(np.exp(-t/4),.1)
        self.assertTrue(np.all(a>0) and np.all(b>0))
        self.assertAlmostEqual(.1*a.sum(),16,places=11)
        self.assertAlmostEqual(.1*a@a,16*(1+.15**2),places=11)
        self.assertAlmostEqual(.1*b.sum(),.1*a.sum(),places=11)
        self.assertAlmostEqual(.1*b@b,.1*a@a,places=11)
        self.assertGreater(np.linalg.norm(a-b),.1)
        for bad in (np.ones(160),np.zeros(160),np.full(160,np.nan)):
            with self.assertRaises(ValueError):h.match_waveform(bad,.1)

    def test_history_write_and_erasure_reset_fast_state(self):
        import history_waveform as h
        cfg=TaskConfig();grid=make_grid();g=initial_material(grid,17)
        src=sample_cue(0,np.random.default_rng(5),cfg,evaluation=False)
        drive=make_drive(src,cfg)
        written=h.write_history(grid,g,[drive]*3,cfg,rate=.005)
        self.assertGreater(np.linalg.norm(written-g),1e-4)
        self.assertAlmostEqual(written.sum(),g.sum(),places=10)
        probe=h.probe_waveform(grid,written,drive,cfg)
        erased=h.probe_waveform(grid,g,drive,cfg)
        np.testing.assert_array_equal(erased,h.probe_waveform(grid,g,drive,cfg))
        self.assertGreater(np.linalg.norm(probe-erased),1e-6)
        np.testing.assert_array_equal(g,initial_material(grid,17))

    def test_receiver_is_a_causal_physical_filter_with_no_label_argument(self):
        import history_waveform as h
        from adaptive_medium import transition
        cfg=TaskConfig();grid=make_grid();g=initial_material(grid,19)
        y=np.zeros((2,160));y[0,0]=1;y[1,80]=1
        actual=h.receive_waveforms(grid,g,y,cfg)
        M=transition(grid,g,dt=cfg.dt,leak=cfg.leak)
        expected=[]
        for pulse in y:
            u=np.zeros(grid.nodes)
            for current in pulse:
                u=M@u;u[40]+=cfg.dt*current
            expected.append(u[40])
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-14)
        self.assertNotEqual(actual[0],actual[1])
        np.testing.assert_allclose(h.receive_waveforms(grid,g,np.ones((2,160)),cfg),
                                      np.repeat(h.receive_waveforms(grid,g,np.ones((1,160)),cfg),2),rtol=1e-12,atol=1e-14)

    def test_paired_histories_have_identical_probe_and_erased_signals(self):
        import history_waveform as h
        cfg=TaskConfig()
        a=h.make_dataset(401,2,cfg,split=1,history_pulses=2,write_rate=.005,noise=.02)
        b=h.make_dataset(401,2,cfg,split=2,history_pulses=2,write_rate=.005,noise=.02)
        np.testing.assert_array_equal(a['labels'],[0,1,0,1])
        np.testing.assert_array_equal(a['erased'][::2],a['erased'][1::2])
        self.assertGreater(np.linalg.norm(a['raw'][0]-a['raw'][1]),1e-4)
        self.assertNotEqual(a['history_digest'],b['history_digest'])
        self.assertEqual(a['memory']['box1_persistent_scalars'],272)
        np.testing.assert_array_equal(a['probe_drive'],b['probe_drive'])

    def test_threshold_is_fitted_on_training_only(self):
        import history_waveform as h
        model=h.fit_threshold(np.array([0.,.1,2.,2.1]),np.array([0,0,1,1]))
        np.testing.assert_array_equal(h.predict_threshold(np.array([.2,1.9]),model),[0,1])
        with self.assertRaises(ValueError):h.fit_threshold(np.array([0.,np.nan]),np.array([0,1]))


if __name__=='__main__':unittest.main()
