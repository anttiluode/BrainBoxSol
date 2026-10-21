import copy
import importlib.util
import unittest
import numpy as np

AVAILABLE=importlib.util.find_spec('local_slowness_gate') is not None
if AVAILABLE:import local_slowness_gate as gate


class AvailabilityTests(unittest.TestCase):
    def test_gate_exists(self):self.assertTrue(AVAILABLE,'experiment gate missing')


@unittest.skipUnless(AVAILABLE,'experiment not implemented')
class GateTests(unittest.TestCase):
    def test_background_noise_is_identical_within_each_class_triplet(self):
        p=gate.load_protocol();cfg=p['model'];d=gate.drives(81,1,2,cfg,tag=1)
        # After the cue ends, all three class currents share exactly the same nuisance.
        stop=int(round(cfg['cue_duration']/cfg['dt']))
        for i in range(0,6,3):
            np.testing.assert_array_equal(d[i,stop:],d[i+1,stop:])
            np.testing.assert_array_equal(d[i,stop:],d[i+2,stop:])

    def test_train_only_decoder_predicts_separate_examples(self):
        x=np.array([[0.,0.],[.1,0.],[1.,0.],[1.1,0.],[0.,1.],[0.,1.1]])
        y=np.array([0,0,1,1,2,2]);model=gate.fit_decoder(x,y,.002)
        np.testing.assert_array_equal(gate.predict(x,model),y)
        before=copy.deepcopy(model)
        gate.predict(x+100,model)
        self.assertEqual(before,model)
        with self.assertRaises(ValueError):gate.fit_decoder(x[:4],y[:4],.002)

    def test_full_run_requires_source_lock(self):
        with self.assertRaises(ValueError):gate.run_experiment('full')

    def test_smoke_controls_have_no_history_information_and_costs_match(self):
        r=gate.run_experiment('smoke');d=r['devices'][0]
        for row in d['conditions'].values():
            self.assertAlmostEqual(row['uniform']['accuracy'],1/3)
            self.assertAlmostEqual(row['noise_only']['accuracy'],1/3)
        self.assertEqual(d['costs']['local'],d['costs']['trace'])
        self.assertLess(d['max_material_budget_error'],1e-10)
        self.assertEqual(d['memory']['persistent_material_scalars_per_arm'],25)

    def test_replay_rejects_changed_prediction(self):
        r=gate.run_experiment('smoke')
        row=next(iter(r['devices'][0]['conditions'].values()))
        row['local']['predictions'][0]=(row['local']['predictions'][0]+1)%3
        with self.assertRaises(ValueError):gate.audit_receipt(r)

    def test_seed_partition_and_temporal_conditions_are_locked_in_protocol(self):
        p=gate.load_protocol();groups=list(p['seeds'].values())
        self.assertEqual(sum(map(len,groups)),len(set(sum(groups,[]))))
        self.assertEqual(p['evaluation']['primary_delay'],60.)
        self.assertEqual(p['evaluation']['primary_rewrite_delay'],0.)


if __name__=='__main__':unittest.main()
