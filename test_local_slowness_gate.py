import copy
import importlib.util
import unittest
from unittest import mock
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

    def test_retention_gate_accounts_for_initial_readability(self):
        r=gate.run_experiment('smoke');d=r['devices'][0];p=r['effective_config']
        d['conditions']['retention:0']['local']['accuracy']=1.
        d['conditions']['retention:0']['trace']['accuracy']=.8
        d['conditions']['retention:2']['local']['accuracy']=.8
        d['conditions']['retention:2']['trace']['accuracy']=.6
        s=gate.summarize([d],p,False)
        self.assertAlmostEqual(s['retention_gain_local_minus_trace'],0.)

    def test_rewrite_gate_rejects_two_equally_bad_rewriters(self):
        r=gate.run_experiment('smoke');d=r['devices'][0]
        d['conditions']['rewrite:0']['local']['accuracy']=.1
        d['conditions']['rewrite:0']['trace']['accuracy']=.1
        s=gate.summarize([d],r['effective_config'],False)
        self.assertFalse(s['gates']['retention_without_rewrite_cost']['criteria'].get('rewrite_quality',True))

    def test_refinement_uses_actual_training_triplet_and_all_primary_phases(self):
        p=gate.effective(gate.load_protocol(),'smoke');d=gate.run_device(81,p)
        drive=gate.drives(81,1,p['evaluation']['training_pairs'],p['model'],tag=1)[:3]
        self.assertIn('training_drive_samples',d['refinement'])
        np.testing.assert_allclose(d['refinement']['training_drive_samples'],drive[:,[0,-1],:],rtol=1e-12,atol=1e-14)
        for writer in ('local','trace'):
            self.assertEqual(set(d['refinement']['by_writer'][writer]),{'after_write','after_wait','after_rewrite'})

    def test_source_change_during_execution_is_rejected(self):
        hashes=gate.source_hashes()
        with mock.patch.object(gate,'source_hashes',side_effect=[hashes,hashes|{'changed':'source'}]):
            with self.assertRaisesRegex(ValueError,'source changed'):
                gate.run_experiment('smoke')

    def test_full_replay_rejects_unlocked_protocol_before_running_devices(self):
        r=gate.run_experiment('smoke');r.update(mode='full',held_out=True,
            lock_commit='436feaec27ae13e3b272e70dcaf3fdd0febf99ef')
        r['protocol']['locked']=False
        with self.assertRaisesRegex(ValueError,'locked protocol'):
            gate.audit_receipt(r)


if __name__=='__main__':unittest.main()
