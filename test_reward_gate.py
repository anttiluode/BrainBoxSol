import copy
import unittest
import numpy as np
from reward_gate import summarize_reward, validate_reward_protocol


class RewardGateTests(unittest.TestCase):
    def test_all_reward_criteria_are_required(self):
        def row(reward=.95,frozen=.5,shuffled_reward=.5,random_updates=.5,erased=.5):
            a={k:{'accuracy':v,'invalid':0} for k,v in locals().copy().items()}
            return {'seed':211,'mappings':[{'arms':a,'numerical_passed':True,'invalid_training':0}]*2}
        good=row()
        self.assertTrue(summarize_reward([good])['passed'])
        for bad in (row(reward=.79),row(frozen=.8),row(shuffled_reward=.8),row(random_updates=.8),row(erased=.9)):
            self.assertFalse(summarize_reward([bad])['passed'])
        bad=copy.deepcopy(good);bad['mappings'][0]['numerical_passed']=False
        self.assertFalse(summarize_reward([bad])['passed'])
        with self.assertRaises(ValueError):summarize_reward([])

    def test_smoke_is_not_held_out_evidence(self):
        import json
        import tempfile
        from pathlib import Path
        from reward_gate import main
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'smoke.json'
            replay=Path(folder)/'replay.html'
            self.assertEqual(main(['--mode','smoke','--out',str(out),'--replay',str(replay)]),0)
            page=replay.read_text()
            self.assertIn('first evaluation seed',page)
            self.assertIn('one-bit rewards',page)
            self.assertNotIn('supervised learning',page)
            self.assertNotIn('six-parameter linear',page)
            receipt=json.loads(out.read_text())
            self.assertTrue(receipt['complete'])
            self.assertFalse(receipt['held_out'])
            self.assertFalse(receipt['scientific_gate_passed'])
            self.assertEqual([r['seed'] for r in receipt['seed_results']],[211])
            for m in receipt['seed_results'][0]['mappings']:
                for name in ('reward','shuffled_reward','random_updates'):
                    self.assertEqual(m['costs'][name]['feedback_bits'],8)
                    self.assertEqual(m['costs'][name]['adjoint_rollouts'],0)
            rejected=Path(folder)/'rejected.json'
            self.assertEqual(main(['--mode','full','--out',str(rejected)]),2)
            self.assertFalse(rejected.exists())

    def test_seed_split_and_budget_are_validated(self):
        from reward_gate import default_reward_protocol
        p=default_reward_protocol();validate_reward_protocol(p)
        for key,value in [('development_seeds',[301]),('heldout_seeds',[211]),
                          ('updates',0),('pairs',3),('thresholds',{'accuracy':.1})]:
            bad={**p,key:value}
            with self.assertRaises(ValueError):validate_reward_protocol(bad)


if __name__=='__main__':unittest.main()
