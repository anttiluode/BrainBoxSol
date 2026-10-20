import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from adaptive_medium import make_grid
from adaptive_medium_task import TaskConfig, episode_from_energies
from adaptive_medium_gate import (train_material, train_linear, make_evaluation_sources,
    evaluate_material, save_material, load_material, interventions, score_episodes)


class TrainingTests(unittest.TestCase):
    def test_motor_energy_summary_retains_measured_extrema(self):
        cfg=TaskConfig()
        rows=[episode_from_energies([1,2],cfg),episode_from_energies([3,4],cfg)]
        s=score_episodes(rows,np.array([1,1]))
        np.testing.assert_array_equal(s['energy_summary']['mean'],[2,3])
        np.testing.assert_array_equal(s['energy_summary']['min'],[1,2])
        np.testing.assert_array_equal(s['energy_summary']['max'],[3,4])

    def test_deterministic_writes_and_identical_training_examples(self):
        grid=make_grid(); cfg=TaskConfig()
        a=train_material(grid,11,(1,0),cfg,updates=3)
        b=train_material(grid,11,(1,0),cfg,updates=3)
        traffic=train_material(grid,11,(1,0),cfg,updates=3,mode='traffic')
        w,bias,cost=train_linear(11,(1,0),cfg,updates=3)
        np.testing.assert_array_equal(a.final,b.final)
        self.assertGreater(np.linalg.norm(a.final-a.initial),0)
        self.assertTrue(np.isfinite([r['loss'] for r in a.curve]).all())
        self.assertAlmostEqual(a.final.sum(),21.76,places=9)
        self.assertGreaterEqual(a.final.min(),.005-1e-12)
        self.assertLessEqual(a.final.max(),.4+1e-12)
        self.assertEqual(a.costs['input_digest'],traffic.costs['input_digest'])
        self.assertEqual(a.costs['input_digest'],cost['input_digest'])
        self.assertEqual(a.costs['forward_rollouts'],6)
        self.assertEqual(a.costs['adjoint_rollouts'],6)
        self.assertEqual(traffic.costs['adjoint_rollouts'],0)
        mats=interventions(a,11,2)
        np.testing.assert_array_equal(mats['erased'],a.initial)
        np.testing.assert_array_equal(np.sort(mats['shuffled'][0]),np.sort(a.final))

    def test_evaluation_and_serialization_do_not_carry_activity(self):
        grid=make_grid(); cfg=TaskConfig(); run=train_material(grid,23,(0,1),cfg,updates=2)
        sources,cues=make_evaluation_sources(23,cfg,3)
        before=run.final.copy()
        episodes=evaluate_material(grid,run.final,sources,cfg)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'material.json'; save_material(p,run.final,cfg)
            restored=load_material(p,grid,cfg)
            np.testing.assert_array_equal(restored,run.final)
            again=evaluate_material(grid,restored,sources,cfg)
            np.testing.assert_array_equal([x.energies for x in again],[x.energies for x in episodes])
            content=json.loads(p.read_text())
            for key,value in [('size',8),('conductances',[float('nan')]*272),('conductances',[.01]*272)]:
                corrupt={**content,key:value}; p.write_text(json.dumps(corrupt))
                with self.assertRaises(ValueError): load_material(p,grid,cfg)
            p.write_text('{bad')
            with self.assertRaises(ValueError): load_material(p,grid,cfg)
        np.testing.assert_array_equal(before,run.final)
        self.assertEqual(len(episodes),6)
        zero=episode_from_energies([0,0],cfg)
        tie=episode_from_energies([1,1],cfg)
        metrics=score_episodes([zero,tie],np.array([0,1]))
        self.assertEqual(metrics['accuracy'],0)
        self.assertEqual(metrics['invalid'],1)


class GateTests(unittest.TestCase):
    def test_protocol_cannot_silently_change_unused_constants_or_seed_split(self):
        from adaptive_medium_gate import main
        base=json.loads(Path('protocols/adaptive_medium.json').read_text())
        with tempfile.TemporaryDirectory() as d:
            for key,value in [('rate',.5),('thresholds',{'accuracy':.1}),('development_seeds',[101])]:
                p=Path(d)/'protocol.json'; p.write_text(json.dumps({**base,key:value}))
                self.assertEqual(main(['--mode','smoke','--protocol',str(p),'--out',str(Path(d)/'out.json')]),2)

    def test_cli_smoke_cannot_claim_heldout_success(self):
        from adaptive_medium_gate import main, write_json
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'smoke.json'
            self.assertEqual(main(['--mode','smoke','--out',str(out)]),0)
            r=json.loads(out.read_text())
            self.assertFalse(r['scientific_gate_passed'])
            self.assertFalse(r['held_out'])
            self.assertEqual(r['shuffle_count'],2)
            self.assertEqual(len(r['seed_results'][0]['mappings']),2)
            self.assertEqual(main(['--mode','full','--out',str(Path(d)/'bad.json')]),2)
            with self.assertRaises(ValueError): write_json(Path(d)/'nan.json',{'x':float('nan')})

    def test_verdict_uses_seeds_equally_and_each_conjunct(self):
        from adaptive_medium_gate import summarize
        def seed(learned=1., frozen=.5, erased=.5, shuffled=.5, invalid=0):
            arms={k:{'accuracy':v,'invalid':invalid,'count':200} for k,v in
                  [('learned',learned),('frozen',frozen),('erased',erased),('shuffled',shuffled)]}
            return {'seed':11,'mappings':[{'arms':arms},{'arms':arms}], 'numerical':{'passed':True}}
        checks={'passed':True}
        self.assertTrue(summarize([seed()],numerical_checks=checks)['passed'])
        for bad in (seed(learned=.7),seed(frozen=.9),seed(erased=.95),seed(shuffled=.95),seed(invalid=1)):
            self.assertFalse(summarize([bad],numerical_checks=checks)['passed'])
        self.assertFalse(summarize([seed()],numerical_checks={'passed':False})['passed'])
        x,y=seed(learned=1),seed(learned=.8)
        for row in y['mappings']: row['arms']['learned']['count']=2
        self.assertAlmostEqual(summarize([x,y],numerical_checks=checks)['arms']['learned']['mean'],.9)
        with self.assertRaises(ValueError): summarize([],numerical_checks=checks)

    def test_lock_rejects_wrong_sha_dirty_tree_and_untracked_protocol(self):
        from adaptive_medium_gate import verify_lock
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            def git(*args):
                return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
            git('init','-q'); git('config','user.email','test@example.com'); git('config','user.name','Test')
            names=['adaptive_medium.py','adaptive_medium_task.py','adaptive_medium_gate.py','adaptive_medium_replay.py']
            for name in names: (root/name).write_text('# fixture\n')
            p=root/'protocols/adaptive_medium.json'; p.parent.mkdir(); p.write_text('{"locked":true}')
            git('add','.'); git('commit','-qm','fixture'); sha=git('rev-parse','HEAD')
            verify_lock(p,sha,root=root)
            with self.assertRaises(ValueError): verify_lock(p,'0'*40,root=root)
            (root/names[0]).write_text('# changed\n')
            with self.assertRaises(ValueError): verify_lock(p,sha,root=root)
            git('checkout','--',names[0]); q=root/'other.json'; q.write_text('{"locked":true}')
            with self.assertRaises(ValueError): verify_lock(q,sha,root=root)


class ReplayTests(unittest.TestCase):
    def test_replay_embeds_real_states_and_escapes_metadata(self):
        from adaptive_medium_replay import write_replay, replay_episodes
        from dataclasses import asdict
        cfg=TaskConfig(); grid=make_grid(); run=train_material(grid,11,(1,0),cfg,updates=2)
        receipt={'config':asdict(cfg),'code_commit':'</script><script>bad()</script>',
            'mode':'development','seed_results':[{'seed':11,'mappings':[{'mapping':[1,0],
                'initial_material':run.initial.tolist(),'learned_material':run.final.tolist()}]}],
            'summary':{'arms':{'learned':{'mean':.5}},'passed':False}}
        episodes=replay_episodes(receipt)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'replay.html'; write_replay(p,grid=grid,receipt=receipt,episodes=episodes)
            text=p.read_text()
            payload=text.split('<script type="application/json" id="data">')[1].split('</script>')[0]
            data=json.loads(payload)
            np.testing.assert_array_equal(data['episodes'][0]['states'],episodes[0]['states'])
            self.assertNotIn('<script>bad()',text)
            self.assertNotIn('src="http',text)
            self.assertIn('<noscript>',text); self.assertIn('<table',text)
            bad=[{**episodes[0],'states':[]}]
            with self.assertRaises(ValueError): write_replay(p,grid=grid,receipt=receipt,episodes=bad)


if __name__=='__main__': unittest.main()
