import unittest
from unittest.mock import patch
import numpy as np
from adaptive_medium import make_grid, initial_material, rollout
from adaptive_medium_task import TaskConfig, sample_cue, make_drive, read_motor
from reward_medium import RewardWriter, reward_trial, train_reward_material


class RewardTests(unittest.TestCase):
    def test_boolean_feedback_is_the_same_one_bit_signal(self):
        cfg=TaskConfig();g=initial_material(make_grid(),11)
        a=RewardWriter(g,cfg,seed=11,pairs=2);b=RewardWriter(g,cfg,seed=11,pairs=2)
        a.propose();b.propose();bits=np.array([[True,False],[False,True]])
        a.observe(bits);b.observe(bits.astype(int))
        np.testing.assert_array_equal(a.material,b.material)

    def test_forward_trial_matches_full_rollout(self):
        grid=make_grid();cfg=TaskConfig();g=initial_material(grid,11)
        s=sample_cue(0,np.random.default_rng(4),cfg,evaluation=False)
        before=g.copy(); ep=reward_trial(grid,g,s,cfg)
        expected=read_motor(rollout(grid,g,make_drive(s,cfg),dt=cfg.dt,leak=cfg.leak),cfg)
        np.testing.assert_allclose(ep.energies,expected.energies,rtol=1e-12,atol=1e-14)
        np.testing.assert_array_equal(g,before)

    def test_writer_only_accepts_bits_and_preserves_budget(self):
        grid=make_grid();cfg=TaskConfig();g=initial_material(grid,11)
        w=RewardWriter(g,cfg,seed=11,pairs=4)
        p=w.propose();self.assertEqual(p.shape,(4,2,272))
        np.testing.assert_allclose(p.sum(axis=2),21.76,atol=1e-9)
        self.assertTrue(np.all((p>=.005-1e-12)&(p<=.4+1e-12)))
        for bad in (np.ones((4,2))*.5,np.ones((4,1)),np.full((4,2),np.nan)):
            with self.assertRaises(ValueError):w.observe(bad)
        bits=np.array([[1,0],[0,1],[1,0],[0,1]])
        w.observe(bits)
        self.assertGreater(np.linalg.norm(w.material-g),0)
        self.assertAlmostEqual(w.material.sum(),21.76,places=9)
        with self.assertRaises(ValueError):w.observe(bits)
        twin=RewardWriter(g,cfg,seed=11,pairs=4)
        np.testing.assert_array_equal(twin.propose(),p);twin.observe(bits)
        np.testing.assert_array_equal(twin.material,w.material)

    def test_reward_training_uses_no_target_gradient_and_is_repeatable(self):
        cfg=TaskConfig();grid=make_grid()
        with patch('adaptive_medium_gate.loss_gradient',side_effect=AssertionError('teacher gradient leaked')):
            a=train_reward_material(grid,11,(1,0),cfg,updates=3,pairs=4)
            b=train_reward_material(grid,11,(1,0),cfg,updates=3,pairs=4)
            control=train_reward_material(grid,11,(1,0),cfg,updates=3,pairs=4,mode='shuffled_reward')
        np.testing.assert_array_equal(a.final,b.final)
        self.assertEqual(a.costs['adjoint_rollouts'],0)
        self.assertEqual(a.costs['forward_rollouts'],24)
        self.assertEqual(a.costs['input_digest'],control.costs['input_digest'])
        self.assertEqual(a.costs['feedback_bits'],24)


if __name__=='__main__':unittest.main()
