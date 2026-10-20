import unittest
import numpy as np
from adaptive_medium import make_grid, operator, rollout, project_budget, initial_material


class NumericalTests(unittest.TestCase):
    def test_graph_and_operator(self):
        grid = make_grid(9)
        self.assertEqual(grid.nodes, 81)
        self.assertEqual(len(grid.edges), 272)
        g = initial_material(grid, 11)
        A = operator(grid, g, leak=.02)
        np.testing.assert_allclose(A, A.T, atol=1e-14)
        self.assertGreater(np.linalg.eigvalsh(A).min(), 0)
        pairs = {tuple(e) for e in grid.edges}
        self.assertEqual(len(pairs), 272)
        self.assertNotIn((0, 8), pairs)

    def test_conservation_decay_and_locality(self):
        grid = make_grid(9)
        g = np.full(272, .08)
        drive = np.zeros((80, 81)); drive[0, 0] = 10
        states = rollout(grid, g, drive, dt=.1, leak=0)
        np.testing.assert_allclose(states[1:].sum(axis=1), 1, atol=1e-12)
        self.assertEqual(set(np.flatnonzero(states[2])), {0, 1, 9, 10})
        leaky = rollout(grid, g, drive, dt=.1, leak=.02)
        self.assertLess(np.linalg.norm(leaky[-1]), np.linalg.norm(leaky[1]))
        np.testing.assert_array_equal(states[0], np.zeros(81))
        np.testing.assert_array_equal(states, rollout(grid, g, drive, dt=.1, leak=0))

    def test_projection_is_feasible_and_preserves_input(self):
        x = np.array([-.5, .2, 2.]); before = x.copy()
        y = project_budget(x, .5, .005, .4)
        np.testing.assert_allclose(y, [.005, .095, .4], atol=1e-12)
        np.testing.assert_array_equal(x, before)
        for budget in (.015, 1.2):
            p = project_budget(x, budget, .005, .4)
            self.assertAlmostEqual(p.sum(), budget, places=12)
        with self.assertRaises(ValueError): project_budget(x, 2., .005, .4)
        with self.assertRaises(ValueError): project_budget(np.array([np.nan]), .1, .005, .4)

    def test_invalid_inputs_fail_before_rollout(self):
        with self.assertRaises(ValueError): make_grid(1)
        grid = make_grid(3); g = initial_material(grid, 1)
        drive = np.zeros((4, 9))
        for dt, leak in ((0, .02), (.1, -1), (10, .02), (np.nan, .02)):
            with self.subTest(dt=dt, leak=leak), self.assertRaises(ValueError):
                rollout(grid, g, drive, dt=dt, leak=leak)
        for bad in (g[:-1], np.full_like(g, np.nan), -g):
            with self.assertRaises(ValueError): operator(grid, bad, leak=.02)
        with self.assertRaises(ValueError): rollout(grid, g, np.full((4,9), np.inf), dt=.1, leak=.02)
        with self.assertRaises(ValueError): rollout(grid, g, np.zeros((4,8)), dt=.1, leak=.02)


class SensoryGradientTests(unittest.TestCase):
    def test_cues_timing_and_motor_integration(self):
        from adaptive_medium_task import TaskConfig, sample_cue, make_drive, read_motor
        cfg = TaskConfig()
        for cue, rows in ((0, {1,2,3}), (1, {5,6,7})):
            s = sample_cue(cue, np.random.default_rng(12), cfg, evaluation=False)
            self.assertTrue(.8 <= s.sum() <= 1.2)
            self.assertTrue(set(np.flatnonzero(s)//9) <= rows)
            self.assertTrue(set(np.flatnonzero(s)%9) <= {0,1,2})
            np.testing.assert_array_equal(s, sample_cue(cue,np.random.default_rng(12),cfg,evaluation=False))
            self.assertFalse(np.array_equal(s,sample_cue(cue,np.random.default_rng(12),cfg,evaluation=True)))
            drive = make_drive(s,cfg)
            self.assertEqual(np.count_nonzero(drive.sum(axis=1)), 24)
        states = np.zeros((161,81)); states[113:, 2*9+7] = 2
        states[112,2*9+7] = 1000  # excluded
        episode = read_motor(states,cfg)
        np.testing.assert_allclose(episode.energies, [9.6,0], atol=1e-12)
        self.assertEqual(episode.action,0)
        self.assertFalse(read_motor(np.zeros_like(states),cfg).valid)
        states[113:,6*9+7] = 2
        self.assertIsNone(read_motor(states,cfg).action)

    def test_transient_gradient_and_exact_map(self):
        from adaptive_medium import loss_gradient, motor_response_map
        from adaptive_medium_task import TaskConfig, sample_cue, make_drive, read_motor
        cfg=TaskConfig(); grid=make_grid(); g=initial_material(grid,23)
        before=g.copy(); source=sample_cue(0,np.random.default_rng(7),cfg,evaluation=False)
        drive=make_drive(source,cfg)
        for sign in (-1,1):
            loss,ep,grad=loss_gradient(grid,g,drive,cfg,sign)
            for edge in (0,30,140,230,271):
                plus,minus=g.copy(),g.copy(); plus[edge]+=1e-6; minus[edge]-=1e-6
                lp=loss_gradient(grid,plus,drive,cfg,sign)[0]
                lm=loss_gradient(grid,minus,drive,cfg,sign)[0]
                np.testing.assert_allclose(grad[edge],(lp-lm)/2e-6,rtol=2e-4,atol=1e-7)
            states=rollout(grid,g,drive,dt=cfg.dt,leak=cfg.leak)
            np.testing.assert_allclose(motor_response_map(grid,g,cfg)@source,
                                       ep.energies,rtol=1e-10,atol=1e-12)
            np.testing.assert_array_equal(g,before)
            other=make_drive(sample_cue(1,np.random.default_rng(8),cfg,evaluation=False),cfg)
            rollout(grid,g,other,dt=cfg.dt,leak=cfg.leak)
            np.testing.assert_array_equal(states,rollout(grid,g,drive,dt=cfg.dt,leak=cfg.leak))
            saved=states.copy(); drive[:]=0
            np.testing.assert_array_equal(states,saved)
            drive=make_drive(source,cfg)


if __name__ == '__main__': unittest.main()
