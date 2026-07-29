#!/usr/bin/env python3

import unittest

import numpy as np

from gem_control.full_mpc import (
    FullLearnedMpc,
    FullMpcConfig,
    FullMpcInitialCondition,
    load_full_mpc_config,
)
from gem_control.reference_path import build_configured_reference_path


class FullLearnedMpcTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = build_configured_reference_path()[0]
        reference = cls.path.evaluate(np.asarray((20.0,)))
        cls.state = np.asarray(
            (
                reference.x[0],
                reference.y[0],
                reference.yaw[0],
                2.0,
                2.0 * reference.curvature[0],
            )
        )
        cls.command = np.asarray(
            (
                2.0,
                np.clip(
                    np.arctan(1.75 * reference.curvature[0]),
                    -0.3,
                    0.3,
                ),
            )
        )
        cls.history = np.tile(
            np.r_[cls.state[3:5], cls.command], (2, 1)
        )

    def make_mpc(self):
        return FullLearnedMpc(
            FullMpcConfig(
                horizon_steps=3,
                computation_budget_s=0.5,
                reference_speed_mps=2.0,
            ),
            reference_path=self.path,
        )

    def test_repository_configuration_loads_timing_and_horizon(self):
        config = load_full_mpc_config()
        self.assertEqual(config.horizon_steps, 12)
        self.assertAlmostEqual(config.commissioned_takeover_delay_s, 0.005)
        self.assertAlmostEqual(config.computation_budget_s, 0.08)

    def test_solver_returns_a_feasible_direct_command(self):
        mpc = self.make_mpc()
        result = mpc.solve(
            FullMpcInitialCondition(
                state=self.state,
                fixed_history_z=self.history,
                previous_command=self.command,
                previous_progress_m=20.0,
            )
        )
        self.assertTrue(result.success, result.diagnostics)
        self.assertEqual(result.controls.shape, (3, 2))
        self.assertEqual(result.states.shape, (4, 5))
        self.assertLessEqual(
            result.diagnostics.maximum_constraint_violation, 2e-3
        )
        self.assertGreaterEqual(result.first_command[0], 0.0)
        self.assertLessEqual(result.first_command[0], 5.5)
        self.assertLessEqual(abs(result.first_command[1]), 0.3)

    def test_warm_start_shifts_commands_and_repeats_last(self):
        mpc = self.make_mpc()
        initial = FullMpcInitialCondition(
            state=self.state,
            fixed_history_z=self.history,
            previous_command=self.command,
            previous_progress_m=20.0,
        )
        first = mpc.solve(initial)
        self.assertTrue(first.success)
        shifted, progress, source = mpc._initial_guess(20.0, 2.0)
        self.assertEqual(source, "shifted_previous_solution")
        np.testing.assert_allclose(shifted[:-1], first.controls[1:])
        np.testing.assert_allclose(shifted[-1], first.controls[-1])
        self.assertAlmostEqual(progress[0], 20.0)
        self.assertTrue(np.all(np.diff(progress) >= 0.0))
        rerolled = mpc._rollout_states(
            self.state, shifted, self.history
        )
        np.testing.assert_allclose(rerolled[0], self.state)
        self.assertFalse(np.allclose(rerolled[-1], first.states[-1]))

    def test_stationary_warmup_retains_a_primal_solution(self):
        mpc = self.make_mpc()
        stationary_state = self.state.copy()
        stationary_state[3:5] = 0.0
        diagnostics = mpc.stationary_warmup(
            FullMpcInitialCondition(
                state=stationary_state,
                fixed_history_z=np.zeros((2, 4)),
                previous_command=np.zeros(2),
                previous_progress_m=20.0,
            ),
            solve_count=1,
        )
        self.assertTrue(diagnostics[0].solution_accepted)
        self.assertIsNotNone(mpc._previous_controls)


if __name__ == "__main__":
    unittest.main()
