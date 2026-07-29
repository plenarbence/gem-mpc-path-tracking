#!/usr/bin/env python3

import math
import unittest

import numpy as np

from gem_control.simplified_mpc import (
    SimplifiedKinematicMpc,
    load_simplified_mpc_config,
    project_learned_start_state,
)


class SimplifiedMpcTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_simplified_mpc_config()
        cls.mpc = SimplifiedKinematicMpc(config=cls.config)

    def test_frozen_configuration_and_horizon(self):
        self.assertEqual(self.config.horizon_steps, 12)
        self.assertAlmostEqual(self.config.period_s, 0.1)
        self.assertAlmostEqual(
            self.config.maximum_steering_command_rad, 0.5
        )
        self.assertAlmostEqual(
            self.config.steering_gain_rad_per_rad,
            1.3151152437925338,
        )
        self.assertAlmostEqual(
            self.config.weights.yaw, 212.7565413928612
        )

    def test_horizon_duration_is_1p2_seconds(self):
        self.assertAlmostEqual(
            self.config.horizon_steps * self.config.period_s, 1.2
        )

    def test_euler_position_and_lower_p_bicycle_yaw(self):
        reference = self.mpc.reference_path.evaluate(100.0)
        state = np.asarray(
            (reference.x, reference.y, reference.yaw, 3.0, 100.0)
        )
        control = np.asarray((0.2, 0.1))
        result = self.mpc.stage_dynamics_numpy(state, control)
        raw_steering = self.config.steering_gain_rad_per_rad * 0.1
        smooth_steering = (
            self.config.maximum_steering_command_rad
            * math.tanh(
                raw_steering
                / self.config.maximum_steering_command_rad
            )
        )
        expected_yaw = (
            state[2]
            + self.config.period_s
            * state[3]
            / self.config.wheelbase_m
            * math.tan(smooth_steering)
        )
        self.assertAlmostEqual(
            result[0],
            state[0]
            + self.config.period_s * state[3] * math.cos(state[2]),
        )
        self.assertAlmostEqual(result[2], expected_yaw)
        self.assertAlmostEqual(result[3], 3.2)
        self.assertGreater(result[4], state[4])

    def test_stationary_solve_produces_finite_bounded_command(self):
        reference = self.mpc.reference_path.evaluate(0.0)
        state = np.asarray(
            (reference.x, reference.y, reference.yaw, 0.0, 0.0)
        )
        result = self.mpc.solve(
            predicted_vehicle_state=state,
            previous_progress_m=0.0,
            reference_speed_mps=0.5,
            warm_start_allowed=False,
            enforce_deadline=False,
        )
        self.assertTrue(np.isfinite(result.first_command).all())
        self.assertGreaterEqual(result.first_command[0], 0.0)
        self.assertLessEqual(result.first_command[0], 5.5)
        self.assertLessEqual(abs(result.first_command[1]), 0.5)
        self.assertLess(
            result.diagnostics.maximum_dynamics_residual, 1e-4
        )

    def test_negative_learned_speed_is_projected_to_physical_domain(self):
        state = np.asarray((1.0, 2.0, 0.1, -0.003, -0.01))
        projected = project_learned_start_state(state, self.config)
        self.assertEqual(projected[3], 0.0)
        np.testing.assert_array_equal(
            projected[[0, 1, 2, 4]], state[[0, 1, 2, 4]]
        )
        self.assertEqual(state[3], -0.003)

    def test_shifted_warm_start_is_used_after_accepted_solve(self):
        reference = self.mpc.reference_path.evaluate(0.0)
        state = np.asarray(
            (reference.x, reference.y, reference.yaw, 1.0, 0.0)
        )
        first = self.mpc.solve(
            predicted_vehicle_state=state,
            previous_progress_m=0.0,
            reference_speed_mps=1.0,
            enforce_deadline=False,
        )
        second = self.mpc.solve(
            predicted_vehicle_state=state,
            previous_progress_m=0.0,
            reference_speed_mps=1.0,
            enforce_deadline=False,
        )
        self.assertTrue(first.diagnostics.solution_accepted)
        self.assertEqual(
            second.diagnostics.warm_start_source, "shifted_warm_start"
        )


if __name__ == "__main__":
    unittest.main()
