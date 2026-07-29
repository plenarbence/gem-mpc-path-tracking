#!/usr/bin/env python3

import unittest

import numpy as np

from gem_control.learned_dynamics import (
    LearnedDynamics,
    midpoint_pose_step_numpy,
    selected_model_directory,
)
from gem_control.timing_compensation import (
    TimedVehicleState,
    extrapolate_odometry,
    prepare_delayed_mpc_start,
)


class TimingCompensationTest(unittest.TestCase):
    def setUp(self):
        self.model = LearnedDynamics(selected_model_directory())

    def test_odometry_is_extrapolated_to_application_anchor(self):
        measurement = TimedVehicleState(
            10.0, np.asarray((1.0, 2.0, 0.2, 3.0, 0.1))
        )
        actual = extrapolate_odometry(measurement, 10.025)
        dt = 0.025
        yaw_midpoint = 0.2 + 0.5 * dt * 0.1
        expected = np.asarray(
            (
                1.0 + dt * 3.0 * np.cos(yaw_midpoint),
                2.0 + dt * 3.0 * np.sin(yaw_midpoint),
                0.2 + dt * 0.1,
                3.0,
                0.1,
            )
        )
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_deadtime_prediction_starts_at_next_application_anchor(self):
        measurement = TimedVehicleState(
            10.0, np.asarray((1.0, 2.0, 0.2, 3.0, 0.1))
        )
        history = np.asarray(
            (
                (2.9, 0.09, 3.2, 0.04),
                (2.8, 0.08, 3.1, 0.03),
                (2.7, 0.07, 3.0, 0.02),
            )
        )
        result = prepare_delayed_mpc_start(
            model=self.model,
            odometry=measurement,
            command_publish_timestamp_s=10.02,
            commissioned_takeover_delay_s=0.005,
            controller_period_s=0.1,
            applied_history_z=history,
        )
        self.assertAlmostEqual(result.application_anchor_s, 10.025)
        self.assertAlmostEqual(result.optimization_start_s, 10.125)
        aligned_history = history.copy()
        aligned_history[0, :2] = result.aligned_state[3:5]
        next_dynamic = self.model.predict_next_state_numpy(aligned_history)
        next_pose = midpoint_pose_step_numpy(
            result.aligned_state[:3],
            result.aligned_state[3:5],
            next_dynamic,
            0.1,
        )
        np.testing.assert_allclose(
            result.predicted_state,
            np.r_[next_pose, next_dynamic],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result.fixed_history_z, aligned_history[:2]
        )
        np.testing.assert_allclose(result.active_command, history[0, 2:4])

    def test_future_measurement_is_rejected(self):
        measurement = TimedVehicleState(2.0, np.zeros(5))
        with self.assertRaises(ValueError):
            extrapolate_odometry(measurement, 1.9)

    def test_post_publication_measurement_is_rejected(self):
        measurement = TimedVehicleState(10.021, np.zeros(5))
        history = np.zeros((3, 4))
        with self.assertRaises(ValueError):
            prepare_delayed_mpc_start(
                model=self.model,
                odometry=measurement,
                command_publish_timestamp_s=10.02,
                commissioned_takeover_delay_s=0.005,
                controller_period_s=0.1,
                applied_history_z=history,
            )

    def test_available_message_may_have_small_future_header_skew(self):
        measurement = TimedVehicleState(
            10.021,
            np.zeros(5),
            availability_timestamp_s=10.019,
        )
        history = np.zeros((3, 4))
        result = prepare_delayed_mpc_start(
            model=self.model,
            odometry=measurement,
            command_publish_timestamp_s=10.02,
            commissioned_takeover_delay_s=0.005,
            controller_period_s=0.1,
            applied_history_z=history,
        )
        self.assertAlmostEqual(result.application_anchor_s, 10.025)


if __name__ == "__main__":
    unittest.main()
