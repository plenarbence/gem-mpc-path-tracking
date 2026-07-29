#!/usr/bin/env python3

import math
import unittest

import numpy as np

from gem_control.cascaded_p import (
    CascadedPConfig,
    CascadedPPathController,
    OneStepCommandBuffer,
    load_cascaded_p_config,
)


class CascadedPTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_cascaded_p_config()
        cls.controller = CascadedPPathController(config=cls.config)

    def test_frozen_v1_gains_and_intentional_delay(self):
        self.assertAlmostEqual(
            self.config.lateral_to_yaw_gain_rad_per_m, 0.27
        )
        self.assertAlmostEqual(
            self.config.yaw_to_steering_gain_rad_per_rad, 0.9
        )
        self.assertAlmostEqual(
            self.config.desired_yaw_compensation_limit_rad,
            math.radians(30.0),
        )
        self.assertAlmostEqual(self.config.period_s, 0.1)

    def test_outer_loop_steers_toward_path(self):
        reference = self.controller.reference_path.evaluate(100.0)
        normal = np.asarray(
            (-math.sin(reference.yaw), math.cos(reference.yaw))
        )
        left = np.asarray((reference.x, reference.y)) + 0.3 * normal
        result = self.controller.compute_command(
            x_m=float(left[0]),
            y_m=float(left[1]),
            yaw_rad=float(reference.yaw),
            previous_progress_m=100.0,
            reference_speed_mps=2.0,
        )
        self.assertGreater(result.projection.signed_lateral_error, 0.0)
        self.assertLess(result.yaw_compensation_rad, 0.0)
        self.assertLess(result.command[1], 0.0)

    def test_yaw_compensation_and_steering_are_saturated(self):
        config = CascadedPConfig(
            lateral_to_yaw_gain_rad_per_m=10.0,
            yaw_to_steering_gain_rad_per_rad=10.0,
        )
        controller = CascadedPPathController(
            config=config,
            reference_path=self.controller.reference_path,
        )
        reference = controller.reference_path.evaluate(100.0)
        normal = np.asarray(
            (-math.sin(reference.yaw), math.cos(reference.yaw))
        )
        left = np.asarray((reference.x, reference.y)) + 1.0 * normal
        result = controller.compute_command(
            x_m=float(left[0]),
            y_m=float(left[1]),
            yaw_rad=float(reference.yaw),
            previous_progress_m=100.0,
            reference_speed_mps=20.0,
        )
        self.assertAlmostEqual(
            result.yaw_compensation_rad,
            -config.desired_yaw_compensation_limit_rad,
        )
        self.assertAlmostEqual(
            result.command[1], -config.maximum_steering_command_rad
        )
        self.assertAlmostEqual(
            result.command[0], config.maximum_speed_command_mps
        )
        self.assertTrue(result.steering_saturated)
        self.assertTrue(result.speed_saturated)

    def test_command_buffer_holds_for_one_complete_tick(self):
        buffer = OneStepCommandBuffer()
        calculated_at_k = np.asarray((2.0, 0.1))
        np.testing.assert_allclose(buffer.command_for_tick, np.zeros(2))
        buffer.stage_for_next_tick(calculated_at_k)
        np.testing.assert_allclose(
            buffer.command_for_tick, calculated_at_k
        )

    def test_command_buffer_rejects_invalid_values(self):
        buffer = OneStepCommandBuffer()
        with self.assertRaises(ValueError):
            buffer.stage_for_next_tick(np.asarray((1.0,)))
        with self.assertRaises(ValueError):
            buffer.stage_for_next_tick(np.asarray((1.0, np.nan)))


if __name__ == "__main__":
    unittest.main()
