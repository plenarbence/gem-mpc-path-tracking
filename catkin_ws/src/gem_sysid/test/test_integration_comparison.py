#!/usr/bin/env python3

import unittest

import numpy as np

from gem_sysid.integration_comparison import (
    euler_step,
    midpoint_euler_step,
    wrap_angle,
)


class IntegrationComparisonTest(unittest.TestCase):
    def test_straight_constant_motion_matches_both_methods(self) -> None:
        pose = np.asarray([1.0, 2.0, 0.0])
        expected = np.asarray([1.2, 2.0, 0.0])

        np.testing.assert_allclose(
            euler_step(pose, 2.0, 0.0, 0.1),
            expected,
        )
        np.testing.assert_allclose(
            midpoint_euler_step(pose, 2.0, 2.0, 0.0, 0.0, 0.1),
            expected,
        )

    def test_midpoint_reduces_constant_turn_position_error(self) -> None:
        pose = np.asarray([0.0, 0.0, 0.0])
        speed = 3.0
        yaw_rate = 0.8
        dt = 0.1
        radius = speed / yaw_rate
        actual = np.asarray(
            [
                radius * np.sin(yaw_rate * dt),
                radius * (1.0 - np.cos(yaw_rate * dt)),
                yaw_rate * dt,
            ]
        )

        euler = euler_step(pose, speed, yaw_rate, dt)
        midpoint = midpoint_euler_step(
            pose,
            speed,
            speed,
            yaw_rate,
            yaw_rate,
            dt,
        )

        self.assertLess(
            np.linalg.norm(midpoint[:2] - actual[:2]),
            np.linalg.norm(euler[:2] - actual[:2]),
        )
        self.assertAlmostEqual(midpoint[2], actual[2])

    def test_yaw_error_wraps_at_pi(self) -> None:
        error = wrap_angle(np.deg2rad(-179.0) - np.deg2rad(179.0))
        self.assertAlmostEqual(error, np.deg2rad(2.0))


if __name__ == "__main__":
    unittest.main()
