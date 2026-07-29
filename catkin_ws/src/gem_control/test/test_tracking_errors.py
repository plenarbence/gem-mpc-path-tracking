#!/usr/bin/env python3

import math
import unittest

import casadi as ca
import numpy as np

from gem_control.tracking_errors import (
    lateral_error,
    lateral_error_symbolic,
    yaw_error,
    yaw_error_symbolic,
)


class TrackingErrorsTest(unittest.TestCase):
    def test_lateral_error_sign_for_axis_aligned_path(self):
        self.assertAlmostEqual(lateral_error(2.0, 1.5, 2.0, 0.0, 0.0), 1.5)
        self.assertAlmostEqual(lateral_error(2.0, -1.5, 2.0, 0.0, 0.0), -1.5)

    def test_lateral_error_sign_for_rotated_path(self):
        self.assertAlmostEqual(
            lateral_error(-1.0, 4.0, 0.0, 4.0, 0.5 * math.pi),
            1.0,
        )
        self.assertAlmostEqual(
            lateral_error(1.0, 4.0, 0.0, 4.0, 0.5 * math.pi),
            -1.0,
        )

    def test_yaw_error_wraps_across_pi(self):
        error = yaw_error(-math.pi + 0.05, math.pi - 0.05)
        self.assertAlmostEqual(error, 0.1)
        reverse = yaw_error(math.pi - 0.05, -math.pi + 0.05)
        self.assertAlmostEqual(reverse, -0.1)

    def test_vector_inputs(self):
        errors = lateral_error(
            np.array([0.0, 0.0]),
            np.array([1.0, -2.0]),
            0.0,
            0.0,
            0.0,
        )
        np.testing.assert_allclose(errors, np.array([1.0, -2.0]))

    def test_casadi_and_numerical_helpers_match(self):
        values = ca.MX.sym("values", 7)
        function = ca.Function(
            "tracking_error_parity",
            [values],
            [
                lateral_error_symbolic(
                    values[0],
                    values[1],
                    values[2],
                    values[3],
                    values[4],
                ),
                yaw_error_symbolic(values[5], values[6]),
            ],
        )
        samples = (
            (3.0, -1.0, 1.0, 2.0, 0.7, -3.0, 3.1),
            (-2.0, 4.0, 0.5, -1.0, -1.2, 2.9, -2.8),
        )
        for sample in samples:
            symbolic_lateral, symbolic_yaw = function(sample)
            self.assertAlmostEqual(
                float(symbolic_lateral),
                lateral_error(*sample[:5]),
            )
            self.assertAlmostEqual(
                float(symbolic_yaw),
                yaw_error(*sample[5:]),
            )

    def test_numerical_helpers_reject_nonfinite_values(self):
        with self.assertRaises(ValueError):
            lateral_error(float("nan"), 0.0, 0.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            yaw_error(0.0, float("inf"))


if __name__ == "__main__":
    unittest.main()
