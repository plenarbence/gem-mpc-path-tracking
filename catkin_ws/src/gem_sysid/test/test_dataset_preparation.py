#!/usr/bin/env python3

import unittest

import numpy as np

from gem_sysid.dataset_preparation import interpolate_series, wrap_angle


class DatasetPreparationTest(unittest.TestCase):
    def test_interpolation_is_linear(self) -> None:
        times = np.asarray([0.0, 1.0, 2.0])
        values = np.asarray([[0.0], [2.0], [4.0]])

        result, quality = interpolate_series(
            times,
            values,
            np.asarray([0.5, 1.5]),
            "test",
        )

        np.testing.assert_allclose(result[:, 0], [1.0, 3.0])
        self.assertAlmostEqual(quality["max_bracket_span_ms"], 1000.0)

    def test_yaw_can_be_interpolated_across_pi(self) -> None:
        wrapped = np.deg2rad(np.asarray([179.0, -179.0]))
        unwrapped = np.unwrap(wrapped)

        result, _ = interpolate_series(
            np.asarray([0.0, 1.0]),
            unwrapped[:, np.newaxis],
            np.asarray([0.5]),
            "yaw",
        )

        self.assertAlmostEqual(abs(wrap_angle(result[:, 0])[0]), np.pi)

    def test_interpolation_rejects_extrapolation(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "does not bracket"):
            interpolate_series(
                np.asarray([0.0, 1.0]),
                np.asarray([[0.0], [1.0]]),
                np.asarray([1.1]),
                "test",
            )


if __name__ == "__main__":
    unittest.main()
