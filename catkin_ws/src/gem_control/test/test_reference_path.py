#!/usr/bin/env python3

import unittest

import numpy as np

from gem_control.casadi_reference_path import CasadiReferencePath
from gem_control.reference_path import build_configured_reference_path


class ReferencePathTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path, cls.diagnostics, cls.settings = (
            build_configured_reference_path()
        )

    def test_selected_configuration_and_preprocessing(self):
        self.assertEqual(self.settings.waypoint_package, "gem_pure_pursuit_sim")
        self.assertAlmostEqual(
            self.settings.preprocessing.smoothing_factor,
            0.5,
        )
        self.assertEqual(self.diagnostics.raw_point_count, 3822)
        self.assertEqual(self.diagnostics.cleaned_point_count, 3168)
        self.assertEqual(self.diagnostics.lap_point_count, 3151)
        self.assertEqual(self.diagnostics.first_lap_raw_index, 662)
        self.assertEqual(self.diagnostics.last_lap_raw_index, 3812)
        self.assertAlmostEqual(
            self.diagnostics.closure_error_after_m,
            0.0,
            places=12,
        )

    def test_path_matches_validated_v1_reference(self):
        self.assertAlmostEqual(self.path.length, 831.7180087726773, places=5)
        expected = np.array(
            [
                [0.1277221954, -2.0034035920, -0.0019451707, 0.0102518749],
                [1.1277195480, -2.0017811473, 0.0035085088, 0.0001478571],
                [10.1272817288, -2.0776546839, -0.0119178877, 0.0001971714],
                [97.9619484585, 9.9003267878, 0.5292980652, 0.0115370548],
                [16.3188848421, 197.1603214733, -3.1235404732, -0.0024684586],
            ]
        )
        actual = self.path.evaluate(np.array([0.0, 1.0, 10.0, 100.0, 400.0]))
        actual_matrix = np.column_stack(
            [actual.x, actual.y, actual.yaw, actual.curvature]
        )
        np.testing.assert_allclose(actual_matrix, expected, rtol=1e-5, atol=1e-6)

    def test_periodic_evaluation_and_seam_quality(self):
        start = self.path.evaluate(0.0)
        end = self.path.evaluate(self.path.length)
        self.assertAlmostEqual(float(start.x), float(end.x), places=10)
        self.assertAlmostEqual(float(start.y), float(end.y), places=10)
        self.assertAlmostEqual(
            float(self.path.evaluate(-1.0).x),
            float(self.path.evaluate(self.path.length - 1.0).x),
            places=9,
        )
        seam = self.path.seam_diagnostics()
        self.assertLess(seam.seam_position_gap_m, 0.01)
        self.assertLess(seam.seam_tangent_angle_gap_rad, 0.001)
        self.assertLess(seam.seam_curvature_gap_1pm, 0.001)

    def test_projection_preserves_continuous_progress_hint(self):
        expected_s = 0.35 * self.path.length
        point = self.path.evaluate(expected_s)
        projection = self.path.project(float(point.x), float(point.y))
        self.assertLess(projection.distance, 1e-5)
        self.assertAlmostEqual(projection.s_wrapped, expected_s, places=4)

        hinted_s = expected_s + self.path.length
        hinted = self.path.project(
            float(point.x),
            float(point.y),
            s_hint=hinted_s,
        )
        self.assertLess(hinted.distance, 1e-5)
        self.assertAlmostEqual(hinted.s, hinted_s, places=4)

    def test_local_projection_matches_general_projection(self):
        expected_s = 100.3
        point = self.path.evaluate(expected_s)
        x = float(point.x - 0.4 * np.sin(point.yaw))
        y = float(point.y + 0.4 * np.cos(point.yaw))
        general = self.path.project(x, y, s_hint=expected_s - 0.2)
        local = self.path.project_local(x, y, expected_s - 0.2)
        self.assertAlmostEqual(local.s, general.s, places=6)
        self.assertAlmostEqual(
            local.signed_lateral_error,
            general.signed_lateral_error,
            places=7,
        )

    def test_casadi_path_matches_python_path(self):
        casadi_path = CasadiReferencePath(
            self.path,
            sample_count=self.settings.casadi_sample_count,
        )
        progress = np.linspace(
            0.0,
            self.path.length,
            500,
            endpoint=False,
        )
        parity = casadi_path.parity_check(progress)
        self.assertLess(parity.maximum_position_difference_m, 0.0001)
        self.assertLess(parity.maximum_yaw_difference_rad, 0.001)
        self.assertLess(parity.maximum_curvature_difference_1pm, 0.001)


if __name__ == "__main__":
    unittest.main()
