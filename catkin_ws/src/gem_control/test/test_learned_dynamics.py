#!/usr/bin/env python3

import unittest

import casadi as ca
import numpy as np

from gem_control.learned_dynamics import (
    LearnedDynamics,
    midpoint_pose_step_numpy,
    midpoint_pose_step_symbolic,
    selected_model_directory,
)


class LearnedDynamicsTest(unittest.TestCase):
    def setUp(self):
        self.model = LearnedDynamics(selected_model_directory())
        self.history = np.asarray(
            (
                (2.0, 0.1, 2.2, 0.05),
                (1.9, 0.09, 2.1, 0.04),
                (1.8, 0.08, 2.0, 0.03),
            )
        )

    def test_frozen_model_regression(self):
        predicted = self.model.predict_next_state_numpy(self.history)
        np.testing.assert_allclose(
            predicted,
            np.asarray((2.01904446, 0.09684179)),
            atol=1e-8,
            rtol=0.0,
        )

    def test_symbolic_and_numpy_inference_match(self):
        symbolic_history = ca.MX.sym("history", 3, 4)
        function = ca.Function(
            "model_parity_test",
            [symbolic_history],
            [self.model.predict_next_state_symbolic(symbolic_history)],
        )
        symbolic = np.asarray(
            function(self.history), dtype=float
        ).reshape(2)
        numpy_value = self.model.predict_next_state_numpy(self.history)
        np.testing.assert_allclose(symbolic, numpy_value, atol=1e-12)

    def test_symbolic_and_numpy_pose_steps_match(self):
        pose = np.asarray((1.0, 2.0, 0.3))
        current = np.asarray((2.0, 0.1))
        following = np.asarray((2.2, 0.12))
        numpy_value = midpoint_pose_step_numpy(
            pose, current, following, 0.1
        )
        pose_symbol = ca.MX.sym("pose", 3)
        current_symbol = ca.MX.sym("current", 2)
        next_symbol = ca.MX.sym("next", 2)
        function = ca.Function(
            "pose_parity_test",
            [pose_symbol, current_symbol, next_symbol],
            [
                midpoint_pose_step_symbolic(
                    pose_symbol, current_symbol, next_symbol, 0.1
                )
            ],
        )
        symbolic = np.asarray(
            function(pose, current, following), dtype=float
        ).reshape(3)
        np.testing.assert_allclose(symbolic, numpy_value, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
