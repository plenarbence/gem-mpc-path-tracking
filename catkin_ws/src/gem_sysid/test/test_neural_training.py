#!/usr/bin/env python3

import unittest

import numpy as np
import torch

from gem_sysid.neural_data import StandardScaler
from gem_sysid.neural_model import ResidualDynamicsPair
from gem_sysid.neural_training import (
    midpoint_pose_step,
    recursive_rollout,
    scaler_tensors,
)


class NeuralTrainingTest(unittest.TestCase):
    def test_midpoint_pose_step(self) -> None:
        pose = torch.tensor([[0.0, 0.0, 0.0]])
        current = torch.tensor([[2.0, 0.4]])
        nxt = torch.tensor([[2.0, 0.4]])

        result = midpoint_pose_step(
            pose,
            current,
            nxt,
            torch.tensor([0.1]),
        )

        self.assertAlmostEqual(float(result[0, 0]), 0.2 * np.cos(0.02))
        self.assertAlmostEqual(float(result[0, 1]), 0.2 * np.sin(0.02))
        self.assertAlmostEqual(float(result[0, 2]), 0.04)

    def test_recursive_rollout_uses_predictions_after_initial_state(self) -> None:
        model = ResidualDynamicsPair(input_size=4)
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
        scaler = StandardScaler(
            mean=np.zeros(4),
            scale=np.ones(4),
        )
        target_scaler = StandardScaler(
            mean=np.zeros(2),
            scale=np.ones(2),
        )
        history = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])
        commands = torch.tensor(
            [[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]
        )

        states, poses = recursive_rollout(
            model,
            history,
            commands,
            torch.full((1, 3), 0.1),
            torch.zeros((1, 3)),
            scaler_tensors(scaler, target_scaler),
        )

        np.testing.assert_allclose(
            states.detach().numpy()[0],
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            atol=1e-7,
        )
        np.testing.assert_allclose(
            poses.detach().numpy()[0, :, 0],
            [0.1, 0.2, 0.3],
            atol=1e-7,
        )


if __name__ == "__main__":
    unittest.main()
