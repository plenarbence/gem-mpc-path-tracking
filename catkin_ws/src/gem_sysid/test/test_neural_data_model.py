#!/usr/bin/env python3

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from gem_sysid.neural_data import (
    ProfileSequence,
    StandardScaler,
    build_one_step_arrays,
    build_rollout_arrays,
    history_features,
)
from gem_sysid.neural_model import (
    IdentifiedDynamicsModel,
    ResidualDynamicsPair,
    load_model_pair,
    parameter_count,
    save_model_pair,
)


def synthetic_profile(count: int = 30) -> ProfileSequence:
    time = np.arange(count, dtype=float) * 0.1
    state = np.column_stack((time, -time))
    command = np.column_stack((10.0 + time, 20.0 + time))
    pose = np.column_stack((time, 2.0 * time, 0.1 * time))
    return ProfileSequence(
        name="synthetic",
        split="train",
        time=time,
        state=state,
        command=command,
        pose=pose,
    )


class NeuralDataModelTest(unittest.TestCase):
    def test_history_is_ordered_current_to_oldest(self) -> None:
        profile = synthetic_profile()

        result = history_features(profile.z, index=4, history_depth=2)

        np.testing.assert_allclose(
            result,
            np.concatenate((profile.z[4], profile.z[3], profile.z[2])),
        )

    def test_windows_remain_inside_each_profile(self) -> None:
        profiles = [synthetic_profile(30), synthetic_profile(25)]

        one_step = build_one_step_arrays(profiles, history_depth=2)
        rollout = build_rollout_arrays(
            profiles,
            history_depth=2,
            horizon=20,
        )

        self.assertEqual(len(one_step.features), (30 - 3) + (25 - 3))
        self.assertEqual(len(rollout.initial_pose), (30 - 22) + (25 - 22))
        self.assertEqual(rollout.target_poses.shape[1:], (20, 3))
        self.assertTrue(
            np.all(rollout.start_index >= 2)
        )

    def test_scaler_round_trip_and_model_shape(self) -> None:
        values = np.asarray([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
        scaler = StandardScaler.fit(values)
        np.testing.assert_allclose(
            scaler.inverse_transform(scaler.transform(values)),
            values,
        )

        model = ResidualDynamicsPair(input_size=12)
        output = model(torch.zeros((7, 12)))
        self.assertEqual(tuple(output.shape), (7, 2))
        self.assertEqual(
            parameter_count(model),
            parameter_count(model.speed_model)
            + parameter_count(model.yaw_rate_model),
        )

    def test_model_checkpoint_round_trip(self) -> None:
        torch.manual_seed(5)
        model = ResidualDynamicsPair(input_size=4)
        features = torch.randn((6, 4))
        expected = model(features).detach().numpy()

        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            save_model_pair(
                model,
                output_dir,
                {
                    "architecture": "mlp",
                    "feature_order": ["v", "omega", "vcmd", "dcmd"],
                    "hidden_width": 32,
                    "hidden_layers": 2,
                },
            )
            loaded = load_model_pair(output_dir)

        np.testing.assert_allclose(
            loaded(features).detach().numpy(),
            expected,
        )

    def test_portable_inference_matches_pytorch_export(self) -> None:
        torch.manual_seed(7)
        model = ResidualDynamicsPair(input_size=12)
        input_scaler = StandardScaler(
            mean=np.linspace(-0.5, 0.5, 12),
            scale=np.linspace(0.8, 1.9, 12),
        )
        target_scaler = StandardScaler(
            mean=np.asarray([0.01, -0.02]),
            scale=np.asarray([0.2, 0.1]),
        )
        history = np.arange(12, dtype=float).reshape(3, 4) / 10.0
        normalized = input_scaler.transform(
            history.reshape(1, -1)
        ).astype(np.float32)
        with torch.no_grad():
            expected_delta = target_scaler.inverse_transform(
                model(torch.as_tensor(normalized)).numpy()
            )[0]

        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            (output_dir / "scalers.json").write_text(
                json.dumps(
                    {
                        "input": input_scaler.to_dict(),
                        "target_delta": target_scaler.to_dict(),
                    }
                ),
                encoding="ascii",
            )
            save_model_pair(
                model,
                output_dir,
                {
                    "architecture": "mlp",
                    "feature_order": [f"feature_{i}" for i in range(12)],
                    "history_depth": 2,
                    "hidden_width": 32,
                    "hidden_layers": 2,
                    "scalers_file": "scalers.json",
                },
            )
            portable = IdentifiedDynamicsModel.load(output_dir)

        np.testing.assert_allclose(
            portable.predict_delta(history),
            expected_delta,
            rtol=1e-5,
            atol=1e-7,
        )
        states, poses = portable.rollout(
            history,
            commands=np.asarray([[0.5, 0.1], [0.6, 0.2]]),
            dt=np.asarray([0.1, 0.1]),
            initial_pose=np.zeros(3),
        )
        self.assertEqual(states.shape, (2, 2))
        self.assertEqual(poses.shape, (2, 3))
        np.testing.assert_allclose(
            states[0],
            history[0, :2] + expected_delta,
            rtol=1e-5,
            atol=1e-7,
        )


if __name__ == "__main__":
    unittest.main()
