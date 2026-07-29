from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from gem_sysid.neural_data import (
    EvaluationScales,
    OneStepArrays,
    RolloutArrays,
    StandardScaler,
    wrap_angle,
)
from gem_sysid.neural_model import ResidualDynamicsPair
from gem_sysid.neural_training import (
    midpoint_pose_step,
    predict_physical_delta,
    recursive_rollout,
    scaler_tensors,
    tensor,
)


PRIMARY_METRICS = (
    "state_speed_rmse",
    "state_yaw_rate_rmse",
    "pose_xy_rmse",
    "pose_yaw_rmse",
    "rollout20_xy_rmse",
    "rollout20_yaw_rmse",
)


@dataclass(frozen=True)
class OneStepPredictions:
    state: np.ndarray
    pose: np.ndarray


@dataclass(frozen=True)
class RolloutPredictions:
    state: np.ndarray
    pose: np.ndarray


def _metrics(values: np.ndarray) -> dict[str, float]:
    absolute = np.abs(values)
    return {
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
        "mae": float(np.mean(absolute)),
        "p95_abs": float(np.percentile(absolute, 95)),
        "max_abs": float(np.max(absolute)),
    }


def _record_metrics(
    output: dict[str, float],
    prefix: str,
    values: np.ndarray,
) -> None:
    for metric, value in _metrics(values).items():
        output[f"{prefix}_{metric}"] = value


def predict_one_step(
    model: ResidualDynamicsPair,
    arrays: OneStepArrays,
    input_scaler: StandardScaler,
    target_scaler: StandardScaler,
    batch_size: int = 512,
) -> OneStepPredictions:
    model.eval()
    scalers = scaler_tensors(input_scaler, target_scaler)
    states: list[np.ndarray] = []
    poses: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(arrays.features), batch_size):
            stop = min(start + batch_size, len(arrays.features))
            indices = slice(start, stop)
            current_state = tensor(arrays.current_state[indices])
            delta = predict_physical_delta(
                model,
                tensor(arrays.features[indices]),
                scalers,
            )
            predicted_state = current_state + delta
            predicted_pose = midpoint_pose_step(
                tensor(arrays.current_pose[indices]),
                current_state,
                predicted_state,
                tensor(arrays.dt[indices]),
            )
            states.append(predicted_state.numpy())
            poses.append(predicted_pose.numpy())
    return OneStepPredictions(
        state=np.concatenate(states),
        pose=np.concatenate(poses),
    )


def predict_rollouts(
    model: ResidualDynamicsPair,
    arrays: RolloutArrays,
    input_scaler: StandardScaler,
    target_scaler: StandardScaler,
    batch_size: int = 512,
) -> RolloutPredictions:
    model.eval()
    scalers = scaler_tensors(input_scaler, target_scaler)
    states: list[np.ndarray] = []
    poses: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(arrays.initial_pose), batch_size):
            stop = min(start + batch_size, len(arrays.initial_pose))
            indices = slice(start, stop)
            predicted_state, predicted_pose = recursive_rollout(
                model,
                tensor(arrays.initial_history_z[indices]),
                tensor(arrays.commands[indices]),
                tensor(arrays.dt[indices]),
                tensor(arrays.initial_pose[indices]),
                scalers,
            )
            states.append(predicted_state.numpy())
            poses.append(predicted_pose.numpy())
    return RolloutPredictions(
        state=np.concatenate(states),
        pose=np.concatenate(poses),
    )


def evaluate_model(
    model: ResidualDynamicsPair,
    one_step: OneStepArrays,
    rollout: RolloutArrays,
    input_scaler: StandardScaler,
    target_scaler: StandardScaler,
    batch_size: int = 512,
) -> dict[str, float | int]:
    one_step_prediction = predict_one_step(
        model,
        one_step,
        input_scaler,
        target_scaler,
        batch_size,
    )
    rollout_prediction = predict_rollouts(
        model,
        rollout,
        input_scaler,
        target_scaler,
        batch_size,
    )
    state_error = one_step_prediction.state - one_step.next_state
    pose_error = one_step_prediction.pose - one_step.next_pose
    pose_xy_error = np.linalg.norm(pose_error[:, :2], axis=1)
    pose_yaw_error = wrap_angle(pose_error[:, 2])

    rollout_pose_error = rollout_prediction.pose - rollout.target_poses
    rollout_trajectory_xy_error = np.linalg.norm(
        rollout_pose_error[:, :, :2],
        axis=2,
    )
    rollout_trajectory_yaw_error = wrap_angle(
        rollout_pose_error[:, :, 2]
    )
    rollout_xy_error = rollout_trajectory_xy_error[:, -1]
    rollout_yaw_error = rollout_trajectory_yaw_error[:, -1]
    trajectory_xy_error = rollout_trajectory_xy_error.reshape(-1)
    trajectory_yaw_error = rollout_trajectory_yaw_error.reshape(-1)

    output: dict[str, float | int] = {
        "one_step_sample_count": len(state_error),
        "rollout20_window_count": len(rollout_xy_error),
    }
    _record_metrics(output, "state_speed", state_error[:, 0])
    _record_metrics(output, "state_yaw_rate", state_error[:, 1])
    _record_metrics(output, "pose_xy", pose_xy_error)
    _record_metrics(output, "pose_yaw", pose_yaw_error)
    _record_metrics(output, "rollout20_xy", rollout_xy_error)
    _record_metrics(output, "rollout20_yaw", rollout_yaw_error)
    _record_metrics(
        output,
        "rollout20_trajectory_xy",
        trajectory_xy_error,
    )
    _record_metrics(
        output,
        "rollout20_trajectory_yaw",
        trajectory_yaw_error,
    )
    return output


def validation_composite_score(
    metrics: dict[str, float | int],
    scales: EvaluationScales,
) -> float:
    normalized = (
        float(metrics["state_speed_rmse"]) / scales.state_speed,
        float(metrics["state_yaw_rate_rmse"]) / scales.state_yaw_rate,
        float(metrics["pose_xy_rmse"]) / scales.pose_xy,
        float(metrics["pose_yaw_rmse"]) / scales.pose_yaw,
        float(metrics["rollout20_xy_rmse"]) / scales.rollout_xy,
        float(metrics["rollout20_yaw_rmse"]) / scales.rollout_yaw,
    )
    return float(np.mean(normalized))


def fit_linear_baseline(
    train_one_step: OneStepArrays,
    input_scaler: StandardScaler,
    target_scaler: StandardScaler,
    ridge: float = 1e-6,
) -> ResidualDynamicsPair:
    if ridge < 0.0:
        raise ValueError("ridge cannot be negative")
    normalized_features = input_scaler.transform(train_one_step.features)
    normalized_targets = target_scaler.transform(
        train_one_step.target_delta
    )
    design = np.column_stack(
        (normalized_features, np.ones(len(normalized_features)))
    )
    regularization = ridge * np.eye(design.shape[1])
    regularization[-1, -1] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + regularization,
        design.T @ normalized_targets,
    )

    model = ResidualDynamicsPair(
        input_size=train_one_step.features.shape[1],
        hidden_layers=0,
    )
    for target_index, submodel in enumerate(
        (model.speed_model, model.yaw_rate_model)
    ):
        linear = submodel.network[0]
        with torch.no_grad():
            linear.weight.copy_(
                torch.as_tensor(
                    coefficients[:-1, target_index][None, :],
                    dtype=linear.weight.dtype,
                )
            )
            linear.bias.copy_(
                torch.as_tensor(
                    [coefficients[-1, target_index]],
                    dtype=linear.bias.dtype,
                )
            )
    return model
