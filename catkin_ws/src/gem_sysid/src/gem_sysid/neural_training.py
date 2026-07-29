from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from gem_sysid.neural_data import (
    EvaluationScales,
    OneStepArrays,
    RolloutArrays,
    StandardScaler,
)
from gem_sysid.neural_model import ResidualDynamicsPair


OBJECTIVES = ("state_one_step", "pose_one_step", "pose_rollout20")
TORCH_DTYPE = torch.float32


@dataclass(frozen=True)
class TrainingSettings:
    learning_rate: float = 1e-3
    batch_size: int = 128
    max_epochs: int = 150
    early_stopping_patience: int = 20
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    minimum_relative_improvement: float = 1e-5

    def validate(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")

    def to_dict(self) -> dict[str, float | int]:
        return {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "weight_decay": self.weight_decay,
            "gradient_clip_norm": self.gradient_clip_norm,
            "minimum_relative_improvement": (
                self.minimum_relative_improvement
            ),
        }


@dataclass(frozen=True)
class TrainingResult:
    model: ResidualDynamicsPair
    history: list[dict[str, float | int]]
    best_epoch: int
    best_validation_loss: float
    final_epoch: int
    termination_reason: str


@dataclass(frozen=True)
class ScalerTensors:
    input_mean: torch.Tensor
    input_scale: torch.Tensor
    target_mean: torch.Tensor
    target_scale: torch.Tensor


def tensor(values: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(values, dtype=TORCH_DTYPE)


def scaler_tensors(
    input_scaler: StandardScaler,
    target_scaler: StandardScaler,
) -> ScalerTensors:
    return ScalerTensors(
        input_mean=tensor(input_scaler.mean),
        input_scale=tensor(input_scaler.scale),
        target_mean=tensor(target_scaler.mean),
        target_scale=tensor(target_scaler.scale),
    )


def predict_physical_delta(
    model: ResidualDynamicsPair,
    features: torch.Tensor,
    scalers: ScalerTensors,
) -> torch.Tensor:
    normalized_features = (
        features - scalers.input_mean
    ) / scalers.input_scale
    normalized_delta = model(normalized_features)
    return normalized_delta * scalers.target_scale + scalers.target_mean


def wrap_angle_torch(values: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(values), torch.cos(values))


def midpoint_pose_step(
    pose: torch.Tensor,
    current_state: torch.Tensor,
    next_state: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    speed_midpoint = 0.5 * (
        current_state[:, 0] + next_state[:, 0]
    )
    yaw_rate_midpoint = 0.5 * (
        current_state[:, 1] + next_state[:, 1]
    )
    yaw_midpoint = pose[:, 2] + 0.5 * dt * yaw_rate_midpoint
    return torch.stack(
        (
            pose[:, 0] + dt * speed_midpoint * torch.cos(yaw_midpoint),
            pose[:, 1] + dt * speed_midpoint * torch.sin(yaw_midpoint),
            pose[:, 2] + dt * yaw_rate_midpoint,
        ),
        dim=1,
    )


def recursive_rollout(
    model: ResidualDynamicsPair,
    initial_history_z: torch.Tensor,
    commands: torch.Tensor,
    dt: torch.Tensor,
    initial_pose: torch.Tensor,
    scalers: ScalerTensors,
) -> tuple[torch.Tensor, torch.Tensor]:
    if initial_history_z.ndim != 3 or initial_history_z.shape[2] != 4:
        raise ValueError("initial_history_z must have shape [batch, depth, 4]")
    if commands.ndim != 3 or commands.shape[2] != 2:
        raise ValueError("commands must have shape [batch, horizon, 2]")
    if dt.shape != commands.shape[:2]:
        raise ValueError("dt shape must match command batch and horizon")

    history_z = initial_history_z
    pose = initial_pose
    predicted_states: list[torch.Tensor] = []
    predicted_poses: list[torch.Tensor] = []
    horizon = commands.shape[1]

    for step in range(horizon):
        features = history_z.reshape(history_z.shape[0], -1)
        current_state = history_z[:, 0, :2]
        delta = predict_physical_delta(model, features, scalers)
        next_state = current_state + delta
        pose = midpoint_pose_step(
            pose,
            current_state,
            next_state,
            dt[:, step],
        )
        predicted_states.append(next_state)
        predicted_poses.append(pose)

        if step + 1 < horizon:
            next_z = torch.cat(
                (next_state, commands[:, step + 1, :]),
                dim=1,
            )
            history_z = torch.cat(
                (next_z[:, None, :], history_z[:, :-1, :]),
                dim=1,
            )

    return (
        torch.stack(predicted_states, dim=1),
        torch.stack(predicted_poses, dim=1),
    )


def _state_loss(
    predicted_next_state: torch.Tensor,
    target_next_state: torch.Tensor,
    target_scale: torch.Tensor,
) -> torch.Tensor:
    normalized_error = (
        predicted_next_state - target_next_state
    ) / target_scale
    return torch.mean(torch.square(normalized_error))


def _pose_loss(
    predicted_pose: torch.Tensor,
    target_pose: torch.Tensor,
    scales: EvaluationScales,
) -> torch.Tensor:
    xy_error = predicted_pose[..., :2] - target_pose[..., :2]
    yaw_error = wrap_angle_torch(
        predicted_pose[..., 2] - target_pose[..., 2]
    )
    xy_loss = torch.mean(
        torch.sum(torch.square(xy_error), dim=-1)
        / (scales.pose_xy * scales.pose_xy)
    )
    yaw_loss = torch.mean(
        torch.square(yaw_error) / (scales.pose_yaw * scales.pose_yaw)
    )
    return 0.5 * (xy_loss + yaw_loss)


def _one_step_batch_loss(
    model: ResidualDynamicsPair,
    arrays: OneStepArrays,
    indices: np.ndarray,
    objective: str,
    scalers: ScalerTensors,
    evaluation_scales: EvaluationScales,
) -> torch.Tensor:
    features = tensor(arrays.features[indices])
    current_state = tensor(arrays.current_state[indices])
    target_next_state = tensor(arrays.next_state[indices])
    delta = predict_physical_delta(model, features, scalers)
    predicted_next_state = current_state + delta

    if objective == "state_one_step":
        return _state_loss(
            predicted_next_state,
            target_next_state,
            scalers.target_scale,
        )
    if objective != "pose_one_step":
        raise ValueError(f"Invalid one-step objective: {objective}")
    predicted_pose = midpoint_pose_step(
        tensor(arrays.current_pose[indices]),
        current_state,
        predicted_next_state,
        tensor(arrays.dt[indices]),
    )
    return _pose_loss(
        predicted_pose,
        tensor(arrays.next_pose[indices]),
        evaluation_scales,
    )


def _rollout_batch_loss(
    model: ResidualDynamicsPair,
    arrays: RolloutArrays,
    indices: np.ndarray,
    scalers: ScalerTensors,
    evaluation_scales: EvaluationScales,
) -> torch.Tensor:
    _, predicted_pose = recursive_rollout(
        model,
        tensor(arrays.initial_history_z[indices]),
        tensor(arrays.commands[indices]),
        tensor(arrays.dt[indices]),
        tensor(arrays.initial_pose[indices]),
        scalers,
    )
    return _pose_loss(
        predicted_pose,
        tensor(arrays.target_poses[indices]),
        evaluation_scales,
    )


def _batch_indices(
    sample_count: int,
    batch_size: int,
    generator: torch.Generator | None,
) -> list[np.ndarray]:
    if generator is None:
        order = np.arange(sample_count)
    else:
        order = torch.randperm(
            sample_count,
            generator=generator,
        ).numpy()
    return [
        order[start : start + batch_size]
        for start in range(0, sample_count, batch_size)
    ]


def _dataset_size(
    objective: str,
    one_step: OneStepArrays,
    rollout: RolloutArrays,
) -> int:
    if objective in ("state_one_step", "pose_one_step"):
        return len(one_step.features)
    if objective == "pose_rollout20":
        return len(rollout.initial_pose)
    raise ValueError(f"Unknown objective: {objective}")


def _objective_loss(
    model: ResidualDynamicsPair,
    objective: str,
    one_step: OneStepArrays,
    rollout: RolloutArrays,
    indices: np.ndarray,
    scalers: ScalerTensors,
    evaluation_scales: EvaluationScales,
) -> torch.Tensor:
    if objective in ("state_one_step", "pose_one_step"):
        return _one_step_batch_loss(
            model,
            one_step,
            indices,
            objective,
            scalers,
            evaluation_scales,
        )
    if objective == "pose_rollout20":
        return _rollout_batch_loss(
            model,
            rollout,
            indices,
            scalers,
            evaluation_scales,
        )
    raise ValueError(f"Unknown objective: {objective}")


def _evaluate_loss(
    model: ResidualDynamicsPair,
    objective: str,
    one_step: OneStepArrays,
    rollout: RolloutArrays,
    scalers: ScalerTensors,
    evaluation_scales: EvaluationScales,
    batch_size: int,
) -> float:
    sample_count = _dataset_size(objective, one_step, rollout)
    weighted_loss = 0.0
    model.eval()
    with torch.no_grad():
        for indices in _batch_indices(sample_count, batch_size, None):
            loss = _objective_loss(
                model,
                objective,
                one_step,
                rollout,
                indices,
                scalers,
                evaluation_scales,
            )
            weighted_loss += float(loss) * len(indices)
    return weighted_loss / sample_count


def train_model_pair(
    model: ResidualDynamicsPair,
    objective: str,
    train_one_step: OneStepArrays,
    validation_one_step: OneStepArrays,
    train_rollout: RolloutArrays,
    validation_rollout: RolloutArrays,
    input_scaler: StandardScaler,
    target_scaler: StandardScaler,
    evaluation_scales: EvaluationScales,
    settings: TrainingSettings,
    seed: int,
) -> TrainingResult:
    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown objective: {objective}")
    settings.validate()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    scalers = scaler_tensors(input_scaler, target_scaler)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    generator = torch.Generator().manual_seed(seed)
    train_count = _dataset_size(
        objective,
        train_one_step,
        train_rollout,
    )

    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    termination_reason = "max_epochs"

    for epoch in range(1, settings.max_epochs + 1):
        model.train()
        weighted_train_loss = 0.0
        for indices in _batch_indices(
            train_count,
            settings.batch_size,
            generator,
        ):
            optimizer.zero_grad(set_to_none=True)
            loss = _objective_loss(
                model,
                objective,
                train_one_step,
                train_rollout,
                indices,
                scalers,
                evaluation_scales,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss at epoch {epoch}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                settings.gradient_clip_norm,
            )
            optimizer.step()
            weighted_train_loss += float(loss.detach()) * len(indices)

        train_loss = weighted_train_loss / train_count
        validation_loss = _evaluate_loss(
            model,
            objective,
            validation_one_step,
            validation_rollout,
            scalers,
            evaluation_scales,
            settings.batch_size,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )

        threshold = (
            0.0
            if best_state is None
            else max(
                1e-10,
                abs(best_loss) * settings.minimum_relative_improvement,
            )
        )
        if best_state is None or validation_loss < best_loss - threshold:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if (
                epochs_without_improvement
                >= settings.early_stopping_patience
            ):
                termination_reason = "early_stopping"
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a finite checkpoint")
    model.load_state_dict(best_state)
    return TrainingResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        final_epoch=len(history),
        termination_reason=termination_reason,
    )
