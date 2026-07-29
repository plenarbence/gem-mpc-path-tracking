from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import casadi as ca
import numpy as np


PathLike = Union[str, Path]


@dataclass(frozen=True)
class PortableLayer:
    kind: str
    weight: np.ndarray | None = None
    bias: np.ndarray | None = None


class PortableResidualMlp:
    """Small tanh MLP that supports identical NumPy and CasADi inference."""

    def __init__(self, layers: list[PortableLayer]) -> None:
        if not layers or layers[-1].kind != "linear":
            raise ValueError("Portable model must end with a linear layer")
        self.layers = layers

    @classmethod
    def load(cls, path: PathLike) -> "PortableResidualMlp":
        payload = json.loads(Path(path).read_text(encoding="ascii"))
        layers: list[PortableLayer] = []
        for index, item in enumerate(payload.get("layers", [])):
            kind = str(item.get("type", ""))
            if kind == "tanh":
                layers.append(PortableLayer(kind=kind))
                continue
            if kind != "linear":
                raise ValueError(
                    "Unsupported portable layer {!r} at index {}".format(
                        kind, index
                    )
                )
            weight = np.asarray(item["weight"], dtype=float)
            bias = np.asarray(item["bias"], dtype=float)
            if (
                weight.ndim != 2
                or bias.ndim != 1
                or weight.shape[0] != bias.size
                or not np.isfinite(weight).all()
                or not np.isfinite(bias).all()
            ):
                raise ValueError(
                    "Invalid portable linear layer at index {}".format(index)
                )
            layers.append(
                PortableLayer(kind=kind, weight=weight, bias=bias)
            )
        return cls(layers)

    def predict_numpy(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=float)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("features must be a finite vector or matrix")
        for layer in self.layers:
            if layer.kind == "tanh":
                values = np.tanh(values)
            else:
                assert layer.weight is not None and layer.bias is not None
                if values.shape[1] != layer.weight.shape[1]:
                    raise ValueError("Feature width does not match model input")
                values = values @ layer.weight.T + layer.bias
        return values

    def predict_symbolic(self, features):
        values = features
        for layer in self.layers:
            if layer.kind == "tanh":
                values = ca.tanh(values)
            else:
                assert layer.weight is not None and layer.bias is not None
                values = ca.mtimes(ca.DM(layer.weight), values) + ca.DM(
                    layer.bias
                )
        return values


class LearnedDynamics:
    """Selected H2 residual model for [speed, yaw-rate] dynamics."""

    expected_feature_order = (
        "speed_k",
        "yaw_rate_k",
        "speed_command_k",
        "steering_command_k",
        "speed_k-1",
        "yaw_rate_k-1",
        "speed_command_k-1",
        "steering_command_k-1",
        "speed_k-2",
        "yaw_rate_k-2",
        "speed_command_k-2",
        "steering_command_k-2",
    )

    def __init__(self, model_directory: PathLike) -> None:
        self.model_directory = Path(model_directory)
        metadata = json.loads(
            (self.model_directory / "model_metadata.json").read_text(
                encoding="ascii"
            )
        )
        if int(metadata["history_depth"]) != 2:
            raise ValueError("MPC requires the selected H2 model")
        if tuple(metadata["feature_order"]) != self.expected_feature_order:
            raise ValueError("Selected model feature order is incompatible")
        scalers = json.loads(
            (
                self.model_directory / str(metadata["scalers_file"])
            ).read_text(encoding="ascii")
        )
        self.input_mean = np.asarray(
            scalers["input"]["mean"], dtype=float
        )
        self.input_scale = np.asarray(
            scalers["input"]["scale"], dtype=float
        )
        self.target_mean = np.asarray(
            scalers["target_delta"]["mean"], dtype=float
        )
        self.target_scale = np.asarray(
            scalers["target_delta"]["scale"], dtype=float
        )
        if (
            self.input_mean.shape != (12,)
            or self.input_scale.shape != (12,)
            or self.target_mean.shape != (2,)
            or self.target_scale.shape != (2,)
            or np.any(self.input_scale <= 0.0)
            or np.any(self.target_scale <= 0.0)
        ):
            raise ValueError("Selected model scaler dimensions are invalid")
        self.speed_model = PortableResidualMlp.load(
            self.model_directory / "speed_model_weights.json"
        )
        self.yaw_rate_model = PortableResidualMlp.load(
            self.model_directory / "yaw_rate_model_weights.json"
        )

    def _validate_history(self, history_z: np.ndarray) -> np.ndarray:
        history = np.asarray(history_z, dtype=float)
        if history.shape != (3, 4) or not np.isfinite(history).all():
            raise ValueError(
                "history_z must have shape (3, 4), newest to oldest"
            )
        return history

    def predict_delta_numpy(self, history_z: np.ndarray) -> np.ndarray:
        history = self._validate_history(history_z)
        features = history.reshape(12)
        normalized = (features - self.input_mean) / self.input_scale
        normalized_delta = np.asarray(
            (
                self.speed_model.predict_numpy(normalized)[0, 0],
                self.yaw_rate_model.predict_numpy(normalized)[0, 0],
            )
        )
        return normalized_delta * self.target_scale + self.target_mean

    def predict_next_state_numpy(self, history_z: np.ndarray) -> np.ndarray:
        history = self._validate_history(history_z)
        return history[0, :2] + self.predict_delta_numpy(history)

    def predict_delta_symbolic(self, history_z):
        features = ca.reshape(history_z.T, 12, 1)
        normalized = (
            features - ca.DM(self.input_mean)
        ) / ca.DM(self.input_scale)
        normalized_delta = ca.vertcat(
            self.speed_model.predict_symbolic(normalized)[0],
            self.yaw_rate_model.predict_symbolic(normalized)[0],
        )
        return (
            normalized_delta * ca.DM(self.target_scale)
            + ca.DM(self.target_mean)
        )

    def predict_next_state_symbolic(self, history_z):
        return history_z[0, 0:2].T + self.predict_delta_symbolic(history_z)


def midpoint_pose_step_numpy(
    pose: np.ndarray,
    current_state: np.ndarray,
    next_state: np.ndarray,
    dt: float,
) -> np.ndarray:
    pose_value = np.asarray(pose, dtype=float)
    current = np.asarray(current_state, dtype=float)
    following = np.asarray(next_state, dtype=float)
    if (
        pose_value.shape != (3,)
        or current.shape != (2,)
        or following.shape != (2,)
        or not np.isfinite(dt)
        or dt <= 0.0
    ):
        raise ValueError("Invalid pose integration input")
    midpoint = 0.5 * (current + following)
    yaw_midpoint = pose_value[2] + 0.5 * dt * midpoint[1]
    return np.asarray(
        (
            pose_value[0] + dt * midpoint[0] * np.cos(yaw_midpoint),
            pose_value[1] + dt * midpoint[0] * np.sin(yaw_midpoint),
            pose_value[2] + dt * midpoint[1],
        )
    )


def midpoint_pose_step_symbolic(pose, current_state, next_state, dt: float):
    midpoint = 0.5 * (current_state + next_state)
    yaw_midpoint = pose[2] + 0.5 * dt * midpoint[1]
    return ca.vertcat(
        pose[0] + dt * midpoint[0] * ca.cos(yaw_midpoint),
        pose[1] + dt * midpoint[0] * ca.sin(yaw_midpoint),
        pose[2] + dt * midpoint[1],
    )


def selected_model_directory() -> Path:
    try:
        import rospkg
    except ImportError:
        pass
    else:
        try:
            package_path = Path(rospkg.RosPack().get_path("gem_control"))
            candidate = package_path / "models" / "selected"
            if candidate.exists():
                return candidate
        except rospkg.ResourceNotFound:
            pass
    return Path(__file__).resolve().parents[2] / "models" / "selected"
