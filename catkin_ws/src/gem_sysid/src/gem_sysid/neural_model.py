from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from gem_sysid.neural_data import StandardScaler


HIDDEN_WIDTH = 32
HIDDEN_LAYERS = 2


class ResidualMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_width: int = HIDDEN_WIDTH,
        hidden_layers: int = HIDDEN_LAYERS,
    ) -> None:
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        if hidden_width <= 0:
            raise ValueError("hidden_width must be positive")
        if hidden_layers < 0:
            raise ValueError("hidden_layers cannot be negative")

        layers: list[nn.Module] = []
        previous_size = input_size
        for _ in range(hidden_layers):
            layers.append(nn.Linear(previous_size, hidden_width))
            layers.append(nn.Tanh())
            previous_size = hidden_width
        layers.append(nn.Linear(previous_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class ResidualDynamicsPair(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_width: int = HIDDEN_WIDTH,
        hidden_layers: int = HIDDEN_LAYERS,
    ) -> None:
        super().__init__()
        self.speed_model = ResidualMLP(
            input_size,
            hidden_width,
            hidden_layers,
        )
        self.yaw_rate_model = ResidualMLP(
            input_size,
            hidden_width,
            hidden_layers,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                self.speed_model(features),
                self.yaw_rate_model(features),
            ),
            dim=1,
        )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def save_model_pair(
    model: ResidualDynamicsPair,
    output_dir: Path,
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.speed_model.state_dict(), output_dir / "speed_model.pt")
    torch.save(
        model.yaw_rate_model.state_dict(),
        output_dir / "yaw_rate_model.pt",
    )
    (output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="ascii",
    )
    _write_portable_weights(
        model.speed_model,
        output_dir / "speed_model_weights.json",
    )
    _write_portable_weights(
        model.yaw_rate_model,
        output_dir / "yaw_rate_model_weights.json",
    )


def load_model_pair(model_dir: Path) -> ResidualDynamicsPair:
    metadata = json.loads(
        (model_dir / "model_metadata.json").read_text(encoding="ascii")
    )
    model = ResidualDynamicsPair(
        input_size=len(metadata["feature_order"]),
        hidden_width=(
            int(metadata["hidden_width"])
            if metadata["architecture"] == "mlp"
            else HIDDEN_WIDTH
        ),
        hidden_layers=int(metadata["hidden_layers"]),
    )
    model.speed_model.load_state_dict(
        torch.load(
            model_dir / "speed_model.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    model.yaw_rate_model.load_state_dict(
        torch.load(
            model_dir / "yaw_rate_model.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    model.eval()
    return model


def _write_portable_weights(model: ResidualMLP, path: Path) -> None:
    layers: list[dict[str, Any]] = []
    for layer in model.network:
        if isinstance(layer, nn.Linear):
            layers.append(
                {
                    "type": "linear",
                    "weight": (
                        layer.weight.detach().cpu().numpy().tolist()
                    ),
                    "bias": layer.bias.detach().cpu().numpy().tolist(),
                }
            )
        elif isinstance(layer, nn.Tanh):
            layers.append({"type": "tanh"})
        else:
            raise TypeError(f"Unsupported layer for export: {type(layer)}")
    path.write_text(
        json.dumps({"layers": layers}, indent=2) + "\n",
        encoding="ascii",
    )


@dataclass(frozen=True)
class PortableLayer:
    kind: str
    weight: np.ndarray | None = None
    bias: np.ndarray | None = None


class PortableResidualMLP:
    def __init__(self, layers: list[PortableLayer]) -> None:
        if not layers or layers[-1].kind != "linear":
            raise ValueError("Portable model must end with a linear layer")
        self.layers = layers

    @classmethod
    def load(cls, path: Path) -> "PortableResidualMLP":
        data = json.loads(path.read_text(encoding="ascii"))
        layers: list[PortableLayer] = []
        for index, layer in enumerate(data.get("layers", [])):
            kind = str(layer.get("type", ""))
            if kind == "tanh":
                layers.append(PortableLayer(kind=kind))
                continue
            if kind != "linear":
                raise ValueError(
                    f"Unsupported portable layer {kind!r} at index {index}"
                )
            weight = np.asarray(layer["weight"], dtype=np.float32)
            bias = np.asarray(layer["bias"], dtype=np.float32)
            if (
                weight.ndim != 2
                or bias.ndim != 1
                or weight.shape[0] != len(bias)
                or not np.isfinite(weight).all()
                or not np.isfinite(bias).all()
            ):
                raise ValueError(
                    f"Invalid portable linear layer at index {index}"
                )
            layers.append(
                PortableLayer(
                    kind=kind,
                    weight=weight,
                    bias=bias,
                )
            )
        return cls(layers)

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("features must be a finite vector or matrix")

        for layer in self.layers:
            if layer.kind == "tanh":
                values = np.tanh(values)
            else:
                if layer.weight is None or layer.bias is None:
                    raise RuntimeError("Portable linear layer has no weights")
                if values.shape[1] != layer.weight.shape[1]:
                    raise ValueError(
                        "Feature width does not match portable model input"
                    )
                values = values @ layer.weight.T + layer.bias
        return values


class IdentifiedDynamicsModel:
    """Portable residual dynamics inference with an explicit history contract."""

    def __init__(
        self,
        *,
        speed_model: PortableResidualMLP,
        yaw_rate_model: PortableResidualMLP,
        input_scaler: StandardScaler,
        target_scaler: StandardScaler,
        history_depth: int,
        feature_order: tuple[str, ...],
    ) -> None:
        if history_depth not in (0, 1, 2):
            raise ValueError("history_depth must be 0, 1, or 2")
        expected_width = 4 * (history_depth + 1)
        if len(feature_order) != expected_width:
            raise ValueError("feature_order does not match history depth")
        if input_scaler.mean.shape != (expected_width,):
            raise ValueError("Input scaler does not match feature width")
        if target_scaler.mean.shape != (2,):
            raise ValueError("Target scaler must describe two residuals")

        self.speed_model = speed_model
        self.yaw_rate_model = yaw_rate_model
        self.input_scaler = input_scaler
        self.target_scaler = target_scaler
        self.history_depth = history_depth
        self.feature_order = feature_order

    @classmethod
    def load(cls, model_dir: Path) -> "IdentifiedDynamicsModel":
        metadata = json.loads(
            (model_dir / "model_metadata.json").read_text(encoding="ascii")
        )
        scalers = json.loads(
            (model_dir / metadata["scalers_file"]).read_text(
                encoding="ascii"
            )
        )
        return cls(
            speed_model=PortableResidualMLP.load(
                model_dir / "speed_model_weights.json"
            ),
            yaw_rate_model=PortableResidualMLP.load(
                model_dir / "yaw_rate_model_weights.json"
            ),
            input_scaler=StandardScaler.from_dict(scalers["input"]),
            target_scaler=StandardScaler.from_dict(
                scalers["target_delta"]
            ),
            history_depth=int(metadata["history_depth"]),
            feature_order=tuple(metadata["feature_order"]),
        )

    def _history_matrix(self, history_z: np.ndarray) -> np.ndarray:
        history = np.asarray(history_z, dtype=np.float32)
        expected_shape = (self.history_depth + 1, 4)
        if history.shape != expected_shape or not np.isfinite(history).all():
            raise ValueError(
                "history_z must be finite with shape "
                f"{expected_shape}, ordered current to oldest"
            )
        return history

    def predict_delta(self, history_z: np.ndarray) -> np.ndarray:
        history = self._history_matrix(history_z)
        normalized_features = self.input_scaler.transform(
            history.reshape(1, -1)
        ).astype(np.float32)
        normalized_delta = np.column_stack(
            (
                self.speed_model.predict(normalized_features)[:, 0],
                self.yaw_rate_model.predict(normalized_features)[:, 0],
            )
        )
        return self.target_scaler.inverse_transform(normalized_delta)[0]

    def predict_next_state(self, history_z: np.ndarray) -> np.ndarray:
        history = self._history_matrix(history_z)
        return history[0, :2].astype(float) + self.predict_delta(history)

    @staticmethod
    def midpoint_pose_step(
        pose: np.ndarray,
        current_state: np.ndarray,
        next_state: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        pose_array = np.asarray(pose, dtype=float)
        current = np.asarray(current_state, dtype=float)
        next_value = np.asarray(next_state, dtype=float)
        if (
            pose_array.shape != (3,)
            or current.shape != (2,)
            or next_value.shape != (2,)
        ):
            raise ValueError("pose/state dimensions are invalid")

        midpoint_state = 0.5 * (current + next_value)
        midpoint_yaw = (
            pose_array[2] + 0.5 * dt * midpoint_state[1]
        )
        return np.asarray(
            (
                pose_array[0]
                + dt * midpoint_state[0] * np.cos(midpoint_yaw),
                pose_array[1]
                + dt * midpoint_state[0] * np.sin(midpoint_yaw),
                pose_array[2] + dt * midpoint_state[1],
            )
        )

    def rollout(
        self,
        initial_history_z: np.ndarray,
        commands: np.ndarray,
        dt: np.ndarray,
        initial_pose: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        history = self._history_matrix(initial_history_z).copy()
        command_array = np.asarray(commands, dtype=float)
        dt_array = np.asarray(dt, dtype=float)
        pose = np.asarray(initial_pose, dtype=float).copy()
        if (
            command_array.ndim != 2
            or command_array.shape[1] != 2
            or dt_array.shape != (len(command_array),)
            or pose.shape != (3,)
        ):
            raise ValueError("Rollout input dimensions are invalid")

        states: list[np.ndarray] = []
        poses: list[np.ndarray] = []
        for step in range(len(command_array)):
            current_state = history[0, :2].astype(float)
            next_state = self.predict_next_state(history)
            pose = self.midpoint_pose_step(
                pose,
                current_state,
                next_state,
                float(dt_array[step]),
            )
            states.append(next_state)
            poses.append(pose)

            if step + 1 < len(command_array):
                next_z = np.concatenate(
                    (next_state, command_array[step + 1])
                )
                history = np.concatenate(
                    (next_z[None, :], history[:-1]),
                    axis=0,
                )
        return np.asarray(states), np.asarray(poses)
