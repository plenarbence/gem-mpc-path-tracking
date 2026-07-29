from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


STATE_COLUMNS = ("speed_longitudinal_mps", "yaw_rate_radps")
COMMAND_COLUMNS = ("speed_command_mps", "steering_command_rad")
POSE_COLUMNS = ("x_m", "y_m", "yaw_unwrapped_rad")
REQUIRED_COLUMNS = (
    "profile_name",
    "split",
    "sample_index",
    "anchor_time_s",
    *STATE_COLUMNS,
    *COMMAND_COLUMNS,
    *POSE_COLUMNS,
)
SCALER_EPSILON = 1e-8


@dataclass(frozen=True)
class ProfileSequence:
    name: str
    split: str
    time: np.ndarray
    state: np.ndarray
    command: np.ndarray
    pose: np.ndarray

    @property
    def z(self) -> np.ndarray:
        return np.column_stack((self.state, self.command))


@dataclass(frozen=True)
class StandardScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "StandardScaler":
        array = np.asarray(values, dtype=float)
        if array.ndim != 2 or len(array) == 0:
            raise ValueError("Scaler input must be a nonempty matrix")
        mean = np.mean(array, axis=0)
        scale = np.std(array, axis=0)
        scale = np.where(scale < SCALER_EPSILON, 1.0, scale)
        return cls(mean=mean, scale=scale)

    @classmethod
    def from_dict(cls, data: dict[str, list[float]]) -> "StandardScaler":
        mean = np.asarray(data["mean"], dtype=float)
        scale = np.asarray(data["scale"], dtype=float)
        if mean.ndim != 1 or scale.shape != mean.shape:
            raise ValueError("Invalid scaler dictionary")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError("Scaler dictionary contains non-finite values")
        if np.any(scale <= 0.0):
            raise ValueError("Scaler scale values must be positive")
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.mean) / self.scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.scale + self.mean

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
        }


@dataclass(frozen=True)
class OneStepArrays:
    features: np.ndarray
    current_state: np.ndarray
    next_state: np.ndarray
    target_delta: np.ndarray
    current_pose: np.ndarray
    next_pose: np.ndarray
    dt: np.ndarray
    profile_index: np.ndarray
    sample_index: np.ndarray


@dataclass(frozen=True)
class RolloutArrays:
    initial_history_z: np.ndarray
    commands: np.ndarray
    dt: np.ndarray
    initial_pose: np.ndarray
    target_states: np.ndarray
    target_poses: np.ndarray
    profile_index: np.ndarray
    start_index: np.ndarray


@dataclass(frozen=True)
class EvaluationScales:
    state_speed: float
    state_yaw_rate: float
    pose_xy: float
    pose_yaw: float
    rollout_xy: float
    rollout_yaw: float

    def to_dict(self) -> dict[str, float]:
        return {
            "state_speed": self.state_speed,
            "state_yaw_rate": self.state_yaw_rate,
            "pose_xy": self.pose_xy,
            "pose_yaw": self.pose_yaw,
            "rollout_xy": self.rollout_xy,
            "rollout_yaw": self.rollout_yaw,
        }


def wrap_angle(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def load_profile(path: Path, expected_split: str) -> ProfileSequence:
    with path.open(newline="", encoding="ascii") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise RuntimeError(f"{path} has no CSV header")
        missing = [
            column for column in REQUIRED_COLUMNS
            if column not in reader.fieldnames
        ]
        if missing:
            raise RuntimeError(
                f"{path} is missing columns: {', '.join(missing)}"
            )
        rows = list(reader)

    if len(rows) < 2:
        raise RuntimeError(f"{path} contains fewer than two samples")
    profile_names = {row["profile_name"] for row in rows}
    splits = {row["split"] for row in rows}
    if len(profile_names) != 1 or splits != {expected_split}:
        raise RuntimeError(f"{path} mixes profiles or dataset splits")

    indices = np.asarray(
        [int(row["sample_index"]) for row in rows],
        dtype=int,
    )
    if not np.array_equal(indices, np.arange(len(rows))):
        raise RuntimeError(f"{path} sample indices are not continuous")

    def matrix(columns: tuple[str, ...]) -> np.ndarray:
        return np.asarray(
            [[float(row[column]) for column in columns] for row in rows],
            dtype=float,
        )

    time = matrix(("anchor_time_s",))[:, 0]
    state = matrix(STATE_COLUMNS)
    command = matrix(COMMAND_COLUMNS)
    pose = matrix(POSE_COLUMNS)
    if (
        not np.isfinite(time).all()
        or not np.isfinite(state).all()
        or not np.isfinite(command).all()
        or not np.isfinite(pose).all()
    ):
        raise RuntimeError(f"{path} contains non-finite values")
    if np.any(np.diff(time) <= 0.0):
        raise RuntimeError(f"{path} timestamps are not increasing")
    return ProfileSequence(
        name=profile_names.pop(),
        split=expected_split,
        time=time,
        state=state,
        command=command,
        pose=pose,
    )


def load_split(
    processed_root: Path,
    split: str,
) -> list[ProfileSequence]:
    split_dir = processed_root / split
    paths = sorted(split_dir.glob("*.csv"))
    if not paths:
        raise RuntimeError(f"No profile CSV files found in {split_dir}")
    profiles = [load_profile(path, split) for path in paths]
    names = [profile.name for profile in profiles]
    if len(names) != len(set(names)):
        raise RuntimeError(f"Duplicate profile names in {split_dir}")
    return profiles


def history_features(
    z: np.ndarray,
    index: int,
    history_depth: int,
) -> np.ndarray:
    if history_depth not in (0, 1, 2):
        raise ValueError("history_depth must be 0, 1, or 2")
    if index < history_depth:
        raise ValueError("index does not have the requested history")
    return np.concatenate(
        [z[index - offset] for offset in range(history_depth + 1)]
    )


def build_one_step_arrays(
    profiles: list[ProfileSequence],
    history_depth: int,
) -> OneStepArrays:
    features: list[np.ndarray] = []
    current_state: list[np.ndarray] = []
    next_state: list[np.ndarray] = []
    current_pose: list[np.ndarray] = []
    next_pose: list[np.ndarray] = []
    dt: list[float] = []
    profile_indices: list[int] = []
    sample_indices: list[int] = []

    for profile_index, profile in enumerate(profiles):
        z = profile.z
        for index in range(history_depth, len(profile.time) - 1):
            features.append(history_features(z, index, history_depth))
            current_state.append(profile.state[index])
            next_state.append(profile.state[index + 1])
            current_pose.append(profile.pose[index])
            next_pose.append(profile.pose[index + 1])
            dt.append(profile.time[index + 1] - profile.time[index])
            profile_indices.append(profile_index)
            sample_indices.append(index)

    if not features:
        raise RuntimeError("No one-step samples could be constructed")
    current_state_array = np.asarray(current_state)
    next_state_array = np.asarray(next_state)
    return OneStepArrays(
        features=np.asarray(features),
        current_state=current_state_array,
        next_state=next_state_array,
        target_delta=next_state_array - current_state_array,
        current_pose=np.asarray(current_pose),
        next_pose=np.asarray(next_pose),
        dt=np.asarray(dt),
        profile_index=np.asarray(profile_indices, dtype=int),
        sample_index=np.asarray(sample_indices, dtype=int),
    )


def build_rollout_arrays(
    profiles: list[ProfileSequence],
    history_depth: int,
    horizon: int,
) -> RolloutArrays:
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    histories: list[np.ndarray] = []
    commands: list[np.ndarray] = []
    dts: list[np.ndarray] = []
    initial_poses: list[np.ndarray] = []
    target_states: list[np.ndarray] = []
    target_poses: list[np.ndarray] = []
    profile_indices: list[int] = []
    start_indices: list[int] = []

    for profile_index, profile in enumerate(profiles):
        z = profile.z
        final_start = len(profile.time) - horizon - 1
        for start in range(history_depth, final_start + 1):
            histories.append(
                np.stack(
                    [
                        z[start - offset]
                        for offset in range(history_depth + 1)
                    ]
                )
            )
            commands.append(profile.command[start : start + horizon])
            dts.append(
                np.diff(profile.time[start : start + horizon + 1])
            )
            initial_poses.append(profile.pose[start])
            target_states.append(
                profile.state[start + 1 : start + horizon + 1]
            )
            target_poses.append(
                profile.pose[start + 1 : start + horizon + 1]
            )
            profile_indices.append(profile_index)
            start_indices.append(start)

    if not histories:
        raise RuntimeError("No rollout windows could be constructed")
    return RolloutArrays(
        initial_history_z=np.asarray(histories),
        commands=np.asarray(commands),
        dt=np.asarray(dts),
        initial_pose=np.asarray(initial_poses),
        target_states=np.asarray(target_states),
        target_poses=np.asarray(target_poses),
        profile_index=np.asarray(profile_indices, dtype=int),
        start_index=np.asarray(start_indices, dtype=int),
    )


def fit_scalers(
    train_one_step: OneStepArrays,
) -> tuple[StandardScaler, StandardScaler]:
    return (
        StandardScaler.fit(train_one_step.features),
        StandardScaler.fit(train_one_step.target_delta),
    )


def fit_evaluation_scales(
    train_one_step: OneStepArrays,
    train_rollout: RolloutArrays,
) -> EvaluationScales:
    state_scale = np.std(train_one_step.next_state, axis=0)
    pose_delta = train_one_step.next_pose - train_one_step.current_pose
    pose_xy = np.sqrt(np.mean(np.sum(np.square(pose_delta[:, :2]), axis=1)))
    pose_yaw = np.sqrt(
        np.mean(np.square(wrap_angle(pose_delta[:, 2])))
    )

    rollout_delta = (
        train_rollout.target_poses[:, -1, :]
        - train_rollout.initial_pose
    )
    rollout_xy = np.sqrt(
        np.mean(np.sum(np.square(rollout_delta[:, :2]), axis=1))
    )
    rollout_yaw = np.sqrt(
        np.mean(np.square(wrap_angle(rollout_delta[:, 2])))
    )

    def positive(value: float) -> float:
        return max(float(value), SCALER_EPSILON)

    return EvaluationScales(
        state_speed=positive(state_scale[0]),
        state_yaw_rate=positive(state_scale[1]),
        pose_xy=positive(pose_xy),
        pose_yaw=positive(pose_yaw),
        rollout_xy=positive(rollout_xy),
        rollout_yaw=positive(rollout_yaw),
    )
