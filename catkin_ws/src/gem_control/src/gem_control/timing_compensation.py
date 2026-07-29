from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gem_control.learned_dynamics import (
    LearnedDynamics,
    midpoint_pose_step_numpy,
)


@dataclass(frozen=True)
class TimedVehicleState:
    timestamp_s: float
    state: np.ndarray
    availability_timestamp_s: float | None = None

    def validated_state(self) -> np.ndarray:
        value = np.asarray(self.state, dtype=float)
        if value.shape != (5,) or not np.isfinite(value).all():
            raise ValueError("state must be finite [x, y, yaw, speed, yaw_rate]")
        if not np.isfinite(self.timestamp_s):
            raise ValueError("timestamp must be finite")
        if (
            self.availability_timestamp_s is not None
            and not np.isfinite(self.availability_timestamp_s)
        ):
            raise ValueError("availability timestamp must be finite")
        return value


@dataclass(frozen=True)
class DelayedMpcStart:
    application_anchor_s: float
    optimization_start_s: float
    aligned_state: np.ndarray
    predicted_state: np.ndarray
    fixed_history_z: np.ndarray
    active_command: np.ndarray


def extrapolate_odometry(
    measurement: TimedVehicleState,
    target_timestamp_s: float,
    maximum_age_s: float = 0.12,
) -> np.ndarray:
    """Move odometry to an anchor while holding measured speed/yaw-rate."""

    state = measurement.validated_state()
    age = float(target_timestamp_s) - float(measurement.timestamp_s)
    if not np.isfinite(age) or age < 0.0:
        raise ValueError("Odometry measurement must not be after the anchor")
    if age > maximum_age_s:
        raise ValueError("Odometry measurement is too old for compensation")
    yaw_midpoint = state[2] + 0.5 * age * state[4]
    return np.asarray(
        (
            state[0] + age * state[3] * np.cos(yaw_midpoint),
            state[1] + age * state[3] * np.sin(yaw_midpoint),
            state[2] + age * state[4],
            state[3],
            state[4],
        )
    )


def prepare_delayed_mpc_start(
    *,
    model: LearnedDynamics,
    odometry: TimedVehicleState,
    command_publish_timestamp_s: float,
    commissioned_takeover_delay_s: float,
    controller_period_s: float,
    applied_history_z: np.ndarray,
    maximum_odometry_age_s: float = 0.12,
) -> DelayedMpcStart:
    """Prepare the state where the next optimized command will take effect.

    ``applied_history_z`` is newest-to-oldest. Its first row contains the
    currently active command; its measured state is replaced by odometry
    aligned to the commissioned application anchor.
    """

    history = np.asarray(applied_history_z, dtype=float)
    if history.shape != (3, 4) or not np.isfinite(history).all():
        raise ValueError("applied_history_z must have shape (3, 4)")
    if controller_period_s <= 0.0:
        raise ValueError("controller_period_s must be positive")
    if commissioned_takeover_delay_s < 0.0:
        raise ValueError("commissioned_takeover_delay_s must be non-negative")
    availability_timestamp = (
        odometry.timestamp_s
        if odometry.availability_timestamp_s is None
        else odometry.availability_timestamp_s
    )
    if availability_timestamp > command_publish_timestamp_s:
        raise ValueError(
            "Odometry must have been available before command publication"
        )

    anchor = (
        float(command_publish_timestamp_s)
        + float(commissioned_takeover_delay_s)
    )
    aligned = extrapolate_odometry(
        odometry,
        anchor,
        maximum_age_s=maximum_odometry_age_s,
    )
    active_command = history[0, 2:4].copy()
    aligned_history = history.copy()
    aligned_history[0, 0:2] = aligned[3:5]
    next_dynamic_state = model.predict_next_state_numpy(aligned_history)
    next_pose = midpoint_pose_step_numpy(
        aligned[0:3],
        aligned[3:5],
        next_dynamic_state,
        controller_period_s,
    )
    predicted = np.r_[next_pose, next_dynamic_state]
    return DelayedMpcStart(
        application_anchor_s=anchor,
        optimization_start_s=anchor + controller_period_s,
        aligned_state=aligned,
        predicted_state=predicted,
        fixed_history_z=aligned_history[0:2].copy(),
        active_command=active_command,
    )
