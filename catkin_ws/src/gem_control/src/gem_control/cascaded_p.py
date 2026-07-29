from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from gem_control.reference_path import (
    ClosedReferencePath,
    ProjectionResult,
    build_configured_reference_path,
)


@dataclass(frozen=True)
class CascadedPConfig:
    period_s: float = 0.1
    lateral_to_yaw_gain_rad_per_m: float = 0.27
    yaw_to_steering_gain_rad_per_rad: float = 0.9
    desired_yaw_compensation_limit_rad: float = math.radians(30.0)
    minimum_speed_command_mps: float = 0.0
    maximum_speed_command_mps: float = 5.5
    maximum_steering_command_rad: float = 0.3
    maximum_odometry_age_s: float = 0.12

    def validate(self) -> None:
        values = (
            self.period_s,
            self.lateral_to_yaw_gain_rad_per_m,
            self.yaw_to_steering_gain_rad_per_rad,
            self.desired_yaw_compensation_limit_rad,
            self.maximum_speed_command_mps,
            self.maximum_steering_command_rad,
            self.maximum_odometry_age_s,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("positive cascaded-P parameters must be finite")
        if (
            not np.isfinite(self.minimum_speed_command_mps)
            or self.minimum_speed_command_mps < 0.0
            or self.minimum_speed_command_mps
            >= self.maximum_speed_command_mps
        ):
            raise ValueError("invalid cascaded-P speed-command limits")


@dataclass(frozen=True)
class CascadedPCommand:
    command: np.ndarray
    raw_steering_command_rad: float
    desired_yaw_rad: float
    yaw_compensation_rad: float
    inner_yaw_error_rad: float
    steering_saturated: bool
    speed_saturated: bool
    projection: ProjectionResult


class OneStepCommandBuffer:
    """Hold a calculated command until the following controller tick."""

    def __init__(self, initial_command: np.ndarray | None = None) -> None:
        command = (
            np.zeros(2)
            if initial_command is None
            else np.asarray(initial_command, dtype=float)
        )
        self._command = self._validated_copy(command)

    @property
    def command_for_tick(self) -> np.ndarray:
        return self._command.copy()

    def stage_for_next_tick(self, command: np.ndarray) -> None:
        self._command = self._validated_copy(command)

    @staticmethod
    def _validated_copy(command: np.ndarray) -> np.ndarray:
        value = np.asarray(command, dtype=float)
        if value.shape != (2,) or not np.isfinite(value).all():
            raise ValueError("buffered command must be a finite pair")
        return value.copy()


class CascadedPPathController:
    """Lateral-error P loop followed by a yaw-error P loop."""

    def __init__(
        self,
        config: CascadedPConfig | None = None,
        reference_path: ClosedReferencePath | None = None,
    ) -> None:
        self.config = config or load_cascaded_p_config()
        self.config.validate()
        self.reference_path = (
            reference_path or build_configured_reference_path()[0]
        )

    def compute_command(
        self,
        *,
        x_m: float,
        y_m: float,
        yaw_rad: float,
        previous_progress_m: float,
        reference_speed_mps: float,
    ) -> CascadedPCommand:
        values = np.asarray(
            (x_m, y_m, yaw_rad, previous_progress_m, reference_speed_mps),
            dtype=float,
        )
        if not np.isfinite(values).all():
            raise ValueError("cascaded-P inputs must be finite")

        projection = self.reference_path.project_local(
            x_m,
            y_m,
            previous_progress_m,
        )
        raw_compensation = (
            -self.config.lateral_to_yaw_gain_rad_per_m
            * projection.signed_lateral_error
        )
        compensation = float(
            np.clip(
                raw_compensation,
                -self.config.desired_yaw_compensation_limit_rad,
                self.config.desired_yaw_compensation_limit_rad,
            )
        )
        desired_yaw = float(projection.yaw + compensation)
        inner_yaw_error = math.atan2(
            math.sin(desired_yaw - yaw_rad),
            math.cos(desired_yaw - yaw_rad),
        )
        raw_steering = (
            self.config.yaw_to_steering_gain_rad_per_rad
            * inner_yaw_error
        )
        steering = float(
            np.clip(
                raw_steering,
                -self.config.maximum_steering_command_rad,
                self.config.maximum_steering_command_rad,
            )
        )
        speed = float(
            np.clip(
                reference_speed_mps,
                self.config.minimum_speed_command_mps,
                self.config.maximum_speed_command_mps,
            )
        )
        return CascadedPCommand(
            command=np.asarray((speed, steering), dtype=float),
            raw_steering_command_rad=float(raw_steering),
            desired_yaw_rad=desired_yaw,
            yaw_compensation_rad=compensation,
            inner_yaw_error_rad=float(inner_yaw_error),
            steering_saturated=not math.isclose(
                steering, raw_steering, rel_tol=0.0, abs_tol=1e-12
            ),
            speed_saturated=not math.isclose(
                speed,
                reference_speed_mps,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            projection=projection,
        )


def load_cascaded_p_config(
    path: Path | str | None = None,
) -> CascadedPConfig:
    config_path = (
        Path(path)
        if path is not None
        else _control_package_path() / "config" / "cascaded_p.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="ascii"))
    gains = payload["gains"]
    limits = payload["limits"]
    timing = payload["timing"]
    config = CascadedPConfig(
        period_s=float(payload["period_s"]),
        lateral_to_yaw_gain_rad_per_m=float(
            gains["lateral_to_yaw_rad_per_m"]
        ),
        yaw_to_steering_gain_rad_per_rad=float(
            gains["yaw_to_steering_rad_per_rad"]
        ),
        desired_yaw_compensation_limit_rad=math.radians(
            float(gains["yaw_compensation_limit_deg"])
        ),
        minimum_speed_command_mps=float(
            limits["minimum_speed_command_mps"]
        ),
        maximum_speed_command_mps=float(
            limits["maximum_speed_command_mps"]
        ),
        maximum_steering_command_rad=float(
            limits["maximum_steering_command_rad"]
        ),
        maximum_odometry_age_s=float(timing["maximum_odometry_age_s"]),
    )
    config.validate()
    return config


def _control_package_path() -> Path:
    try:
        import rospkg
    except ImportError:
        return Path(__file__).resolve().parents[2]
    try:
        return Path(rospkg.RosPack().get_path("gem_control"))
    except rospkg.ResourceNotFound:
        return Path(__file__).resolve().parents[2]
