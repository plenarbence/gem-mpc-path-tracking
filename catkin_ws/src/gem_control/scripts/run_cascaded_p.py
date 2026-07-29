#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import rospy
import tf2_ros
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path as NavigationPath
from std_msgs.msg import Float64, Header

from gem_control.cascaded_p import (
    CascadedPPathController,
    OneStepCommandBuffer,
    load_cascaded_p_config,
)


def yaw_from_quaternion(quaternion) -> float:
    sin_yaw = 2.0 * (
        quaternion.w * quaternion.z
        + quaternion.x * quaternion.y
    )
    cos_yaw = 1.0 - 2.0 * (
        quaternion.y * quaternion.y
        + quaternion.z * quaternion.z
    )
    return math.atan2(sin_yaw, cos_yaw)


def odometry_state(message: Odometry) -> np.ndarray:
    yaw = yaw_from_quaternion(message.pose.pose.orientation)
    velocity = message.twist.twist.linear
    longitudinal_speed = (
        math.cos(yaw) * velocity.x + math.sin(yaw) * velocity.y
    )
    state = np.asarray(
        (
            message.pose.pose.position.x,
            message.pose.pose.position.y,
            yaw,
            longitudinal_speed,
            message.twist.twist.angular.z,
        ),
        dtype=float,
    )
    if not np.isfinite(state).all():
        raise ValueError("odometry state must be finite")
    return state


def sleep_until(target: rospy.Time) -> None:
    while not rospy.is_shutdown() and rospy.Time.now() < target:
        remaining = (target - rospy.Time.now()).to_sec()
        rospy.rostime.wallsleep(min(max(remaining, 0.0), 0.002))


class CascadedPExecutionNode:
    def __init__(self) -> None:
        self.config = load_cascaded_p_config(
            rospy.get_param("~config_path", None)
        )
        self.controller = CascadedPPathController(config=self.config)
        self.reference_speed_mps = float(
            rospy.get_param("~reference_speed_mps", 2.0)
        )
        self.drive_duration_s = float(
            rospy.get_param("~drive_duration_s", 12.0)
        )
        self.stop_ramp_duration_s = float(
            rospy.get_param("~stop_ramp_duration_s", 3.0)
        )
        self.target_laps = float(
            rospy.get_param("~target_laps", 0.0)
        )
        self.maximum_lateral_error_m = float(
            rospy.get_param("~maximum_lateral_error_m", 0.8)
        )
        self.maximum_yaw_error_rad = float(
            rospy.get_param("~maximum_yaw_error_rad", 0.7)
        )
        self.output_directory = Path(
            rospy.get_param(
                "~output_directory",
                "/workspace/results/cascaded_p/simulator_run",
            )
        )
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self._validate_parameters()

        self._lock = threading.Lock()
        self._latest_odometry: Odometry | None = None
        self._profile_start: rospy.Time | None = None
        self._commissioned_delay_s: float | None = None
        self._stop_published = False
        self._visualization_frame_id = "world"
        self._vehicle_path = NavigationPath()
        self._vehicle_path.header.frame_id = self._visualization_frame_id
        self._transform_broadcaster = tf2_ros.TransformBroadcaster()

        self.command_publisher = rospy.Publisher(
            rospy.get_param("~command_topic", "/gem/ackermann_cmd"),
            AckermannDrive,
            queue_size=1,
        )
        self.stamped_publisher = rospy.Publisher(
            rospy.get_param(
                "~stamped_command_topic",
                "/gem_control/cascaded_p/ackermann_cmd_stamped",
            ),
            AckermannDriveStamped,
            queue_size=10,
        )
        self.reference_path_publisher = rospy.Publisher(
            rospy.get_param(
                "~reference_path_topic", "/gem_control/reference_path"
            ),
            NavigationPath,
            queue_size=1,
            latch=True,
        )
        self.vehicle_path_publisher = rospy.Publisher(
            rospy.get_param(
                "~vehicle_path_topic", "/gem_control/vehicle_path"
            ),
            NavigationPath,
            queue_size=1,
            latch=True,
        )
        rospy.Subscriber(
            rospy.get_param("~odometry_topic", "/gem/base_footprint/odom"),
            Odometry,
            self._odometry_callback,
            queue_size=10,
        )
        rospy.Subscriber(
            rospy.get_param(
                "~profile_start_topic", "/gem_sysid/profile_start"
            ),
            Header,
            self._profile_start_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            rospy.get_param(
                "~commissioning_delay_topic",
                "/gem_sysid/commissioning_delay_ms",
            ),
            Float64,
            self._commissioning_delay_callback,
            queue_size=1,
        )
        rospy.on_shutdown(self.publish_stop)
        rospy.loginfo(
            "Cascaded P ready: Ky=%.3f rad/m, Kpsi=%.3f, "
            "one-step command delay without state prediction",
            self.config.lateral_to_yaw_gain_rad_per_m,
            self.config.yaw_to_steering_gain_rad_per_rad,
        )

    @staticmethod
    def _path_pose(
        x_m: float,
        y_m: float,
        yaw_rad: float,
        stamp: rospy.Time,
        frame_id: str = "world",
    ) -> PoseStamped:
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = frame_id
        pose.pose.position.x = float(x_m)
        pose.pose.position.y = float(y_m)
        pose.pose.orientation.z = math.sin(0.5 * float(yaw_rad))
        pose.pose.orientation.w = math.cos(0.5 * float(yaw_rad))
        return pose

    def _validate_parameters(self) -> None:
        if not (
            0.0
            < self.reference_speed_mps
            <= self.config.maximum_speed_command_mps
        ):
            raise ValueError(
                "reference_speed_mps must be positive and inside limits"
            )
        if self.drive_duration_s <= 0.0:
            raise ValueError("drive_duration_s must be positive")
        if self.stop_ramp_duration_s <= 0.0:
            raise ValueError("stop_ramp_duration_s must be positive")
        if not 0.0 <= self.target_laps <= 10.0:
            raise ValueError("target_laps must be in [0, 10]")
        if self.maximum_lateral_error_m <= 0.0:
            raise ValueError("maximum_lateral_error_m must be positive")
        if self.maximum_yaw_error_rad <= 0.0:
            raise ValueError("maximum_yaw_error_rad must be positive")

    def _odometry_callback(self, message: Odometry) -> None:
        transform = TransformStamped()
        transform.header.stamp = message.header.stamp
        transform.header.frame_id = (
            message.header.frame_id.lstrip("/") or "world"
        )
        transform.child_frame_id = (
            message.child_frame_id.lstrip("/") or "base_footprint"
        )
        transform.transform.translation.x = message.pose.pose.position.x
        transform.transform.translation.y = message.pose.pose.position.y
        transform.transform.translation.z = message.pose.pose.position.z
        transform.transform.rotation = message.pose.pose.orientation
        self._transform_broadcaster.sendTransform(transform)
        with self._lock:
            self._latest_odometry = message

    def _profile_start_callback(self, message: Header) -> None:
        with self._lock:
            self._profile_start = message.stamp

    def _commissioning_delay_callback(self, message: Float64) -> None:
        with self._lock:
            self._commissioned_delay_s = float(message.data) / 1000.0

    def _snapshot(
        self,
    ) -> tuple[Odometry | None, rospy.Time | None, float | None]:
        with self._lock:
            return (
                self._latest_odometry,
                self._profile_start,
                self._commissioned_delay_s,
            )

    def wait_for_startup_data(self, timeout_s: float = 30.0) -> None:
        deadline = time.monotonic() + timeout_s
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            odometry, profile_start, delay_s = self._snapshot()
            if (
                odometry is not None
                and profile_start is not None
                and delay_s is not None
                and not rospy.Time.now().is_zero()
            ):
                if not 0.0 < delay_s < self.config.period_s:
                    raise RuntimeError(
                        "commissioned delay is outside one controller period"
                    )
                rospy.loginfo(
                    "Startup data ready: commissioned mean delay %.3f ms",
                    1000.0 * delay_s,
                )
                return
            rospy.rostime.wallsleep(0.02)
        raise RuntimeError("timed out waiting for odometry and commissioning")

    def publish_command(
        self, command: np.ndarray, timestamp: rospy.Time | None = None
    ) -> rospy.Time:
        command_value = np.asarray(command, dtype=float)
        if command_value.shape != (2,) or not np.isfinite(
            command_value
        ).all():
            raise ValueError("command must be a finite [speed, steering] pair")
        stamp = rospy.Time.now() if timestamp is None else timestamp
        drive = AckermannDrive()
        drive.speed = float(command_value[0])
        drive.steering_angle = float(command_value[1])
        stamped = AckermannDriveStamped()
        stamped.header.stamp = stamp
        stamped.header.frame_id = "base_footprint"
        stamped.drive = drive
        self.stamped_publisher.publish(stamped)
        self.command_publisher.publish(drive)
        return stamp

    def publish_stop(self) -> None:
        if self._stop_published:
            return
        self._stop_published = True
        for _ in range(5):
            if rospy.is_shutdown():
                break
            self.publish_command(np.zeros(2))
            rospy.rostime.wallsleep(0.02)

    def publish_reference_path(self) -> None:
        stamp = rospy.Time.now()
        message = NavigationPath()
        message.header.stamp = stamp
        message.header.frame_id = self._visualization_frame_id
        progress = np.linspace(
            0.0, self.controller.reference_path.length, 1201
        )
        reference = self.controller.reference_path.evaluate(progress)
        message.poses = [
            self._path_pose(x_m, y_m, yaw_rad, stamp)
            for x_m, y_m, yaw_rad in zip(
                reference.x, reference.y, reference.yaw
            )
        ]
        self.reference_path_publisher.publish(message)

    def publish_vehicle_path_point(
        self, state: np.ndarray, stamp: rospy.Time
    ) -> None:
        self._vehicle_path.header.stamp = stamp
        self._vehicle_path.poses.append(
            self._path_pose(state[0], state[1], state[2], stamp)
        )
        self.vehicle_path_publisher.publish(self._vehicle_path)

    def reference_speed(self, elapsed_s: float) -> float:
        if elapsed_s <= self.drive_duration_s:
            return self.reference_speed_mps
        ramp_elapsed = elapsed_s - self.drive_duration_s
        fraction = max(
            0.0, 1.0 - ramp_elapsed / self.stop_ramp_duration_s
        )
        return self.reference_speed_mps * fraction

    def lap_reference_speed(
        self,
        elapsed_s: float,
        progress_m: float,
        target_progress_m: float,
    ) -> float:
        remaining_m = max(0.0, target_progress_m - progress_m)
        slowdown_distance_m = max(
            12.0, 4.0 * self.reference_speed_mps
        )
        finish_limited_speed = max(
            0.5,
            self.reference_speed_mps
            * min(1.0, remaining_m / slowdown_distance_m),
        )
        startup_limited_speed = max(
            0.5,
            self.reference_speed_mps * min(1.0, elapsed_s / 3.0),
        )
        return min(finish_limited_speed, startup_limited_speed)

    def run(self) -> None:
        self.wait_for_startup_data()
        odometry_message, profile_start, commissioned_delay_s = (
            self._snapshot()
        )
        assert odometry_message is not None
        assert profile_start is not None
        assert commissioned_delay_s is not None

        initial_state = odometry_state(odometry_message)
        self._visualization_frame_id = (
            odometry_message.header.frame_id.lstrip("/") or "world"
        )
        self._vehicle_path.header.frame_id = self._visualization_frame_id
        self.publish_reference_path()
        self.publish_vehicle_path_point(initial_state, rospy.Time.now())
        initial_projection = self.controller.reference_path.project(
            initial_state[0], initial_state[1]
        )

        next_tick = profile_start
        minimum_start = rospy.Time.now() + rospy.Duration(0.2)
        while next_tick < minimum_start:
            next_tick += rospy.Duration(self.config.period_s)

        command_buffer = OneStepCommandBuffer()
        active_target_speed_mps = 0.0
        progress_hint = initial_projection.s
        terminal_progress_m = initial_projection.s
        target_progress_m = (
            initial_projection.s
            + self.target_laps * self.controller.reference_path.length
            if self.target_laps > 0.0
            else None
        )
        rows: list[dict[str, object]] = []
        run_start = next_tick
        termination_reason = "completed"
        if target_progress_m is None:
            total_duration = (
                self.drive_duration_s + self.stop_ramp_duration_s
            )
        else:
            nominal_lap_time_s = (
                self.target_laps
                * self.controller.reference_path.length
                / self.reference_speed_mps
            )
            total_duration = 1.75 * nominal_lap_time_s + 60.0
            rospy.loginfo(
                "Cascaded-P lap mode: %.3f laps, %.1f m target, "
                "%.1f s timeout",
                self.target_laps,
                target_progress_m - initial_projection.s,
                total_duration,
            )

        while not rospy.is_shutdown():
            sleep_until(next_tick)
            elapsed = (next_tick - run_start).to_sec()
            if elapsed > total_duration + 1e-6:
                termination_reason = (
                    "lap_timeout"
                    if target_progress_m is not None
                    else "completed"
                )
                break

            odometry_message = self._snapshot()[0]
            if odometry_message is None:
                termination_reason = "odometry_missing"
                break
            measured_state = odometry_state(odometry_message)
            measurement_timestamp_s = (
                odometry_message.header.stamp.to_sec()
            )
            projection = self.controller.reference_path.project_local(
                measured_state[0],
                measured_state[1],
                progress_hint,
            )
            terminal_progress_m = projection.s
            measured_yaw_error = math.atan2(
                math.sin(measured_state[2] - projection.yaw),
                math.cos(measured_state[2] - projection.yaw),
            )
            if (
                abs(projection.signed_lateral_error)
                > self.maximum_lateral_error_m
                or abs(measured_yaw_error) > self.maximum_yaw_error_rad
            ):
                termination_reason = "tracking_safety_limit"
                break
            if (
                target_progress_m is not None
                and projection.s >= target_progress_m
            ):
                termination_reason = "completed_laps"
                break

            published_command = command_buffer.command_for_tick
            publish_stamp = self.publish_command(published_command)
            measurement_age_at_takeover_s = (
                publish_stamp.to_sec()
                + commissioned_delay_s
                - measurement_timestamp_s
            )
            if (
                measurement_age_at_takeover_s
                > self.config.maximum_odometry_age_s
            ):
                termination_reason = "odometry_too_old"
                break

            target_speed = (
                self.reference_speed(elapsed)
                if target_progress_m is None
                else self.lap_reference_speed(
                    elapsed, projection.s, target_progress_m
                )
            )
            compute_start = time.perf_counter()
            result = self.controller.compute_command(
                x_m=measured_state[0],
                y_m=measured_state[1],
                yaw_rad=measured_state[2],
                previous_progress_m=progress_hint,
                reference_speed_mps=target_speed,
            )
            controller_compute_s = time.perf_counter() - compute_start
            command_buffer.stage_for_next_tick(result.command)

            if len(rows) % 5 == 0:
                self.publish_vehicle_path_point(
                    measured_state, rospy.Time.now()
                )
            rows.append(
                {
                    "elapsed_s": elapsed,
                    "publish_time_s": publish_stamp.to_sec(),
                    "odometry_time_s": measurement_timestamp_s,
                    "measurement_age_at_takeover_ms": (
                        1000.0 * measurement_age_at_takeover_s
                    ),
                    "commissioned_delay_ms": (
                        1000.0 * commissioned_delay_s
                    ),
                    "command_delay_steps": 1,
                    "state_prediction_enabled": False,
                    "x_m": measured_state[0],
                    "y_m": measured_state[1],
                    "yaw_rad": measured_state[2],
                    "speed_mps": measured_state[3],
                    "yaw_rate_radps": measured_state[4],
                    "progress_m": projection.s,
                    "lateral_error_m": (
                        projection.signed_lateral_error
                    ),
                    "yaw_error_rad": measured_yaw_error,
                    "reference_speed_mps": active_target_speed_mps,
                    "published_speed_command_mps": published_command[0],
                    "published_steering_command_rad": (
                        published_command[1]
                    ),
                    "next_reference_speed_mps": target_speed,
                    "next_speed_command_mps": result.command[0],
                    "next_steering_command_rad": result.command[1],
                    "next_raw_steering_command_rad": (
                        result.raw_steering_command_rad
                    ),
                    "next_desired_yaw_rad": result.desired_yaw_rad,
                    "next_yaw_compensation_rad": (
                        result.yaw_compensation_rad
                    ),
                    "next_inner_yaw_error_rad": (
                        result.inner_yaw_error_rad
                    ),
                    "next_steering_saturated": (
                        result.steering_saturated
                    ),
                    "next_speed_saturated": result.speed_saturated,
                    "controller_compute_s": controller_compute_s,
                    "deadline_met": (
                        controller_compute_s < self.config.period_s
                    ),
                }
            )
            active_target_speed_mps = target_speed
            progress_hint = projection.s
            next_tick += rospy.Duration(self.config.period_s)

        stop_result = self.hold_stop_until_stationary()
        self.publish_stop()
        self.write_results(
            rows,
            termination_reason,
            stop_result,
            initial_projection.s,
            target_progress_m,
            terminal_progress_m,
        )
        rospy.loginfo(
            "Cascaded-P run finished: %s, %d control cycles",
            termination_reason,
            len(rows),
        )

    def hold_stop_until_stationary(
        self, timeout_s: float = 4.0
    ) -> dict[str, object]:
        start = rospy.Time.now()
        next_publish = start
        stationary_samples = 0
        final_speed = float("nan")
        while (
            not rospy.is_shutdown()
            and (rospy.Time.now() - start).to_sec() <= timeout_s
        ):
            sleep_until(next_publish)
            self.publish_command(np.zeros(2))
            odometry_message = self._snapshot()[0]
            if odometry_message is not None:
                final_speed = float(odometry_state(odometry_message)[3])
                stationary_samples = (
                    stationary_samples + 1
                    if abs(final_speed) < 0.1
                    else 0
                )
                if stationary_samples >= 3:
                    break
            next_publish += rospy.Duration(self.config.period_s)
        return {
            "stationary": stationary_samples >= 3,
            "settling_duration_s": (
                rospy.Time.now() - start
            ).to_sec(),
            "final_speed_mps": final_speed,
        }

    def write_results(
        self,
        rows: list[dict[str, object]],
        termination_reason: str,
        stop_result: dict[str, object],
        initial_progress_m: float,
        target_progress_m: float | None,
        terminal_progress_m: float,
    ) -> None:
        if rows:
            with (self.output_directory / "control_log.csv").open(
                "w", newline="", encoding="ascii"
            ) as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=list(rows[0].keys())
                )
                writer.writeheader()
                writer.writerows(rows)
        lateral_errors = np.asarray(
            [float(row["lateral_error_m"]) for row in rows], dtype=float
        )
        compute_times = np.asarray(
            [float(row["controller_compute_s"]) for row in rows],
            dtype=float,
        )
        summary = {
            "controller": "cascaded_p",
            "termination_reason": termination_reason,
            "cycle_count": len(rows),
            "reference_speed_mps": self.reference_speed_mps,
            "target_laps": self.target_laps,
            "path_length_m": self.controller.reference_path.length,
            "initial_progress_m": initial_progress_m,
            "target_progress_m": target_progress_m,
            "last_logged_progress_m": (
                float(rows[-1]["progress_m"])
                if rows
                else initial_progress_m
            ),
            "final_progress_m": terminal_progress_m,
            "achieved_laps": (
                (terminal_progress_m - initial_progress_m)
                / self.controller.reference_path.length
            ),
            "gains": {
                "lateral_to_yaw_rad_per_m": (
                    self.config.lateral_to_yaw_gain_rad_per_m
                ),
                "yaw_to_steering_rad_per_rad": (
                    self.config.yaw_to_steering_gain_rad_per_rad
                ),
                "yaw_compensation_limit_rad": (
                    self.config.desired_yaw_compensation_limit_rad
                ),
            },
            "timing_design": {
                "controller_period_s": self.config.period_s,
                "command_delay_steps": 1,
                "state_prediction_enabled": False,
                "commissioned_delay_s": (
                    self._commissioned_delay_s
                ),
            },
            "deadline_miss_count": int(
                sum(not bool(row["deadline_met"]) for row in rows)
            ),
            "mean_controller_compute_time_s": (
                float(np.mean(compute_times))
                if len(compute_times)
                else None
            ),
            "maximum_controller_compute_time_s": (
                float(np.max(compute_times))
                if len(compute_times)
                else None
            ),
            "maximum_absolute_lateral_error_m": (
                float(np.max(np.abs(lateral_errors)))
                if len(lateral_errors)
                else None
            ),
            "stop": stop_result,
        }
        (self.output_directory / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="ascii",
        )


def main() -> None:
    rospy.init_node("cascaded_p_controller")
    node: CascadedPExecutionNode | None = None
    try:
        node = CascadedPExecutionNode()
        node.run()
    except Exception as error:
        if node is not None:
            node.publish_stop()
        rospy.logfatal("Cascaded P failed: %s", error)
        raise


if __name__ == "__main__":
    main()
