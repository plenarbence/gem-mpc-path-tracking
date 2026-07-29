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

from gem_control.full_mpc import (
    FullLearnedMpc,
    FullMpcInitialCondition,
    load_full_mpc_config,
)
from gem_control.timing_compensation import (
    TimedVehicleState,
    prepare_delayed_mpc_start,
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


def odometry_state(
    message: Odometry,
    availability_timestamp_s: float | None = None,
) -> TimedVehicleState:
    yaw = yaw_from_quaternion(message.pose.pose.orientation)
    velocity = message.twist.twist.linear
    longitudinal_speed = (
        math.cos(yaw) * velocity.x + math.sin(yaw) * velocity.y
    )
    return TimedVehicleState(
        timestamp_s=message.header.stamp.to_sec(),
        state=np.asarray(
            (
                message.pose.pose.position.x,
                message.pose.pose.position.y,
                yaw,
                longitudinal_speed,
                message.twist.twist.angular.z,
            )
        ),
        availability_timestamp_s=availability_timestamp_s,
    )


def sleep_until(target: rospy.Time) -> None:
    while not rospy.is_shutdown() and rospy.Time.now() < target:
        remaining = (target - rospy.Time.now()).to_sec()
        rospy.rostime.wallsleep(min(max(remaining, 0.0), 0.002))


class FullMpcExecutionNode:
    def __init__(self) -> None:
        self.config = load_full_mpc_config(
            rospy.get_param("~config_path", None)
        )
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
        self.maximum_consecutive_failures = int(
            rospy.get_param("~maximum_consecutive_failures", 3)
        )
        self.output_directory = Path(
            rospy.get_param(
                "~output_directory",
                "/workspace/results/mpc/simulator_run",
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
                "/gem_control/ackermann_cmd_stamped",
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
            "Building horizon-%d learned MPC",
            self.config.horizon_steps,
        )
        self.mpc = FullLearnedMpc(config=self.config)
        rospy.loginfo(
            "MPC solvers built in %.3f s",
            self.mpc.solver_construction_time_s,
        )

    @staticmethod
    def _path_pose(
        x_m: float,
        y_m: float,
        yaw_rad: float,
        stamp: rospy.Time,
        frame_id: str = "odom",
    ) -> PoseStamped:
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = frame_id
        pose.pose.position.x = float(x_m)
        pose.pose.position.y = float(y_m)
        pose.pose.orientation.z = math.sin(0.5 * float(yaw_rad))
        pose.pose.orientation.w = math.cos(0.5 * float(yaw_rad))
        return pose

    def publish_reference_path(self) -> None:
        stamp = rospy.Time.now()
        message = NavigationPath()
        message.header.stamp = stamp
        message.header.frame_id = self._visualization_frame_id
        progress = np.linspace(
            0.0,
            self.mpc.reference_path.length,
            1201,
        )
        reference = self.mpc.reference_path.evaluate(progress)
        message.poses = [
            self._path_pose(x_m, y_m, yaw_rad, stamp)
            for x_m, y_m, yaw_rad in zip(
                reference.x,
                reference.y,
                reference.yaw,
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

    def _validate_parameters(self) -> None:
        if not (
            0.0
            < self.reference_speed_mps
            <= self.config.limits.maximum_speed_command_mps
        ):
            raise ValueError(
                "reference_speed_mps must be positive and inside the "
                "identified command limit"
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
        if self.maximum_consecutive_failures < 1:
            raise ValueError("maximum_consecutive_failures must be positive")

    def _odometry_callback(self, message: Odometry) -> None:
        transform = TransformStamped()
        transform.header.stamp = message.header.stamp
        transform.header.frame_id = (
            message.header.frame_id.lstrip("/") or "world"
        )
        transform.child_frame_id = (
            message.child_frame_id.lstrip("/") or "base_footprint"
        )
        transform.transform.translation.x = (
            message.pose.pose.position.x
        )
        transform.transform.translation.y = (
            message.pose.pose.position.y
        )
        transform.transform.translation.z = (
            message.pose.pose.position.z
        )
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
                        "Commissioned delay is outside one controller period"
                    )
                rospy.loginfo(
                    "Startup data ready: commissioned mean delay %.3f ms",
                    1000.0 * delay_s,
                )
                return
            rospy.rostime.wallsleep(0.02)
        raise RuntimeError("Timed out waiting for odometry and commissioning")

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
        stop = np.zeros(2)
        for _ in range(5):
            if rospy.is_shutdown():
                break
            self.publish_command(stop)
            rospy.rostime.wallsleep(0.02)

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

        initial_measurement = odometry_state(odometry_message)
        initial_state = initial_measurement.validated_state()
        self._visualization_frame_id = (
            odometry_message.header.frame_id.lstrip("/") or "world"
        )
        self._vehicle_path.header.frame_id = (
            self._visualization_frame_id
        )
        self.publish_reference_path()
        self.publish_vehicle_path_point(initial_state, rospy.Time.now())
        initial_projection = self.mpc.reference_path.project(
            initial_state[0], initial_state[1]
        )
        stationary_initial = FullMpcInitialCondition(
            state=initial_state,
            fixed_history_z=np.tile(
                np.r_[initial_state[3:5], np.zeros(2)], (2, 1)
            ),
            previous_command=np.zeros(2),
            previous_progress_m=initial_projection.s,
        )
        warmup = self.mpc.stationary_warmup(stationary_initial)
        if not all(item.solution_accepted for item in warmup):
            raise RuntimeError("Stationary MPC warmup did not converge")
        rospy.loginfo(
            "Stationary warmup complete: %s ms",
            ", ".join(
                "{:.1f}".format(1000.0 * item.solve_time_s)
                for item in warmup
            ),
        )

        next_tick = profile_start
        minimum_start = rospy.Time.now() + rospy.Duration(0.2)
        while next_tick < minimum_start:
            next_tick += rospy.Duration(self.config.period_s)

        active_command = np.zeros(2)
        history_z = np.tile(
            np.r_[initial_state[3:5], active_command], (3, 1)
        )
        progress_hint = initial_projection.s
        terminal_progress_m = initial_projection.s
        target_progress_m = (
            initial_projection.s
            + self.target_laps * self.mpc.reference_path.length
            if self.target_laps > 0.0
            else None
        )
        consecutive_failures = 0
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
                * self.mpc.reference_path.length
                / self.reference_speed_mps
            )
            total_duration = 1.75 * nominal_lap_time_s + 60.0
            rospy.loginfo(
                "Full-lap mode: %.3f laps, %.1f m target, %.1f s timeout",
                self.target_laps,
                target_progress_m - initial_projection.s,
                total_duration,
            )

        while not rospy.is_shutdown():
            sleep_until(next_tick)
            elapsed = (next_tick - run_start).to_sec()
            if elapsed > total_duration + 1e-6:
                if target_progress_m is not None:
                    termination_reason = "lap_timeout"
                break

            odometry_message = self._snapshot()[0]
            if odometry_message is None:
                termination_reason = "odometry_missing"
                break
            measurement = odometry_state(
                odometry_message,
                availability_timestamp_s=rospy.Time.now().to_sec(),
            )
            measured_state = measurement.validated_state()
            if len(rows) % 5 == 0:
                self.publish_vehicle_path_point(
                    measured_state, rospy.Time.now()
                )
            projection = self.mpc.reference_path.project_local(
                measured_state[0],
                measured_state[1],
                progress_hint,
            )
            terminal_progress_m = projection.s
            yaw_error = math.atan2(
                math.sin(measured_state[2] - projection.yaw),
                math.cos(measured_state[2] - projection.yaw),
            )
            if (
                abs(projection.signed_lateral_error)
                > self.maximum_lateral_error_m
                or abs(yaw_error) > self.maximum_yaw_error_rad
            ):
                termination_reason = "tracking_safety_limit"
                break
            if (
                target_progress_m is not None
                and projection.s >= target_progress_m
            ):
                termination_reason = "completed_laps"
                break

            publish_stamp = self.publish_command(active_command)
            applied_history = np.vstack(
                (
                    np.r_[measured_state[3:5], active_command],
                    history_z[0],
                    history_z[1],
                )
            )
            cycle_start = time.perf_counter()
            delayed = prepare_delayed_mpc_start(
                model=self.mpc.model,
                odometry=measurement,
                command_publish_timestamp_s=publish_stamp.to_sec(),
                commissioned_takeover_delay_s=commissioned_delay_s,
                controller_period_s=self.config.period_s,
                applied_history_z=applied_history,
                maximum_odometry_age_s=self.config.maximum_odometry_age_s,
            )
            target_speed = (
                self.reference_speed(elapsed)
                if target_progress_m is None
                else self.lap_reference_speed(
                    elapsed, projection.s, target_progress_m
                )
            )
            result = self.mpc.solve(
                FullMpcInitialCondition(
                    state=delayed.predicted_state,
                    fixed_history_z=delayed.fixed_history_z,
                    previous_command=delayed.active_command,
                    previous_progress_m=progress_hint,
                ),
                reference_speed_mps=target_speed,
            )
            cycle_compute_s = time.perf_counter() - cycle_start
            history_z = applied_history
            history_z[0, 0:2] = delayed.aligned_state[3:5]
            progress_hint = projection.s

            if result.success:
                active_command = result.first_command.copy()
                consecutive_failures = 0
            else:
                consecutive_failures += 1
            rows.append(
                {
                    "elapsed_s": elapsed,
                    "publish_time_s": publish_stamp.to_sec(),
                    "odometry_time_s": measurement.timestamp_s,
                    "commissioned_delay_ms": 1000.0
                    * commissioned_delay_s,
                    "x_m": measured_state[0],
                    "y_m": measured_state[1],
                    "yaw_rad": measured_state[2],
                    "speed_mps": measured_state[3],
                    "yaw_rate_radps": measured_state[4],
                    "progress_m": projection.s,
                    "lateral_error_m": projection.signed_lateral_error,
                    "yaw_error_rad": yaw_error,
                    "reference_speed_mps": target_speed,
                    "published_speed_command_mps": delayed.active_command[0],
                    "published_steering_command_rad": delayed.active_command[1],
                    "next_speed_command_mps": active_command[0],
                    "next_steering_command_rad": active_command[1],
                    "solver_success": result.success,
                    "solver_status": result.diagnostics.return_status,
                    "solver_iterations": result.diagnostics.iteration_count,
                    "solver_time_s": result.diagnostics.solve_time_s,
                    "mpc_total_compute_s": (
                        result.diagnostics.total_compute_time_s
                    ),
                    "cycle_compute_s": cycle_compute_s,
                    "deadline_met": result.diagnostics.deadline_met,
                    "constraint_violation": (
                        result.diagnostics.maximum_constraint_violation
                    ),
                    "fallback_reason": result.diagnostics.fallback_reason,
                }
            )
            if consecutive_failures >= self.maximum_consecutive_failures:
                termination_reason = "consecutive_solver_failures"
                break
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
            "MPC run finished: %s, %d control cycles",
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
                final_speed = float(
                    odometry_state(odometry_message).state[3]
                )
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
        self.output_directory.mkdir(parents=True, exist_ok=True)
        csv_path = self.output_directory / "control_log.csv"
        if rows:
            with csv_path.open("w", newline="", encoding="ascii") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=list(rows[0].keys())
                )
                writer.writeheader()
                writer.writerows(rows)
        solver_times = np.asarray(
            [float(row["solver_time_s"]) for row in rows], dtype=float
        )
        total_compute_times = np.asarray(
            [float(row["mpc_total_compute_s"]) for row in rows],
            dtype=float,
        )
        lateral_errors = np.asarray(
            [float(row["lateral_error_m"]) for row in rows], dtype=float
        )
        last_logged_progress_m = (
            float(rows[-1]["progress_m"])
            if rows
            else initial_progress_m
        )
        summary = {
            "termination_reason": termination_reason,
            "horizon_steps": self.config.horizon_steps,
            "cycle_count": len(rows),
            "reference_speed_mps": self.reference_speed_mps,
            "drive_duration_s": self.drive_duration_s,
            "stop_ramp_duration_s": self.stop_ramp_duration_s,
            "target_laps": self.target_laps,
            "path_length_m": self.mpc.reference_path.length,
            "initial_progress_m": initial_progress_m,
            "target_progress_m": target_progress_m,
            "last_logged_progress_m": last_logged_progress_m,
            "final_progress_m": terminal_progress_m,
            "achieved_laps": (
                (terminal_progress_m - initial_progress_m)
                / self.mpc.reference_path.length
            ),
            "accepted_solve_count": int(
                sum(bool(row["solver_success"]) for row in rows)
            ),
            "deadline_miss_count": int(
                sum(not bool(row["deadline_met"]) for row in rows)
            ),
            "mean_solver_time_s": (
                float(np.mean(solver_times)) if len(solver_times) else None
            ),
            "p95_solver_time_s": (
                float(np.quantile(solver_times, 0.95))
                if len(solver_times)
                else None
            ),
            "maximum_solver_time_s": (
                float(np.max(solver_times)) if len(solver_times) else None
            ),
            "mean_total_compute_time_s": (
                float(np.mean(total_compute_times))
                if len(total_compute_times)
                else None
            ),
            "p95_total_compute_time_s": (
                float(np.quantile(total_compute_times, 0.95))
                if len(total_compute_times)
                else None
            ),
            "maximum_total_compute_time_s": (
                float(np.max(total_compute_times))
                if len(total_compute_times)
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
    rospy.init_node("full_learned_mpc")
    node: FullMpcExecutionNode | None = None
    try:
        node = FullMpcExecutionNode()
        node.run()
    except Exception as error:
        if node is not None:
            node.publish_stop()
        rospy.logfatal("Full learned MPC failed: %s", error)
        raise


if __name__ == "__main__":
    main()
