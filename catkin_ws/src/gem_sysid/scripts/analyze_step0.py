#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rosbag


STAMPED_COMMAND_TOPIC = "/gem_sysid/ackermann_cmd_stamped"
PROFILE_START_TOPIC = "/gem_sysid/profile_start"
COMMISSIONING_DELAY_TOPIC = "/gem_sysid/commissioning_delay_ms"
ODOM_TOPIC = "/gem/base_footprint/odom"
IMU_TOPIC = "/gem/imu"
JOINT_TOPIC = "/gem/joint_states"
LEFT_STEERING_TARGET_TOPIC = "/gem/left_steering_ctrlr/command"
RIGHT_STEERING_TARGET_TOPIC = "/gem/right_steering_ctrlr/command"
LEFT_FRONT_TARGET_TOPIC = "/gem/left_front_wheel_ctrlr/command"
RIGHT_FRONT_TARGET_TOPIC = "/gem/right_front_wheel_ctrlr/command"
LEFT_REAR_TARGET_TOPIC = "/gem/left_rear_wheel_ctrlr/command"
RIGHT_REAR_TARGET_TOPIC = "/gem/right_rear_wheel_ctrlr/command"

LOWER_COMMAND_TOPICS = (
    LEFT_STEERING_TARGET_TOPIC,
    RIGHT_STEERING_TARGET_TOPIC,
    LEFT_FRONT_TARGET_TOPIC,
    RIGHT_FRONT_TARGET_TOPIC,
    LEFT_REAR_TARGET_TOPIC,
    RIGHT_REAR_TARGET_TOPIC,
)

RECORDED_TOPICS = (
    STAMPED_COMMAND_TOPIC,
    PROFILE_START_TOPIC,
    COMMISSIONING_DELAY_TOPIC,
    ODOM_TOPIC,
    IMU_TOPIC,
    JOINT_TOPIC,
    *LOWER_COMMAND_TOPICS,
)


def message_time(message: Any, bag_time: Any) -> float:
    if hasattr(message, "header") and not message.header.stamp.is_zero():
        return message.header.stamp.to_sec()
    if hasattr(message, "stamp") and not message.stamp.is_zero():
        return message.stamp.to_sec()
    return bag_time.to_sec()


def timing_summary(times: list[float]) -> dict[str, float | int | None]:
    if len(times) < 2:
        return {
            "message_count": len(times),
            "mean_rate_hz": None,
            "min_interval_ms": None,
            "max_interval_ms": None,
        }
    intervals = np.diff(np.asarray(times, dtype=float))
    return {
        "message_count": len(times),
        "mean_rate_hz": float(1.0 / np.mean(intervals)),
        "min_interval_ms": float(1000.0 * np.min(intervals)),
        "max_interval_ms": float(1000.0 * np.max(intervals)),
    }


def joint_position(message: Any, joint_name: str) -> float:
    try:
        index = list(message.name).index(joint_name)
    except ValueError:
        return float("nan")
    return float(message.position[index])


def relative(times: list[float], start_time: float) -> np.ndarray:
    return np.asarray(times, dtype=float) - start_time


def yaw_from_quaternion(orientation: Any) -> float:
    return float(
        np.arctan2(
            2.0
            * (
                orientation.w * orientation.z
                + orientation.x * orientation.y
            ),
            1.0
            - 2.0
            * (
                orientation.y * orientation.y
                + orientation.z * orientation.z
            ),
        )
    )


def extract_profile_commands(
    commands: list[tuple[float, Any]],
    start_time: float,
    sample_period_s: float = 0.1,
    tolerance_s: float = 0.005,
) -> list[tuple[float, Any]]:
    profile: list[tuple[float, Any]] = []
    expected_time = start_time
    for command in commands:
        command_time = command[0]
        if abs(command_time - expected_time) <= tolerance_s:
            profile.append(command)
            expected_time += sample_period_s
        elif command_time > expected_time + tolerance_s:
            break
    return profile


def analyze(
    bag_path: Path,
    output_dir: Path,
    artifact_prefix: str = "step0",
) -> tuple[Path, Path]:
    records: dict[str, list[tuple[float, Any]]] = {
        topic: [] for topic in RECORDED_TOPICS
    }
    with rosbag.Bag(str(bag_path), "r") as bag:
        for topic, message, bag_time in bag.read_messages(
            topics=list(RECORDED_TOPICS)
        ):
            records[topic].append((message_time(message, bag_time), message))

    profile_starts = records[PROFILE_START_TOPIC]
    stamped_commands = records[STAMPED_COMMAND_TOPIC]
    if not stamped_commands:
        raise RuntimeError(
            f"Bag contains no messages on {STAMPED_COMMAND_TOPIC}"
        )
    start_time = (
        profile_starts[-1][1].stamp.to_sec()
        if profile_starts
        else stamped_commands[0][0]
    )

    for topic in RECORDED_TOPICS:
        if topic not in (PROFILE_START_TOPIC, COMMISSIONING_DELAY_TOPIC):
            records[topic] = [
                item for item in records[topic] if item[0] >= start_time
            ]
    stamped_commands = extract_profile_commands(
        records[STAMPED_COMMAND_TOPIC],
        start_time,
    )
    if not stamped_commands:
        raise RuntimeError("Bag contains no profile commands after profile start")
    records[STAMPED_COMMAND_TOPIC] = stamped_commands

    command_times = [item[0] for item in stamped_commands]
    command_speed = [float(item[1].drive.speed) for item in stamped_commands]
    command_steering = [
        float(item[1].drive.steering_angle) for item in stamped_commands
    ]

    odom_times = [item[0] for item in records[ODOM_TOPIC]]
    odom_speed = []
    for _, message in records[ODOM_TOPIC]:
        yaw = yaw_from_quaternion(message.pose.pose.orientation)
        velocity = message.twist.twist.linear
        odom_speed.append(
            float(np.cos(yaw) * velocity.x + np.sin(yaw) * velocity.y)
        )
    imu_times = [item[0] for item in records[IMU_TOPIC]]
    imu_acceleration = [
        float(item[1].linear_acceleration.x) for item in records[IMU_TOPIC]
    ]
    imu_yaw_rate = [
        float(item[1].angular_velocity.z) for item in records[IMU_TOPIC]
    ]
    joint_times = [item[0] for item in records[JOINT_TOPIC]]
    left_steering_actual = [
        joint_position(item[1], "left_steering_hinge_joint")
        for item in records[JOINT_TOPIC]
    ]
    right_steering_actual = [
        joint_position(item[1], "right_steering_hinge_joint")
        for item in records[JOINT_TOPIC]
    ]

    left_target_times = [
        item[0] for item in records[LEFT_STEERING_TARGET_TOPIC]
    ]
    left_target = [
        float(item[1].data) for item in records[LEFT_STEERING_TARGET_TOPIC]
    ]
    right_target_times = [
        item[0] for item in records[RIGHT_STEERING_TARGET_TOPIC]
    ]
    right_target = [
        float(item[1].data) for item in records[RIGHT_STEERING_TARGET_TOPIC]
    ]

    delays_ms = np.asarray([], dtype=float)
    command_time_array = np.asarray(command_times, dtype=float)
    next_lower_times = []
    for topic in LOWER_COMMAND_TOPICS:
        topic_times = np.asarray(
            [item[0] for item in records[topic]],
            dtype=float,
        )
        indices = np.searchsorted(topic_times, command_time_array, side="left")
        if not topic_times.size or np.any(indices >= topic_times.size):
            next_lower_times = []
            break
        next_lower_times.append(topic_times[indices])
    if next_lower_times:
        lower_batch_centers = np.mean(
            np.vstack(next_lower_times),
            axis=0,
        )
        delays_ms = (lower_batch_centers - command_time_array) * 1000.0

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / f"{artifact_prefix}_signals.png"
    figure, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

    axes[0].plot(
        relative(command_times, start_time),
        command_speed,
        label="Ackermann speed command",
    )
    if odom_times:
        axes[0].plot(
            relative(odom_times, start_time),
            odom_speed,
            label="Odometry longitudinal speed",
        )
    axes[0].set_ylabel("Speed [m/s]")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        relative(command_times, start_time),
        command_steering,
        label="Ackermann center steering",
    )
    if left_target_times:
        axes[1].plot(
            relative(left_target_times, start_time),
            left_target,
            label="Left steering setpoint",
        )
    if right_target_times:
        axes[1].plot(
            relative(right_target_times, start_time),
            right_target,
            label="Right steering setpoint",
        )
    if joint_times:
        axes[1].plot(
            relative(joint_times, start_time),
            left_steering_actual,
            label="Left steering measured",
            alpha=0.8,
        )
        axes[1].plot(
            relative(joint_times, start_time),
            right_steering_actual,
            label="Right steering measured",
            alpha=0.8,
        )
    axes[1].set_ylabel("Steering [rad]")
    axes[1].legend(ncol=2)
    axes[1].grid(True)

    if imu_times:
        imu_relative = relative(imu_times, start_time)
        axes[2].plot(
            imu_relative, imu_acceleration, label="Longitudinal acceleration"
        )
        axes[2].plot(imu_relative, imu_yaw_rate, label="Yaw rate")
    axes[2].set_ylabel("IMU values")
    axes[2].legend()
    axes[2].grid(True)

    if delays_ms.size:
        axes[3].plot(
            relative(command_times[: len(delays_ms)], start_time),
            delays_ms,
            marker=".",
            linestyle="none",
            label="Publish to next six-topic lower batch",
        )
    axes[3].set_ylabel("Delay [ms]")
    axes[3].set_xlabel("Time since first command [s]")
    axes[3].legend()
    axes[3].grid(True)

    figure.tight_layout()
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)

    topic_timing = {
        topic: timing_summary([item[0] for item in topic_records])
        for topic, topic_records in records.items()
    }
    delay_summary: dict[str, float | int | None] = {
        "sample_count": int(delays_ms.size),
        "mean_ms": float(np.mean(delays_ms)) if delays_ms.size else None,
        "min_ms": float(np.min(delays_ms)) if delays_ms.size else None,
        "max_ms": float(np.max(delays_ms)) if delays_ms.size else None,
        "p95_ms": float(np.percentile(delays_ms, 95))
        if delays_ms.size
        else None,
    }
    summary = {
        "bag": str(bag_path.resolve()),
        "profile_start_time_s": start_time,
        "commissioning_mean_delay_ms": (
            float(records[COMMISSIONING_DELAY_TOPIC][-1][1].data)
            if records[COMMISSIONING_DELAY_TOPIC]
            else None
        ),
        "topic_timing": topic_timing,
        "ackermann_publish_to_next_lower_update": delay_summary,
        "note": (
            "The lower-batch delay uses the mean bag-receipt time of the next "
            "message on all six lower command topics. It is an observed upper "
            "bound, not a stamped controller callback time."
        ),
    }
    summary_path = output_dir / f"{artifact_prefix}_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="ascii"
    )
    return figure_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Step 0 signals and summarize message timing."
    )
    parser.add_argument("bag", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "step0",
    )
    parser.add_argument(
        "--artifact-prefix",
        default="step0",
    )
    args = parser.parse_args()
    figure_path, summary_path = analyze(
        args.bag,
        args.output_dir,
        args.artifact_prefix,
    )
    print(f"Wrote {figure_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
