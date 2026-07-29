from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import rosbag


STAMPED_COMMAND_TOPIC = "/gem_sysid/ackermann_cmd_stamped"
PROFILE_START_TOPIC = "/gem_sysid/profile_start"
COMMISSIONING_DELAY_TOPIC = "/gem_sysid/commissioning_delay_ms"
ODOM_TOPIC = "/gem/base_footprint/odom"

SAMPLE_PERIOD_S = 0.1
COMMAND_TIME_TOLERANCE_S = 0.005

DATASET_COLUMNS = (
    "profile_name",
    "split",
    "sample_index",
    "nominal_time_s",
    "anchor_time_s",
    "anchor_relative_s",
    "actual_dt_s",
    "command_publish_time_s",
    "commissioned_delay_ms",
    "speed_command_mps",
    "steering_command_rad",
    "x_m",
    "y_m",
    "vx_world_mps",
    "vy_world_mps",
    "speed_longitudinal_mps",
    "yaw_rad",
    "yaw_unwrapped_rad",
    "yaw_rate_radps",
)


def message_time(message: Any, bag_time: Any) -> float:
    if hasattr(message, "header") and not message.header.stamp.is_zero():
        return message.header.stamp.to_sec()
    if hasattr(message, "stamp") and not message.stamp.is_zero():
        return message.stamp.to_sec()
    return bag_time.to_sec()


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


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def extract_profile_commands(
    commands: list[tuple[float, float, float]],
    start_time: float,
    sample_period_s: float = SAMPLE_PERIOD_S,
    tolerance_s: float = COMMAND_TIME_TOLERANCE_S,
) -> list[tuple[float, float, float]]:
    profile: list[tuple[float, float, float]] = []
    expected_time = start_time
    for command in commands:
        command_time = command[0]
        if abs(command_time - expected_time) <= tolerance_s:
            profile.append(command)
            expected_time += sample_period_s
        elif command_time > expected_time + tolerance_s:
            break
    return profile


def _strictly_increasing_series(
    times: list[float],
    values: list[list[float]],
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    if not times:
        raise RuntimeError(f"No {name} samples were recorded")

    time_array = np.asarray(times, dtype=float)
    value_array = np.asarray(values, dtype=float)
    order = np.argsort(time_array, kind="stable")
    time_array = time_array[order]
    value_array = value_array[order]

    # Keep the final value if a source publishes twice with the same timestamp.
    keep = np.concatenate((np.diff(time_array) > 1e-12, [True]))
    time_array = time_array[keep]
    value_array = value_array[keep]
    if time_array.size < 2 or np.any(np.diff(time_array) <= 0.0):
        raise RuntimeError(f"{name} timestamps are not strictly increasing")
    return time_array, value_array


def interpolate_series(
    source_times: np.ndarray,
    source_values: np.ndarray,
    query_times: np.ndarray,
    name: str,
) -> tuple[np.ndarray, dict[str, float | int]]:
    if query_times.size == 0:
        raise RuntimeError("No synchronization anchors were provided")
    if (
        query_times[0] < source_times[0] - 1e-9
        or query_times[-1] > source_times[-1] + 1e-9
    ):
        raise RuntimeError(
            f"{name} does not bracket every synchronization anchor: "
            f"source=[{source_times[0]:.9f}, {source_times[-1]:.9f}], "
            f"anchors=[{query_times[0]:.9f}, {query_times[-1]:.9f}]"
        )

    right = np.searchsorted(source_times, query_times, side="left")
    right = np.minimum(right, source_times.size - 1)
    exact = np.isclose(
        source_times[right],
        query_times,
        rtol=0.0,
        atol=1e-9,
    )
    left = np.where(exact, right, right - 1)
    if np.any(left < 0):
        raise RuntimeError(f"{name} cannot interpolate the first anchor")

    interpolated = np.column_stack(
        [
            np.interp(query_times, source_times, source_values[:, column])
            for column in range(source_values.shape[1])
        ]
    )
    previous_distance = query_times - source_times[left]
    next_distance = source_times[right] - query_times
    bracket_span = source_times[right] - source_times[left]
    nearest_distance = np.minimum(previous_distance, next_distance)
    quality: dict[str, float | int] = {
        "source_sample_count": int(source_times.size),
        "max_bracket_span_ms": float(1000.0 * np.max(bracket_span)),
        "mean_bracket_span_ms": float(1000.0 * np.mean(bracket_span)),
        "max_nearest_sample_distance_ms": float(
            1000.0 * np.max(nearest_distance)
        ),
    }
    return interpolated, quality


def read_profile_bag(
    bag_path: Path,
) -> tuple[
    list[tuple[float, float, float]],
    float,
    float,
    tuple[np.ndarray, np.ndarray],
]:
    commands: list[tuple[float, float, float]] = []
    profile_starts: list[float] = []
    commissioning_delays_ms: list[float] = []
    odom_times: list[float] = []
    odom_values: list[list[float]] = []

    topics = (
        STAMPED_COMMAND_TOPIC,
        PROFILE_START_TOPIC,
        COMMISSIONING_DELAY_TOPIC,
        ODOM_TOPIC,
    )
    with rosbag.Bag(str(bag_path), "r") as bag:
        for topic, message, bag_time in bag.read_messages(topics=list(topics)):
            timestamp = message_time(message, bag_time)
            if topic == STAMPED_COMMAND_TOPIC:
                commands.append(
                    (
                        timestamp,
                        float(message.drive.speed),
                        float(message.drive.steering_angle),
                    )
                )
            elif topic == PROFILE_START_TOPIC:
                profile_starts.append(message.stamp.to_sec())
            elif topic == COMMISSIONING_DELAY_TOPIC:
                commissioning_delays_ms.append(float(message.data))
            elif topic == ODOM_TOPIC:
                pose = message.pose.pose
                twist = message.twist.twist
                odom_times.append(timestamp)
                odom_values.append(
                    [
                        float(pose.position.x),
                        float(pose.position.y),
                        float(twist.linear.x),
                        float(twist.linear.y),
                        yaw_from_quaternion(pose.orientation),
                        float(twist.angular.z),
                    ]
                )

    if not profile_starts:
        raise RuntimeError(f"{bag_path} has no {PROFILE_START_TOPIC} marker")
    if not commissioning_delays_ms:
        raise RuntimeError(
            f"{bag_path} has no {COMMISSIONING_DELAY_TOPIC} value"
        )

    start_time = profile_starts[-1]
    profile_commands = extract_profile_commands(commands, start_time)
    if not profile_commands:
        raise RuntimeError(f"{bag_path} has no commands after profile start")

    odom_time_array, odom_value_array = _strictly_increasing_series(
        odom_times,
        odom_values,
        "odometry",
    )
    odom_value_array[:, 4] = np.unwrap(odom_value_array[:, 4])
    return (
        profile_commands,
        start_time,
        commissioning_delays_ms[-1],
        (odom_time_array, odom_value_array),
    )


def prepare_profile(
    suite_row: dict[str, str],
    output_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], Path, Path]:
    profile_name = suite_row["profile_name"]
    split = suite_row["split"]
    bag_path = Path(suite_row["bag_path"])
    (
        commands,
        profile_start_time,
        commissioning_delay_ms,
        odom_series,
    ) = read_profile_bag(bag_path)

    expected_count = int(suite_row["sample_count"])
    if len(commands) != expected_count:
        raise RuntimeError(
            f"{profile_name} contains {len(commands)} profile commands; "
            f"expected {expected_count}"
        )

    command_times = np.asarray([command[0] for command in commands])
    speed_commands = np.asarray([command[1] for command in commands])
    steering_commands = np.asarray([command[2] for command in commands])
    anchor_times = command_times + commissioning_delay_ms / 1000.0

    odom_interpolated, odom_quality = interpolate_series(
        odom_series[0],
        odom_series[1],
        anchor_times,
        "odometry",
    )

    yaw_unwrapped = odom_interpolated[:, 4]
    yaw = wrap_angle(yaw_unwrapped)
    longitudinal_speed = (
        np.cos(yaw) * odom_interpolated[:, 2]
        + np.sin(yaw) * odom_interpolated[:, 3]
    )
    actual_dt = np.concatenate(([0.0], np.diff(anchor_times)))
    anchor_relative = anchor_times - anchor_times[0]

    rows: list[dict[str, Any]] = []
    for index in range(len(commands)):
        rows.append(
            {
                "profile_name": profile_name,
                "split": split,
                "sample_index": index,
                "nominal_time_s": index * SAMPLE_PERIOD_S,
                "anchor_time_s": anchor_times[index],
                "anchor_relative_s": anchor_relative[index],
                "actual_dt_s": actual_dt[index],
                "command_publish_time_s": command_times[index],
                "commissioned_delay_ms": commissioning_delay_ms,
                "speed_command_mps": speed_commands[index],
                "steering_command_rad": steering_commands[index],
                "x_m": odom_interpolated[index, 0],
                "y_m": odom_interpolated[index, 1],
                "vx_world_mps": odom_interpolated[index, 2],
                "vy_world_mps": odom_interpolated[index, 3],
                "speed_longitudinal_mps": longitudinal_speed[index],
                "yaw_rad": yaw[index],
                "yaw_unwrapped_rad": yaw_unwrapped[index],
                "yaw_rate_radps": odom_interpolated[index, 5],
            }
        )

    split_dir = output_root / split
    csv_path = split_dir / f"{profile_name}.csv"
    metadata_path = split_dir / f"{profile_name}.metadata.json"
    write_rows(rows, csv_path)

    intervals = np.diff(anchor_times)
    metadata = {
        "profile_name": profile_name,
        "split": split,
        "source_bag": str(bag_path),
        "profile_start_time_s": profile_start_time,
        "sample_count": len(rows),
        "nominal_sample_period_s": SAMPLE_PERIOD_S,
        "commissioned_delay_ms": commissioning_delay_ms,
        "anchor_definition": (
            "AckermannDriveStamped header timestamp plus the one-time "
            "commissioned delay for this run"
        ),
        "actual_anchor_period_ms": {
            "mean": float(1000.0 * np.mean(intervals)),
            "min": float(1000.0 * np.min(intervals)),
            "max": float(1000.0 * np.max(intervals)),
        },
        "interpolation": {
            "method": "linear",
            "yaw_handling": (
                "Quaternion yaw is unwrapped before interpolation; both "
                "wrapped and unwrapped results are stored."
            ),
            "yaw_rate_source": "Odometry twist.twist.angular.z",
            "odometry": odom_quality,
        },
        "columns": list(DATASET_COLUMNS),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="ascii",
    )
    return rows, metadata, csv_path, metadata_path


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=DATASET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def prepare_suite(
    suite_summary_path: Path,
    output_root: Path,
) -> tuple[Path, Path]:
    with suite_summary_path.open(newline="", encoding="ascii") as stream:
        suite_rows = list(csv.DictReader(stream))
    if not suite_rows:
        raise RuntimeError(f"No profiles found in {suite_summary_path}")

    split_rows: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    manifest_rows: list[dict[str, Any]] = []
    profile_metadata: list[dict[str, Any]] = []

    for suite_row in suite_rows:
        profile_name = suite_row["profile_name"]
        split = suite_row["split"]
        print(f"[{split}] {profile_name}: synchronizing", flush=True)
        rows, metadata, csv_path, metadata_path = prepare_profile(
            suite_row,
            output_root,
        )
        split_rows[split].extend(rows)
        profile_metadata.append(metadata)
        manifest_rows.append(
            {
                "profile_name": profile_name,
                "split": split,
                "sample_count": len(rows),
                "commissioned_delay_ms": metadata[
                    "commissioned_delay_ms"
                ],
                "source_bag": metadata["source_bag"],
                "dataset_csv": csv_path.relative_to(output_root).as_posix(),
                "metadata_json": metadata_path.relative_to(
                    output_root
                ).as_posix(),
            }
        )

    for split, rows in split_rows.items():
        if rows:
            write_rows(rows, output_root / f"{split}.csv")

    manifest_path = output_root / "dataset_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=manifest_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "source_suite_summary": str(suite_summary_path),
        "anchor_definition": (
            "command_publish_time_s + commissioned_delay_ms / 1000"
        ),
        "nominal_sample_period_s": SAMPLE_PERIOD_S,
        "interpolation_method": "linear using odometry header timestamps",
        "yaw_rate_source": (
            "/gem/base_footprint/odom twist.twist.angular.z"
        ),
        "split_sample_counts": {
            split: len(rows) for split, rows in split_rows.items()
        },
        "total_sample_count": sum(
            len(rows) for rows in split_rows.values()
        ),
        "columns": list(DATASET_COLUMNS),
        "profiles": profile_metadata,
    }
    summary_path = output_root / "dataset_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="ascii",
    )
    return manifest_path, summary_path
