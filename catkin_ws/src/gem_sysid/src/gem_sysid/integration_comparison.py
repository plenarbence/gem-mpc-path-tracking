from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


METHODS = ("euler", "midpoint_euler")
REQUIRED_COLUMNS = (
    "profile_name",
    "split",
    "sample_index",
    "anchor_time_s",
    "anchor_relative_s",
    "x_m",
    "y_m",
    "speed_longitudinal_mps",
    "yaw_unwrapped_rad",
    "yaw_rate_radps",
)
PREDICTION_COLUMNS = (
    "method",
    "sample_index",
    "anchor_start_s",
    "dt_s",
    "actual_x_next_m",
    "actual_y_next_m",
    "actual_yaw_next_rad",
    "predicted_x_next_m",
    "predicted_y_next_m",
    "predicted_yaw_next_rad",
    "error_x_m",
    "error_y_m",
    "xy_error_m",
    "yaw_error_rad",
)
ROLLOUT_COLUMNS = (
    "method",
    "sample_index",
    "anchor_relative_s",
    "actual_x_m",
    "actual_y_m",
    "actual_yaw_rad",
    "predicted_x_m",
    "predicted_y_m",
    "predicted_yaw_rad",
    "error_x_m",
    "error_y_m",
    "xy_error_m",
    "yaw_error_rad",
)
METRIC_COLUMNS = (
    "evaluation",
    "method",
    "target",
    "metric",
    "value",
    "sample_count",
)


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def euler_step(
    pose: np.ndarray,
    speed: float,
    yaw_rate: float,
    dt: float,
) -> np.ndarray:
    x, y, yaw = pose
    return np.asarray(
        [
            x + dt * speed * np.cos(yaw),
            y + dt * speed * np.sin(yaw),
            yaw + dt * yaw_rate,
        ],
        dtype=float,
    )


def midpoint_euler_step(
    pose: np.ndarray,
    speed_start: float,
    speed_end: float,
    yaw_rate_start: float,
    yaw_rate_end: float,
    dt: float,
) -> np.ndarray:
    speed_midpoint = 0.5 * (speed_start + speed_end)
    yaw_rate_midpoint = 0.5 * (yaw_rate_start + yaw_rate_end)
    yaw_midpoint = pose[2] + 0.5 * dt * yaw_rate_midpoint
    return np.asarray(
        [
            pose[0] + dt * speed_midpoint * np.cos(yaw_midpoint),
            pose[1] + dt * speed_midpoint * np.sin(yaw_midpoint),
            pose[2] + dt * yaw_rate_midpoint,
        ],
        dtype=float,
    )


def load_test_dataset(path: Path) -> tuple[dict[str, np.ndarray], str]:
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
                "Dataset is missing columns: " + ", ".join(missing)
            )
        rows = list(reader)

    if len(rows) < 2:
        raise RuntimeError("Test dataset needs at least two samples")
    splits = {row["split"] for row in rows}
    profiles = {row["profile_name"] for row in rows}
    if splits != {"test"}:
        raise RuntimeError(
            f"Integration comparison accepts only the test split, got {splits}"
        )
    if len(profiles) != 1:
        raise RuntimeError(
            "Integration comparison requires exactly one test profile"
        )

    sample_indices = np.asarray(
        [int(row["sample_index"]) for row in rows],
        dtype=int,
    )
    expected_indices = np.arange(len(rows), dtype=int)
    if not np.array_equal(sample_indices, expected_indices):
        raise RuntimeError("Test sample indices are not continuous")

    numeric_columns = REQUIRED_COLUMNS[3:]
    data = {
        column: np.asarray(
            [float(row[column]) for row in rows],
            dtype=float,
        )
        for column in numeric_columns
    }
    if any(not np.isfinite(values).all() for values in data.values()):
        raise RuntimeError("Test dataset contains non-finite values")
    if np.any(np.diff(data["anchor_time_s"]) <= 0.0):
        raise RuntimeError("Test anchor timestamps are not increasing")
    return data, profiles.pop()


def _prediction_error(
    predicted: np.ndarray,
    actual: np.ndarray,
) -> tuple[float, float, float, float]:
    error_x = float(predicted[0] - actual[0])
    error_y = float(predicted[1] - actual[1])
    xy_error = float(np.hypot(error_x, error_y))
    yaw_error = float(wrap_angle(predicted[2] - actual[2]))
    return error_x, error_y, xy_error, yaw_error


def build_one_step_predictions(
    data: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    count = len(data["x_m"])
    for index in range(count - 1):
        pose = np.asarray(
            [
                data["x_m"][index],
                data["y_m"][index],
                data["yaw_unwrapped_rad"][index],
            ]
        )
        actual = np.asarray(
            [
                data["x_m"][index + 1],
                data["y_m"][index + 1],
                data["yaw_unwrapped_rad"][index + 1],
            ]
        )
        dt = float(
            data["anchor_time_s"][index + 1]
            - data["anchor_time_s"][index]
        )
        method_predictions = {
            "euler": euler_step(
                pose,
                data["speed_longitudinal_mps"][index],
                data["yaw_rate_radps"][index],
                dt,
            ),
            "midpoint_euler": midpoint_euler_step(
                pose,
                data["speed_longitudinal_mps"][index],
                data["speed_longitudinal_mps"][index + 1],
                data["yaw_rate_radps"][index],
                data["yaw_rate_radps"][index + 1],
                dt,
            ),
        }
        for method, predicted in method_predictions.items():
            error_x, error_y, xy_error, yaw_error = _prediction_error(
                predicted,
                actual,
            )
            predictions.append(
                {
                    "method": method,
                    "sample_index": index,
                    "anchor_start_s": data["anchor_relative_s"][index],
                    "dt_s": dt,
                    "actual_x_next_m": actual[0],
                    "actual_y_next_m": actual[1],
                    "actual_yaw_next_rad": actual[2],
                    "predicted_x_next_m": predicted[0],
                    "predicted_y_next_m": predicted[1],
                    "predicted_yaw_next_rad": predicted[2],
                    "error_x_m": error_x,
                    "error_y_m": error_y,
                    "xy_error_m": xy_error,
                    "yaw_error_rad": yaw_error,
                }
            )
    return predictions


def build_rollouts(
    data: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    actual_pose = np.column_stack(
        (
            data["x_m"],
            data["y_m"],
            data["yaw_unwrapped_rad"],
        )
    )
    predicted_by_method = {
        method: np.zeros_like(actual_pose) for method in METHODS
    }
    for predicted in predicted_by_method.values():
        predicted[0] = actual_pose[0]

    for index in range(len(actual_pose) - 1):
        dt = float(
            data["anchor_time_s"][index + 1]
            - data["anchor_time_s"][index]
        )
        predicted_by_method["euler"][index + 1] = euler_step(
            predicted_by_method["euler"][index],
            data["speed_longitudinal_mps"][index],
            data["yaw_rate_radps"][index],
            dt,
        )
        predicted_by_method["midpoint_euler"][
            index + 1
        ] = midpoint_euler_step(
            predicted_by_method["midpoint_euler"][index],
            data["speed_longitudinal_mps"][index],
            data["speed_longitudinal_mps"][index + 1],
            data["yaw_rate_radps"][index],
            data["yaw_rate_radps"][index + 1],
            dt,
        )

    rows: list[dict[str, Any]] = []
    for method, predicted_sequence in predicted_by_method.items():
        for index, predicted in enumerate(predicted_sequence):
            actual = actual_pose[index]
            error_x, error_y, xy_error, yaw_error = _prediction_error(
                predicted,
                actual,
            )
            rows.append(
                {
                    "method": method,
                    "sample_index": index,
                    "anchor_relative_s": data["anchor_relative_s"][index],
                    "actual_x_m": actual[0],
                    "actual_y_m": actual[1],
                    "actual_yaw_rad": actual[2],
                    "predicted_x_m": predicted[0],
                    "predicted_y_m": predicted[1],
                    "predicted_yaw_rad": predicted[2],
                    "error_x_m": error_x,
                    "error_y_m": error_y,
                    "xy_error_m": xy_error,
                    "yaw_error_rad": yaw_error,
                }
            )
    return rows


def _error_metrics(values: np.ndarray) -> dict[str, float]:
    absolute = np.abs(values)
    return {
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
        "mae": float(np.mean(absolute)),
        "p95_abs": float(np.percentile(absolute, 95)),
        "max_abs": float(np.max(absolute)),
        "final_abs": float(absolute[-1]),
    }


def build_metrics(
    one_step_rows: list[dict[str, Any]],
    rollout_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metric_rows: list[dict[str, Any]] = []
    nested: dict[str, Any] = {}
    for evaluation, rows in (
        ("one_step", one_step_rows),
        ("rollout", rollout_rows),
    ):
        nested[evaluation] = {}
        for method in METHODS:
            method_rows = [row for row in rows if row["method"] == method]
            nested[evaluation][method] = {}
            for target in ("xy_error_m", "yaw_error_rad"):
                values = np.asarray(
                    [row[target] for row in method_rows],
                    dtype=float,
                )
                metrics = _error_metrics(values)
                nested[evaluation][method][target] = metrics
                for metric, value in metrics.items():
                    metric_rows.append(
                        {
                            "evaluation": evaluation,
                            "method": method,
                            "target": target,
                            "metric": metric,
                            "value": value,
                            "sample_count": len(values),
                        }
                    )
    return metric_rows, nested


def write_csv(
    rows: list[dict[str, Any]],
    columns: tuple[str, ...],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_plot(
    data: dict[str, np.ndarray],
    one_step_rows: list[dict[str, Any]],
    rollout_rows: list[dict[str, Any]],
    path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(data["x_m"], data["y_m"], label="Measured", linewidth=2)

    for method in METHODS:
        rollout = [row for row in rollout_rows if row["method"] == method]
        axes[0, 0].plot(
            [row["predicted_x_m"] for row in rollout],
            [row["predicted_y_m"] for row in rollout],
            label=method.replace("_", " ").title(),
        )
    axes[0, 0].set_title("Full Test Trajectory")
    axes[0, 0].set_xlabel("x [m]")
    axes[0, 0].set_ylabel("y [m]")
    axes[0, 0].axis("equal")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    for method in METHODS:
        one_step = [
            row for row in one_step_rows if row["method"] == method
        ]
        axes[0, 1].plot(
            [row["anchor_start_s"] for row in one_step],
            [row["xy_error_m"] for row in one_step],
            label=method.replace("_", " ").title(),
        )
    axes[0, 1].set_title("One-Step XY Error")
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("Euclidean error [m]")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    for method in METHODS:
        rollout = [row for row in rollout_rows if row["method"] == method]
        axes[1, 0].plot(
            [row["anchor_relative_s"] for row in rollout],
            [row["xy_error_m"] for row in rollout],
            label=method.replace("_", " ").title(),
        )
    axes[1, 0].set_title("Accumulated XY Error")
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel("Euclidean error [m]")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    for method in METHODS:
        one_step = [
            row for row in one_step_rows if row["method"] == method
        ]
        rollout = [row for row in rollout_rows if row["method"] == method]
        label = method.replace("_", " ").title()
        axes[1, 1].plot(
            [row["anchor_start_s"] for row in one_step],
            np.rad2deg([row["yaw_error_rad"] for row in one_step]),
            linestyle="--",
            alpha=0.8,
            label=f"{label}, one-step",
        )
        axes[1, 1].plot(
            [row["anchor_relative_s"] for row in rollout],
            np.rad2deg([row["yaw_error_rad"] for row in rollout]),
            label=f"{label}, rollout",
        )
    axes[1, 1].set_title("Wrapped Yaw Error")
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_ylabel("Error [deg]")
    axes[1, 1].grid(True)
    axes[1, 1].legend(ncol=2)

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def compare_test_dataset(
    dataset_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    data, profile_name = load_test_dataset(dataset_path)
    one_step_rows = build_one_step_predictions(data)
    rollout_rows = build_rollouts(data)
    metric_rows, metrics = build_metrics(one_step_rows, rollout_rows)

    predictions_path = output_dir / "one_step_predictions.csv"
    rollout_path = output_dir / "full_rollout.csv"
    metrics_path = output_dir / "metrics.csv"
    plot_path = output_dir / "integration_comparison.png"
    summary_path = output_dir / "summary.json"
    write_csv(one_step_rows, PREDICTION_COLUMNS, predictions_path)
    write_csv(rollout_rows, ROLLOUT_COLUMNS, rollout_path)
    write_csv(metric_rows, METRIC_COLUMNS, metrics_path)
    write_plot(data, one_step_rows, rollout_rows, plot_path)

    dt = np.diff(data["anchor_time_s"])
    summary = {
        "dataset": str(dataset_path),
        "profile_name": profile_name,
        "split": "test",
        "sample_count": len(data["x_m"]),
        "transition_count": len(data["x_m"]) - 1,
        "dt_source": "difference between consecutive synchronization anchors",
        "dt_ms": {
            "mean": float(1000.0 * np.mean(dt)),
            "min": float(1000.0 * np.min(dt)),
            "max": float(1000.0 * np.max(dt)),
        },
        "methods": {
            "euler": (
                "Uses v_k and omega_k at the beginning of each interval."
            ),
            "midpoint_euler": (
                "Uses endpoint-averaged v and omega and evaluates position "
                "at the resulting midpoint heading. This is an offline, "
                "noncausal integration baseline because it uses k+1 values."
            ),
        },
        "errors": {
            "xy": "Euclidean position error",
            "yaw": "predicted minus measured yaw, wrapped to [-pi, pi)",
        },
        "metrics": metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="ascii",
    )
    return metrics_path, plot_path
