#!/usr/bin/env python3

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from gem_control.casadi_reference_path import CasadiReferencePath
from gem_control.reference_path import (
    build_configured_reference_path,
    load_waypoint_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate the configured smoothed GEM reference path."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional reference_path.yaml override.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/reference_path"),
        help="Directory for the validation evidence.",
    )
    return parser.parse_args()


def project_waypoints(path, waypoint_values):
    rows = []
    progress_hint = None
    for index, (x, y) in enumerate(waypoint_values):
        projection = path.project(
            float(x),
            float(y),
            s_hint=progress_hint,
        )
        progress_hint = projection.s
        rows.append(
            [
                index,
                x,
                y,
                projection.s,
                projection.s_wrapped,
                projection.x,
                projection.y,
                projection.distance,
                projection.signed_lateral_error,
            ]
        )
    return np.asarray(rows, dtype=float)


def error_metrics(values):
    errors = np.asarray(values, dtype=float)
    return {
        "sample_count": len(errors),
        "mean_error_m": float(np.mean(errors)),
        "rmse_error_m": float(np.sqrt(np.mean(errors ** 2))),
        "p95_error_m": float(np.quantile(errors, 0.95)),
        "maximum_error_m": float(np.max(errors)),
    }


def write_projection_csv(path, projection_rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "csv_row",
                "waypoint_x",
                "waypoint_y",
                "projected_s",
                "projected_s_wrapped",
                "projected_x",
                "projected_y",
                "euclidean_error_m",
                "signed_lateral_error_m",
            ]
        )
        writer.writerows(projection_rows)


def plot_waypoint_difference(
    output_path,
    path,
    evaluation,
    waypoint_values,
    projection_rows,
    preprocessing,
):
    row_indices = projection_rows[:, 0].astype(int)
    lap_mask = (
        (row_indices >= preprocessing.first_lap_raw_index)
        & (row_indices <= preprocessing.last_lap_raw_index)
    )
    lap_rows = projection_rows[lap_mask]
    lap_errors = lap_rows[:, 7]
    maximum_lap_row = lap_rows[int(np.argmax(lap_errors))]
    maximum_csv_row = int(maximum_lap_row[0])

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].scatter(
        waypoint_values[lap_mask, 0],
        waypoint_values[lap_mask, 1],
        color="#8a8f98",
        s=5,
        alpha=0.75,
        label="CSV waypoints",
    )
    axes[0, 0].plot(
        evaluation.x,
        evaluation.y,
        color="#006d77",
        linewidth=1.5,
        label="Smoothed path",
    )
    axes[0, 0].axis("equal")
    axes[0, 0].set_xlabel("x [m]")
    axes[0, 0].set_ylabel("y [m]")
    axes[0, 0].set_title("One-lap geometry comparison")
    axes[0, 0].legend()

    window_start = max(0, maximum_csv_row - 4)
    window_end = min(len(waypoint_values), maximum_csv_row + 5)
    zoom_rows = projection_rows[window_start:window_end]
    zoom_progress = np.linspace(
        maximum_lap_row[4] - 1.25,
        maximum_lap_row[4] + 1.25,
        500,
    )
    zoom_path = path.evaluate(zoom_progress)
    axes[0, 1].plot(
        zoom_path.x,
        zoom_path.y,
        color="#006d77",
        linewidth=1.8,
        label="Smoothed path",
    )
    axes[0, 1].scatter(
        zoom_rows[:, 1],
        zoom_rows[:, 2],
        color="#8a8f98",
        s=16,
        label="CSV waypoints",
        zorder=3,
    )
    for row in zoom_rows:
        axes[0, 1].plot(
            [row[1], row[5]],
            [row[2], row[6]],
            color="#c44536",
            linewidth=0.8,
            alpha=0.8,
        )
    axes[0, 1].plot(
        [maximum_lap_row[1], maximum_lap_row[5]],
        [maximum_lap_row[2], maximum_lap_row[6]],
        color="#c44536",
        linewidth=2.0,
        label="Point-to-path difference",
    )
    visible_x = np.concatenate(
        [zoom_rows[:, 1], zoom_rows[:, 5], zoom_path.x]
    )
    visible_y = np.concatenate(
        [zoom_rows[:, 2], zoom_rows[:, 6], zoom_path.y]
    )
    data_span = max(
        float(np.ptp(visible_x)),
        float(np.ptp(visible_y)),
        0.5,
    )
    center_x = 0.5 * (float(np.min(visible_x)) + float(np.max(visible_x)))
    center_y = 0.5 * (float(np.min(visible_y)) + float(np.max(visible_y)))
    margin_span = 1.12 * data_span
    axes[0, 1].set_xlim(
        center_x - 0.5 * margin_span,
        center_x + 0.5 * margin_span,
    )
    axes[0, 1].set_ylim(
        center_y - 0.5 * margin_span,
        center_y + 0.5 * margin_span,
    )
    axes[0, 1].set_aspect("equal", adjustable="box")
    axes[0, 1].set_xlabel("x [m]")
    axes[0, 1].set_ylabel("y [m]")
    axes[0, 1].set_title(
        "Closure-region maximum deviation: {:.3f} m".format(
            maximum_lap_row[7]
        )
    )
    axes[0, 1].legend()

    axes[1, 0].plot(
        row_indices,
        projection_rows[:, 7],
        color="#aeb3ba",
        linewidth=0.8,
        label="Outside selected lap",
    )
    axes[1, 0].plot(
        lap_rows[:, 0],
        lap_errors,
        color="#c44536",
        linewidth=1.0,
        label="Selected lap",
    )
    axes[1, 0].axhline(
        np.mean(lap_errors),
        color="#006d77",
        linestyle="--",
        linewidth=1.2,
        label="Lap mean",
    )
    axes[1, 0].set_xlabel("Original CSV row")
    axes[1, 0].set_ylabel("Euclidean difference [m]")
    axes[1, 0].set_title("Waypoint-to-path distance")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()

    axes[1, 1].scatter(
        lap_rows[:, 4],
        lap_rows[:, 8],
        c=np.abs(lap_rows[:, 8]),
        cmap="coolwarm",
        s=7,
        alpha=0.85,
    )
    axes[1, 1].axhline(0.0, color="#36393f", linewidth=0.8)
    axes[1, 1].set_xlabel("Projected progress s [m]")
    axes[1, 1].set_ylabel("Signed lateral difference [m]")
    axes[1, 1].set_title("Signed difference along the lap")
    axes[1, 1].grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    path, preprocessing, settings = build_configured_reference_path(
        args.config
    )
    waypoint_values = load_waypoint_csv(preprocessing.source_csv)
    progress = np.linspace(0.0, path.length, 5001)
    evaluation = path.evaluate(progress)
    seam = path.seam_diagnostics()

    casadi_path = CasadiReferencePath(
        path,
        sample_count=settings.casadi_sample_count,
    )
    parity_progress = np.linspace(
        0.0,
        path.length,
        1200,
        endpoint=False,
    )
    parity = casadi_path.parity_check(parity_progress)
    waypoint_projections = project_waypoints(path, waypoint_values)
    write_projection_csv(
        output_dir / "waypoint_projection_errors.csv",
        waypoint_projections,
    )
    row_indices = waypoint_projections[:, 0].astype(int)
    lap_mask = (
        (row_indices >= preprocessing.first_lap_raw_index)
        & (row_indices <= preprocessing.last_lap_raw_index)
    )
    lap_projection_rows = waypoint_projections[lap_mask]
    maximum_lap_row = lap_projection_rows[
        int(np.argmax(lap_projection_rows[:, 7]))
    ]
    waypoint_difference = {
        "all_original_csv_rows": error_metrics(waypoint_projections[:, 7]),
        "selected_one_lap_row_range": error_metrics(
            lap_projection_rows[:, 7]
        ),
        "maximum_one_lap_error_csv_row": int(maximum_lap_row[0]),
        "maximum_one_lap_error_projected_s_m": float(maximum_lap_row[4]),
    }
    validation_limits = {
        "seam_position_gap_m": 0.01,
        "seam_tangent_angle_gap_rad": 0.001,
        "seam_curvature_gap_1pm": 0.001,
        "casadi_position_difference_m": 0.0001,
        "casadi_yaw_difference_rad": 0.001,
        "casadi_curvature_difference_1pm": 0.001,
    }
    checks = {
        "seam_position": (
            seam.seam_position_gap_m
            <= validation_limits["seam_position_gap_m"]
        ),
        "seam_tangent": (
            seam.seam_tangent_angle_gap_rad
            <= validation_limits["seam_tangent_angle_gap_rad"]
        ),
        "seam_curvature": (
            seam.seam_curvature_gap_1pm
            <= validation_limits["seam_curvature_gap_1pm"]
        ),
        "casadi_position": (
            parity.maximum_position_difference_m
            <= validation_limits["casadi_position_difference_m"]
        ),
        "casadi_yaw": (
            parity.maximum_yaw_difference_rad
            <= validation_limits["casadi_yaw_difference_rad"]
        ),
        "casadi_curvature": (
            parity.maximum_curvature_difference_1pm
            <= validation_limits["casadi_curvature_difference_1pm"]
        ),
    }

    preprocessing_summary = asdict(preprocessing)
    preprocessing_summary["source_csv"] = "package://{}/{}".format(
        settings.waypoint_package,
        settings.waypoint_relative_path,
    )
    summary = {
        "path_length_m": path.length,
        "smoothing_factor": path.smoothing_factor,
        "preprocessing": preprocessing_summary,
        "seam": asdict(seam),
        "casadi_parity": asdict(parity),
        "waypoint_difference": waypoint_difference,
        "validation_limits": validation_limits,
        "checks": checks,
        "passed": all(checks.values()),
        "sample_count": len(progress),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")

    with (output_dir / "reference_path.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["s", "x", "y", "yaw", "curvature"])
        writer.writerows(
            zip(
                progress,
                evaluation.x,
                evaluation.y,
                evaluation.yaw,
                evaluation.curvature,
            )
        )

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(
        waypoint_values[:, 0],
        waypoint_values[:, 1],
        color="#8a8f98",
        linewidth=0.8,
        label="Raw waypoints",
    )
    axes[0].plot(
        evaluation.x,
        evaluation.y,
        color="#006d77",
        linewidth=1.6,
        label="Smoothed path",
    )
    axes[0].axis("equal")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].set_title("Reference geometry")
    axes[0].legend()

    axes[1].plot(
        progress,
        np.unwrap(evaluation.yaw),
        color="#3a506b",
    )
    axes[1].set_xlabel("Progress s [m]")
    axes[1].set_ylabel("Unwrapped yaw [rad]")
    axes[1].set_title("Continuous path heading")
    axes[1].grid(alpha=0.25)

    axes[2].plot(progress, evaluation.curvature, color="#c44536")
    axes[2].set_xlabel("Progress s [m]")
    axes[2].set_ylabel("Curvature [1/m]")
    axes[2].set_title("Path curvature")
    axes[2].grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_dir / "reference_path_validation.png", dpi=160)
    plt.close(figure)
    plot_waypoint_difference(
        output_dir / "waypoint_smoothing_difference.png",
        path,
        evaluation,
        waypoint_values,
        waypoint_projections,
        preprocessing,
    )

    print("Reference path validation complete")
    print("  length: {:.6f} m".format(path.length))
    print("  smoothing factor: {:.3f}".format(path.smoothing_factor))
    print(
        "  CasADi max position difference: {:.3e} m".format(
            parity.maximum_position_difference_m
        )
    )
    print("  checks passed: {}".format(all(checks.values())))
    print(
        "  waypoint difference mean/p95/max: "
        "{:.4f}/{:.4f}/{:.4f} m".format(
            waypoint_difference["selected_one_lap_row_range"]["mean_error_m"],
            waypoint_difference["selected_one_lap_row_range"]["p95_error_m"],
            waypoint_difference["selected_one_lap_row_range"]["maximum_error_m"],
        )
    )
    print("  output: {}".format(output_dir))
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(
            "reference path validation failed: {}".format(", ".join(failed))
        )


if __name__ == "__main__":
    main()
