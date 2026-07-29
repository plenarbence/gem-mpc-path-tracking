#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONTROLLERS = (
    ("full_learned_mpc", "Full learned MPC"),
    ("cascaded_p", "Cascaded P"),
)
COLORS = {
    "full_learned_mpc": "#0072B2",
    "cascaded_p": "#009E73",
}
SETTLED_LATERAL_ERROR_M = 0.10
SETTLED_YAW_ERROR_RAD = math.radians(5.0)
SETTLED_DURATION_S = 1.0


def load_columns(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"control log is empty: {path}")

    numeric = {}
    for name in rows[0]:
        try:
            numeric[name] = np.asarray(
                [float(row[name]) for row in rows], dtype=float
            )
        except ValueError:
            continue
    return numeric


def find_settled_index(
    elapsed_s: np.ndarray,
    lateral_error_m: np.ndarray,
    yaw_error_rad: np.ndarray,
) -> int | None:
    within = (
        (np.abs(lateral_error_m) <= SETTLED_LATERAL_ERROR_M)
        & (np.abs(yaw_error_rad) <= SETTLED_YAW_ERROR_RAD)
    )
    run_start = None
    for index, is_within in enumerate(within):
        if is_within and run_start is None:
            run_start = index
        elif not is_within:
            run_start = None
        if (
            run_start is not None
            and elapsed_s[index] - elapsed_s[run_start]
            >= SETTLED_DURATION_S
        ):
            return run_start
    return None


def smoothed_progression_rate(
    progress_m: np.ndarray, publish_time_s: np.ndarray
) -> np.ndarray:
    raw_rate = np.gradient(progress_m, publish_time_s)
    window_size = 5
    padding = window_size // 2
    padded = np.pad(raw_rate, (padding, padding), mode="edge")
    return np.convolve(
        padded, np.full(window_size, 1.0 / window_size), mode="valid"
    )


def controller_metrics(
    controller: str,
    label: str,
    run_directory: Path,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    columns = load_columns(run_directory / "control_log.csv")
    run_summary = json.loads(
        (run_directory / "summary.json").read_text(encoding="ascii")
    )
    elapsed = columns["elapsed_s"]
    progress = columns["progress_m"] - columns["progress_m"][0]
    progression_rate = smoothed_progression_rate(
        columns["progress_m"], columns["publish_time_s"]
    )
    reference_progression = columns["reference_speed_mps"]
    lateral = columns["lateral_error_m"]
    yaw = columns["yaw_error_rad"]
    settled_index = find_settled_index(elapsed, lateral, yaw)
    compute_column = (
        "controller_compute_s"
        if controller == "cascaded_p"
        else "mpc_total_compute_s"
    )
    compute_ms = 1000.0 * columns[compute_column]

    metrics = {
        "label": label,
        "termination_reason": run_summary["termination_reason"],
        "completed_50_m": run_summary["termination_reason"]
        == "completed_laps",
        "elapsed_s": float(elapsed[-1]),
        "progress_travelled_m": float(progress[-1]),
        "maximum_progress_m": float(np.max(progress)),
        "initial_lateral_error_m": float(lateral[0]),
        "initial_yaw_error_deg": float(np.degrees(yaw[0])),
        "lateral_error": {
            "rms_m": float(np.sqrt(np.mean(np.square(lateral)))),
            "maximum_absolute_m": float(np.max(np.abs(lateral))),
            "outside_one_m_sample_fraction": float(
                np.mean(np.abs(lateral) > 1.0)
            ),
        },
        "yaw_error": {
            "rms_deg": float(np.degrees(np.sqrt(np.mean(np.square(yaw))))),
            "maximum_absolute_deg": float(np.max(np.abs(np.degrees(yaw)))),
        },
        "settled_recovery": None
        if settled_index is None
        else {
            "time_s": float(elapsed[settled_index]),
            "distance_m": float(progress[settled_index]),
        },
        "maximum_speed_kmh": float(3.6 * np.max(columns["speed_mps"])),
        "path_progression": {
            "maximum_rate_kmh": float(3.6 * np.max(progression_rate)),
            "tracking_rmse_mps": float(
                np.sqrt(
                    np.mean(
                        np.square(
                            progression_rate - reference_progression
                        )
                    )
                )
            ),
        },
        "compute_time_ms": {
            "mean": float(np.mean(compute_ms)),
            "p95": float(np.quantile(compute_ms, 0.95)),
            "maximum": float(np.max(compute_ms)),
            "deadline_miss_count": int(
                run_summary.get("deadline_miss_count", 0)
            ),
        },
    }
    return metrics, {
        "elapsed_s": elapsed,
        "progress_m": progress,
        "lateral_error_m": lateral,
        "yaw_error_rad": yaw,
        "progression_rate_mps": progression_rate,
        "reference_progression_mps": reference_progression,
    }


def analyze(root_directory: Path) -> dict[str, object]:
    controller_results = {}
    plot_data = {}
    for controller, label in CONTROLLERS:
        metrics, columns = controller_metrics(
            controller, label, root_directory / controller
        )
        controller_results[controller] = metrics
        plot_data[controller] = columns

    summary = {
        "scenario": {
            "reference_path_progression_kmh": 10.0,
            "target_distance_m": 50.0,
            "requested_initial_lateral_error_m": 0.5,
            "requested_initial_yaw_error_deg": 40.0,
            "progression_rate_estimate": (
                "Five-sample moving average of the projected path-progress "
                "derivative"
            ),
            "settled_recovery_definition": (
                "First continuous 1.0 s with |lateral error| <= 0.10 m "
                "and |yaw error| <= 5 deg"
            ),
        },
        "controllers": controller_results,
    }
    (root_directory / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="ascii"
    )

    figure, axes = plt.subplots(3, 1, figsize=(10.5, 9.0), sharex=True)
    for controller, label in CONTROLLERS:
        columns = plot_data[controller]
        suffix = (
            ""
            if controller_results[controller]["completed_50_m"]
            else " (incomplete)"
        )
        style = {
            "color": COLORS[controller],
            "linewidth": 1.8,
            "label": label + suffix,
        }
        axes[0].plot(
            columns["progress_m"], columns["lateral_error_m"], **style
        )
        axes[1].plot(
            columns["progress_m"],
            np.degrees(columns["yaw_error_rad"]),
            **style,
        )
        axes[2].plot(
            columns["progress_m"],
            3.6 * columns["progression_rate_mps"],
            **style,
        )
        axes[2].plot(
            columns["progress_m"],
            3.6 * columns["reference_progression_mps"],
            color=COLORS[controller],
            linewidth=1.1,
            linestyle="--",
            alpha=0.75,
            label=label + " requested",
        )

    axes[0].axhline(1.0, color="#555555", linestyle="--", linewidth=1.0)
    axes[0].axhline(-1.0, color="#555555", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("lateral error [m]")
    axes[0].set_title(
        "0.5 m / 40 deg recovery at 10 km/h requested path progression"
    )
    axes[0].legend(loc="best")
    axes[1].axhline(5.0, color="#777777", linestyle=":", linewidth=1.0)
    axes[1].axhline(-5.0, color="#777777", linestyle=":", linewidth=1.0)
    axes[1].set_ylabel("yaw error [deg]")
    axes[2].set_ylabel("path progression rate [km/h]")
    axes[2].set_xlabel("path progress from initial projection [m]")
    axes[2].set_xlim(0.0, 50.0)
    for axis in axes:
        axis.grid(True, alpha=0.25)
    axes[2].legend(loc="best")
    figure.tight_layout()
    figure.savefig(root_directory / "comparison.png", dpi=180)
    plt.close(figure)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root_directory), indent=2))


if __name__ == "__main__":
    main()
