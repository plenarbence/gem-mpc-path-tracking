#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gem_control.reference_path import build_configured_reference_path


def load_columns(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Control log is empty")
    numeric = {}
    for name in rows[0]:
        try:
            numeric[name] = np.asarray(
                [float(row[name]) for row in rows], dtype=float
            )
        except ValueError:
            continue
    return numeric


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def analyze(run_directory: Path) -> dict[str, object]:
    columns = load_columns(run_directory / "control_log.csv")
    controller_summary = json.loads(
        (run_directory / "summary.json").read_text(encoding="ascii")
    )
    path = build_configured_reference_path()[0]
    progress = np.linspace(
        float(np.min(columns["progress_m"])) - 1.0,
        float(np.max(columns["progress_m"])) + 1.0,
        500,
    )
    reference = path.evaluate(progress)
    lateral = columns["lateral_error_m"]
    yaw_error = columns["yaw_error_rad"]
    total_compute = columns["mpc_total_compute_s"]
    publish_interval = np.diff(columns["publish_time_s"])

    summary = {
        "termination_reason": controller_summary["termination_reason"],
        "horizon_steps": controller_summary["horizon_steps"],
        "cycle_count": len(lateral),
        "progress_travelled_m": float(
            columns["progress_m"][-1] - columns["progress_m"][0]
        ),
        "lateral_error": {
            "rms_m": rms(lateral),
            "p95_absolute_m": float(
                np.quantile(np.abs(lateral), 0.95)
            ),
            "maximum_absolute_m": float(np.max(np.abs(lateral))),
        },
        "yaw_error": {
            "rms_rad": rms(yaw_error),
            "maximum_absolute_rad": float(np.max(np.abs(yaw_error))),
        },
        "timing": {
            "mean_total_compute_ms": float(1000.0 * np.mean(total_compute)),
            "p95_total_compute_ms": float(
                1000.0 * np.quantile(total_compute, 0.95)
            ),
            "maximum_total_compute_ms": float(
                1000.0 * np.max(total_compute)
            ),
            "deadline_miss_count": int(
                np.count_nonzero(total_compute > 0.08)
            ),
            "mean_publish_interval_ms": float(
                1000.0 * np.mean(publish_interval)
            ),
            "p95_absolute_publish_interval_error_ms": float(
                1000.0
                * np.quantile(np.abs(publish_interval - 0.1), 0.95)
            ),
        },
        "commissioned_delay_ms": float(
            columns["commissioned_delay_ms"][0]
        ),
        "stop": controller_summary["stop"],
    }

    time_s = columns["elapsed_s"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    path_axis, error_axis, speed_axis, timing_axis = axes.flat
    path_axis.plot(reference.x, reference.y, "k--", label="reference")
    path_axis.plot(columns["x_m"], columns["y_m"], label="vehicle")
    path_axis.set_xlabel("x [m]")
    path_axis.set_ylabel("y [m]")
    path_axis.set_title("Path tracking")
    path_axis.legend()
    path_axis.grid(True, alpha=0.3)

    error_axis.plot(time_s, lateral, label="lateral [m]")
    error_axis.plot(
        time_s,
        np.rad2deg(yaw_error),
        label="yaw [deg]",
    )
    error_axis.set_xlabel("time [s]")
    error_axis.set_title("Tracking errors")
    error_axis.legend()
    error_axis.grid(True, alpha=0.3)

    speed_axis.plot(time_s, columns["speed_mps"], label="measured")
    speed_axis.plot(
        time_s,
        columns["reference_speed_mps"],
        label="reference",
    )
    speed_axis.plot(
        time_s,
        columns["published_speed_command_mps"],
        label="command",
    )
    speed_axis.set_xlabel("time [s]")
    speed_axis.set_ylabel("speed [m/s]")
    speed_axis.set_title("Speed")
    speed_axis.legend()
    speed_axis.grid(True, alpha=0.3)

    timing_axis.plot(time_s, 1000.0 * total_compute, label="complete MPC")
    timing_axis.axhline(80.0, color="r", linestyle="--", label="deadline")
    timing_axis.set_xlabel("time [s]")
    timing_axis.set_ylabel("time [ms]")
    timing_axis.set_title("Runtime")
    timing_axis.legend()
    timing_axis.grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(run_directory / "validation.png", dpi=160)
    plt.close(figure)

    absolute_limit_m = 1.0
    detail_limit_m = max(
        0.06,
        1.1 * float(np.max(np.abs(lateral))),
    )
    error_figure, error_axes = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": (1.0, 1.35)},
    )
    constraint_axis, detail_axis = error_axes
    for axis in error_axes:
        axis.plot(time_s, lateral, color="tab:blue", linewidth=1.2)
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.grid(True, alpha=0.3)
        axis.set_ylabel("cross-track error [m]")
    constraint_axis.axhline(
        absolute_limit_m,
        color="tab:red",
        linestyle="--",
        label="+/- 1 m assignment limit",
    )
    constraint_axis.axhline(
        -absolute_limit_m,
        color="tab:red",
        linestyle="--",
    )
    constraint_axis.set_ylim(-1.05, 1.05)
    constraint_axis.set_title("Cross-track error: constraint view")
    constraint_axis.legend(loc="upper right")
    detail_axis.set_ylim(-detail_limit_m, detail_limit_m)
    detail_axis.set_title(
        "Detail: RMS {:.4f} m, p95 {:.4f} m, maximum {:.4f} m".format(
            summary["lateral_error"]["rms_m"],
            summary["lateral_error"]["p95_absolute_m"],
            summary["lateral_error"]["maximum_absolute_m"],
        )
    )
    detail_axis.set_xlabel("time [s]")
    error_figure.tight_layout()
    error_figure.savefig(
        run_directory / "cross_track_error.png",
        dpi=160,
    )
    plt.close(error_figure)

    (run_directory / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="ascii",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    summary = analyze(args.run_directory)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
