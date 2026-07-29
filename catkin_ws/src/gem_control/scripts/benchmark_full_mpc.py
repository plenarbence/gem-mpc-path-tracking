#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from gem_control.full_mpc import (
    FullLearnedMpc,
    FullMpcConfig,
    FullMpcInitialCondition,
)
from gem_control.learned_dynamics import midpoint_pose_step_numpy
from gem_control.reference_path import build_configured_reference_path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark configurable learned-MPC horizons."
    )
    parser.add_argument("--solve-count", type=int, default=12)
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=(20, 15, 12, 10),
        help="Prediction horizons to benchmark, longest first.",
    )
    parser.add_argument("--reference-speed", type=float, default=3.0)
    parser.add_argument("--start-progress", type=float, default=40.0)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results/mpc/offline_solver_benchmark"),
    )
    return parser.parse_args()


def benchmark_horizon(
    *,
    path,
    horizon: int,
    solve_count: int,
    reference_speed: float,
    start_progress: float,
) -> tuple[dict, list[dict]]:
    config = FullMpcConfig(
        horizon_steps=horizon,
        reference_speed_mps=reference_speed,
    )
    mpc = FullLearnedMpc(config=config, reference_path=path)
    reference = path.evaluate(np.asarray((start_progress,)))
    yaw = float(reference.yaw[0] + np.deg2rad(4.0))
    lateral_offset = 0.15
    stationary_state = np.asarray(
        (
            reference.x[0],
            reference.y[0],
            reference.yaw[0],
            0.0,
            0.0,
        )
    )
    stationary_command = np.zeros(2)
    stationary_history = np.zeros((2, 4))
    warmup_diagnostics = mpc.stationary_warmup(
        FullMpcInitialCondition(
            state=stationary_state,
            fixed_history_z=stationary_history,
            previous_command=stationary_command,
            previous_progress_m=start_progress,
        )
    )
    state = np.asarray(
        (
            reference.x[0] - lateral_offset * np.sin(reference.yaw[0]),
            reference.y[0] + lateral_offset * np.cos(reference.yaw[0]),
            yaw,
            0.0,
            0.0,
        )
    )
    command = np.zeros(2)
    fixed_history = np.tile(np.r_[state[3:5], command], (2, 1))
    progress_hint = start_progress
    rows = []

    for index in range(solve_count):
        result = mpc.solve(
            FullMpcInitialCondition(
                state=state,
                fixed_history_z=fixed_history,
                previous_command=command,
                previous_progress_m=progress_hint,
            )
        )
        command = result.first_command
        rows.append(
            {
                "horizon_steps": horizon,
                "solve_index": index,
                **asdict(result.diagnostics),
                "speed_command_mps": float(command[0]),
                "steering_command_rad": float(command[1]),
            }
        )

        feature = np.vstack(
            (
                np.r_[state[3:5], command],
                fixed_history[0],
                fixed_history[1],
            )
        )
        next_dynamic = mpc.model.predict_next_state_numpy(feature)
        next_pose = midpoint_pose_step_numpy(
            state[:3],
            state[3:5],
            next_dynamic,
            config.period_s,
        )
        fixed_history = np.vstack(
            (np.r_[state[3:5], command], fixed_history[0])
        )
        state = np.r_[next_pose, next_dynamic]
        progress_hint = path.project(
            state[0], state[1], s_hint=progress_hint
        ).s

    solve_times = np.asarray(
        [row["total_compute_time_s"] for row in rows], dtype=float
    )
    successful = np.asarray(
        [row["solution_accepted"] for row in rows], dtype=bool
    )
    summary = {
        "horizon_steps": horizon,
        "solve_count": solve_count,
        "solver_construction_time_s": mpc.solver_construction_time_s,
        "stationary_warmup_solve_times_s": [
            item.solve_time_s for item in warmup_diagnostics
        ],
        "stationary_warmup_accepted_count": int(
            sum(item.solution_accepted for item in warmup_diagnostics)
        ),
        "accepted_solve_count": int(np.count_nonzero(successful)),
        "deadline_miss_count": int(
            np.count_nonzero(solve_times > config.computation_budget_s)
        ),
        "mean_total_compute_time_s": float(np.mean(solve_times)),
        "p95_total_compute_time_s": float(
            np.quantile(solve_times, 0.95)
        ),
        "maximum_total_compute_time_s": float(np.max(solve_times)),
        "computation_budget_s": config.computation_budget_s,
    }
    return summary, rows


def main() -> None:
    args = parse_arguments()
    if args.solve_count < 2:
        raise ValueError("--solve-count must be at least 2")
    path = build_configured_reference_path()[0]
    summaries = []
    all_rows = []
    horizons = tuple(dict.fromkeys(args.horizons))
    if any(horizon < 2 for horizon in horizons):
        raise ValueError("--horizons values must be at least 2")
    for horizon in horizons:
        summary, rows = benchmark_horizon(
            path=path,
            horizon=horizon,
            solve_count=args.solve_count,
            reference_speed=args.reference_speed,
            start_progress=args.start_progress,
        )
        summaries.append(summary)
        all_rows.extend(rows)

    args.output_directory.mkdir(parents=True, exist_ok=True)
    json_path = args.output_directory / "summary.json"
    csv_path = args.output_directory / "solves.csv"
    json_path.write_text(
        json.dumps(
            {
                "reference_speed_mps": args.reference_speed,
                "start_progress_m": args.start_progress,
                "horizons": summaries,
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    with csv_path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print(json.dumps(summaries, indent=2))
    print("Wrote {}".format(json_path))
    print("Wrote {}".format(csv_path))


if __name__ == "__main__":
    main()
