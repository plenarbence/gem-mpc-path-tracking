#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


SAMPLE_PERIOD_S = 0.1
END_TIME_S = 24.0
TARGET_SPEED_MPS = 10.0 / 3.6
STEERING_AMPLITUDE_RAD = 0.10


def command_at(time_s: float) -> tuple[float, float]:
    if time_s < 2.0:
        speed = 0.0
    elif time_s < 6.0:
        speed = TARGET_SPEED_MPS * (time_s - 2.0) / 4.0
    elif time_s < 18.0:
        speed = TARGET_SPEED_MPS
    elif time_s < 22.0:
        speed = TARGET_SPEED_MPS * (22.0 - time_s) / 4.0
    else:
        speed = 0.0

    if 8.0 <= time_s <= 16.0:
        steering = STEERING_AMPLITUDE_RAD * math.sin(
            2.0 * math.pi * (time_s - 8.0) / 8.0
        )
    else:
        steering = 0.0
    return speed, steering


def write_profile(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_count = round(END_TIME_S / SAMPLE_PERIOD_S) + 1
    with output_path.open("w", newline="", encoding="ascii") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(("time_s", "speed_mps", "steering_rad"))
        for index in range(sample_count):
            time_s = index * SAMPLE_PERIOD_S
            speed, steering = command_at(time_s)
            writer.writerow(
                (f"{time_s:.1f}", f"{speed:.6f}", f"{steering:.6f}")
            )


def main() -> None:
    default_output = (
        Path(__file__).resolve().parents[1] / "profiles" / "step0.csv"
    )
    parser = argparse.ArgumentParser(
        description="Generate the 10 Hz Step 0 measurement profile."
    )
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args()
    write_profile(args.output)
    print(f"Wrote Step 0 profile to {args.output}")


if __name__ == "__main__":
    main()
