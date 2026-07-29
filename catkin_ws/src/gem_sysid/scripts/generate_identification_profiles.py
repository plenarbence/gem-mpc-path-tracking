#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


DT = 0.1
MAX_SPEED_MPS = 5.5
MAX_STEERING_RAD = 0.30


@dataclass(frozen=True)
class ProfileDefinition:
    name: str
    split: str
    description: str
    build: Callable[[], "ProfileBuilder"]


class ProfileBuilder:
    def __init__(self) -> None:
        self.commands: list[tuple[float, float]] = []
        self.speed = 0.0
        self.steering = 0.0

    def hold(
        self,
        duration_s: float,
        speed_mps: float | None = None,
        steering_rad: float | None = None,
    ) -> "ProfileBuilder":
        speed = self.speed if speed_mps is None else speed_mps
        steering = self.steering if steering_rad is None else steering_rad
        for _ in range(sample_count(duration_s)):
            self.commands.append((speed, steering))
        self.speed = speed
        self.steering = steering
        return self

    def ramp(
        self,
        duration_s: float,
        speed_mps: float,
        steering_rad: float,
    ) -> "ProfileBuilder":
        count = sample_count(duration_s)
        start_speed = self.speed
        start_steering = self.steering
        for index in range(count):
            fraction = index / count
            self.commands.append(
                (
                    start_speed + fraction * (speed_mps - start_speed),
                    start_steering
                    + fraction * (steering_rad - start_steering),
                )
            )
        self.speed = speed_mps
        self.steering = steering_rad
        return self

    def sine(
        self,
        duration_s: float,
        speed_mps: float,
        amplitude_rad: float,
        frequency_hz: float,
    ) -> "ProfileBuilder":
        count = sample_count(duration_s)
        for index in range(count):
            time_s = index * DT
            self.commands.append(
                (
                    speed_mps,
                    amplitude_rad
                    * math.sin(2.0 * math.pi * frequency_hz * time_s),
                )
            )
        self.speed = speed_mps
        self.steering = amplitude_rad * math.sin(
            2.0 * math.pi * frequency_hz * duration_s
        )
        return self

    def chirp(
        self,
        duration_s: float,
        speed_mps: float,
        amplitude_rad: float,
        start_frequency_hz: float,
        end_frequency_hz: float,
    ) -> "ProfileBuilder":
        count = sample_count(duration_s)
        chirp_rate = (
            end_frequency_hz - start_frequency_hz
        ) / duration_s
        for index in range(count):
            time_s = index * DT
            phase = 2.0 * math.pi * (
                start_frequency_hz * time_s
                + 0.5 * chirp_rate * time_s * time_s
            )
            self.commands.append(
                (speed_mps, amplitude_rad * math.sin(phase))
            )
        final_phase = 2.0 * math.pi * (
            start_frequency_hz * duration_s
            + 0.5 * chirp_rate * duration_s * duration_s
        )
        self.speed = speed_mps
        self.steering = amplitude_rad * math.sin(final_phase)
        return self

    def coupled_sine(
        self,
        duration_s: float,
        speed_offset_mps: float,
        speed_amplitude_mps: float,
        speed_frequency_hz: float,
        steering_amplitude_rad: float,
        steering_frequency_hz: float,
        steering_phase_rad: float,
    ) -> "ProfileBuilder":
        count = sample_count(duration_s)
        for index in range(count):
            time_s = index * DT
            speed = speed_offset_mps + speed_amplitude_mps * math.sin(
                2.0 * math.pi * speed_frequency_hz * time_s
            )
            steering = steering_amplitude_rad * math.sin(
                2.0 * math.pi * steering_frequency_hz * time_s
                + steering_phase_rad
            )
            self.commands.append((speed, steering))
        final_time = duration_s
        self.speed = speed_offset_mps + speed_amplitude_mps * math.sin(
            2.0 * math.pi * speed_frequency_hz * final_time
        )
        self.steering = steering_amplitude_rad * math.sin(
            2.0 * math.pi * steering_frequency_hz * final_time
            + steering_phase_rad
        )
        return self

    def random_holds(
        self,
        duration_s: float,
        seed: int,
        min_hold_s: float = 0.8,
        max_hold_s: float = 2.0,
    ) -> "ProfileBuilder":
        rng = random.Random(seed)
        remaining = sample_count(duration_s)
        min_samples = sample_count(min_hold_s)
        max_samples = sample_count(max_hold_s)
        speed = self.speed
        steering = self.steering
        while remaining:
            hold_samples = min(
                remaining,
                rng.randint(min_samples, max_samples),
            )
            speed = clamp(
                speed + rng.uniform(-1.2, 1.2),
                0.5,
                5.2,
            )
            steering_limit = safe_steering_limit(speed)
            steering = clamp(
                steering + rng.uniform(-0.09, 0.09),
                -steering_limit,
                steering_limit,
            )
            self.commands.extend([(speed, steering)] * hold_samples)
            remaining -= hold_samples
        self.speed = speed
        self.steering = steering
        return self

    def rows(self) -> list[tuple[float, float, float]]:
        rows = [
            (index * DT, speed, steering)
            for index, (speed, steering) in enumerate(self.commands)
        ]
        rows.append(
            (len(self.commands) * DT, self.speed, self.steering)
        )
        return rows


def sample_count(duration_s: float) -> int:
    count = round(duration_s / DT)
    if count <= 0 or abs(count * DT - duration_s) > 1e-9:
        raise ValueError(f"Duration must be a positive {DT}s multiple")
    return count


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def safe_steering_limit(speed_mps: float) -> float:
    if speed_mps >= 4.5:
        return 0.10
    if speed_mps >= 3.5:
        return 0.15
    if speed_mps >= 2.5:
        return 0.20
    return 0.25


def train_speed_multistep() -> ProfileBuilder:
    return (
        ProfileBuilder()
        .hold(2.0)
        .ramp(2.0, 1.0, 0.0)
        .hold(3.0)
        .hold(4.0, 2.0)
        .hold(4.0, 3.5)
        .hold(4.0, 5.0)
        .hold(4.0, 5.5)
        .hold(3.0, 4.0)
        .hold(3.0, 2.0)
        .ramp(4.0, 0.0, 0.0)
        .hold(6.0)
    )


def train_speed_ramps() -> ProfileBuilder:
    return (
        ProfileBuilder()
        .hold(2.0)
        .ramp(12.0, 5.5, 0.0)
        .hold(4.0)
        .ramp(8.0, 1.0, 0.0)
        .hold(3.0)
        .ramp(7.0, 4.0, 0.0)
        .hold(3.0)
        .ramp(8.0, 0.0, 0.0)
        .hold(6.0)
    )


def train_steering_multistep() -> ProfileBuilder:
    profile = ProfileBuilder().hold(2.0).ramp(3.0, 1.5, 0.0).hold(2.0)
    for steering in (
        0.05,
        0.0,
        -0.05,
        0.0,
        0.15,
        0.0,
        -0.15,
        0.0,
        0.30,
        0.0,
        -0.30,
        0.0,
    ):
        profile.hold(2.5, steering_rad=steering)
    profile.ramp(4.0, 3.8, 0.0)
    for steering in (0.05, -0.05, 0.10, -0.10, 0.15, -0.15, 0.0):
        profile.hold(2.5, steering_rad=steering)
    return profile.ramp(6.0, 0.0, 0.0).hold(6.0)


def train_steering_sine() -> ProfileBuilder:
    return (
        ProfileBuilder()
        .hold(2.0)
        .ramp(4.0, 2.5, 0.0)
        .sine(20.0, 2.5, 0.08, 0.10)
        .sine(16.0, 2.5, 0.15, 0.25)
        .ramp(4.0, 4.2, 0.0)
        .sine(20.0, 4.2, 0.10, 0.35)
        .ramp(6.0, 0.0, 0.0)
        .hold(6.0)
    )


def train_steering_chirp() -> ProfileBuilder:
    return (
        ProfileBuilder()
        .hold(2.0)
        .ramp(4.0, 3.2, 0.0)
        .chirp(50.0, 3.2, 0.14, 0.05, 0.75)
        .ramp(6.0, 0.0, 0.0)
        .hold(6.0)
    )


def train_combined_random() -> ProfileBuilder:
    profile = ProfileBuilder().hold(2.0).ramp(4.0, 2.0, 0.0)
    for speed, steering in (
        (1.0, 0.10),
        (2.5, 0.18),
        (4.0, 0.10),
        (5.0, 0.06),
        (3.0, -0.15),
        (1.5, -0.25),
        (4.5, -0.10),
        (2.0, 0.20),
    ):
        profile.hold(2.5, speed, steering)
    return (
        profile.random_holds(45.0, seed=20260729)
        .ramp(6.0, 0.0, 0.0)
        .hold(7.0)
    )


def validation_speed_unseen() -> ProfileBuilder:
    return (
        ProfileBuilder()
        .hold(2.0)
        .ramp(2.0, 0.7, 0.0)
        .hold(3.3)
        .hold(3.7, 1.8)
        .hold(3.4, 3.3)
        .hold(4.2, 4.6)
        .hold(4.0, 5.3)
        .hold(3.5, 2.2)
        .ramp(5.0, 0.0, 0.0)
        .hold(6.0)
    )


def validation_steering_unseen() -> ProfileBuilder:
    return (
        ProfileBuilder()
        .hold(2.0)
        .ramp(4.0, 2.8, 0.0)
        .sine(22.0, 2.8, 0.13, 0.18)
        .ramp(2.0, 2.8, 0.0)
        .ramp(4.0, 4.2, 0.0)
        .sine(20.0, 4.2, 0.09, 0.37)
        .ramp(6.0, 0.0, 0.0)
        .hold(6.0)
    )


def validation_stop_go_turn() -> ProfileBuilder:
    return (
        ProfileBuilder()
        .hold(2.0)
        .ramp(3.0, 2.0, 0.0)
        .hold(5.0, steering_rad=0.15)
        .ramp(4.0, 0.0, 0.15)
        .hold(3.0)
        .ramp(4.0, 1.5, 0.15)
        .hold(3.0, steering_rad=0.0)
        .hold(5.0, 3.5, -0.10)
        .ramp(5.0, 0.0, -0.10)
        .hold(3.0)
        .ramp(4.0, 2.5, -0.10)
        .hold(4.0, steering_rad=0.08)
        .ramp(5.0, 0.0, 0.0)
        .hold(6.0)
    )


def test_long_mixed() -> ProfileBuilder:
    profile = ProfileBuilder().hold(2.0).ramp(4.0, 2.5, 0.0)
    for speed, steering in (
        (1.0, 0.08),
        (3.0, 0.14),
        (5.2, -0.07),
        (2.0, -0.20),
        (4.2, 0.10),
        (1.5, 0.22),
        (4.8, -0.08),
        (2.8, 0.16),
    ):
        profile.hold(2.5, speed, steering)
    return (
        profile.coupled_sine(
            25.0,
            speed_offset_mps=3.0,
            speed_amplitude_mps=1.2,
            speed_frequency_hz=0.04,
            steering_amplitude_rad=0.12,
            steering_frequency_hz=0.16,
            steering_phase_rad=0.5,
        )
        .ramp(3.0, 3.5, 0.0)
        .chirp(25.0, 3.5, 0.10, 0.08, 0.56)
        .random_holds(25.0, seed=56029)
        .ramp(6.0, 0.0, 0.0)
        .hold(8.0)
    )


PROFILES = (
    ProfileDefinition(
        "train_speed_multistep",
        "train",
        "Straight speed steps covering 0 to 5.5 m/s.",
        train_speed_multistep,
    ),
    ProfileDefinition(
        "train_speed_ramps",
        "train",
        "Straight speed ramps with several slopes and operating points.",
        train_speed_ramps,
    ),
    ProfileDefinition(
        "train_steering_multistep",
        "train",
        "Steering steps at low and high speed with speed-dependent limits.",
        train_steering_multistep,
    ),
    ProfileDefinition(
        "train_steering_sine",
        "train",
        "Persistent sine steering excitation at two vehicle speeds.",
        train_steering_sine,
    ),
    ProfileDefinition(
        "train_steering_chirp",
        "train",
        "Broad-band steering chirp at a constant medium speed.",
        train_steering_chirp,
    ),
    ProfileDefinition(
        "train_combined_random",
        "train",
        "Coupled operating points followed by deterministic random holds.",
        train_combined_random,
    ),
    ProfileDefinition(
        "validation_speed_unseen",
        "validation",
        "Unseen straight speed levels and hold durations.",
        validation_speed_unseen,
    ),
    ProfileDefinition(
        "validation_steering_unseen",
        "validation",
        "Unseen sine amplitudes, frequencies, and speeds.",
        validation_steering_unseen,
    ),
    ProfileDefinition(
        "validation_stop_go_turn",
        "validation",
        "Stop, restart, and speed transitions with nonzero steering.",
        validation_stop_go_turn,
    ),
    ProfileDefinition(
        "test_long_mixed",
        "test",
        "Long untouched mixed steps, smooth motion, chirp, and random holds.",
        test_long_mixed,
    ),
)


def validate_rows(
    definition: ProfileDefinition,
    rows: list[tuple[float, float, float]],
) -> None:
    if not rows:
        raise ValueError(f"{definition.name} is empty")
    if rows[0] != (0.0, 0.0, 0.0):
        raise ValueError(f"{definition.name} must start stopped")
    if abs(rows[-1][1]) > 1e-12 or abs(rows[-1][2]) > 1e-12:
        raise ValueError(f"{definition.name} must end stopped")
    for index, (time_s, speed, steering) in enumerate(rows):
        if abs(time_s - index * DT) > 1e-9:
            raise ValueError(f"{definition.name} has an invalid time grid")
        if not 0.0 <= speed <= MAX_SPEED_MPS:
            raise ValueError(f"{definition.name} exceeds the speed limit")
        if abs(steering) > MAX_STEERING_RAD:
            raise ValueError(f"{definition.name} exceeds the steering limit")


def generate(output_root: Path) -> list[dict[str, str | int | float]]:
    manifest: list[dict[str, str | int | float]] = []
    for definition in PROFILES:
        rows = definition.build().rows()
        validate_rows(definition, rows)
        output_path = (
            output_root / definition.split / f"{definition.name}.csv"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="ascii") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("time_s", "speed_mps", "steering_rad"))
            writer.writerows(
                (
                    f"{time_s:.1f}",
                    f"{speed:.9f}",
                    f"{steering:.9f}",
                )
                for time_s, speed, steering in rows
            )
        manifest.append(
            {
                "profile_name": definition.name,
                "split": definition.split,
                "duration_s": rows[-1][0],
                "sample_count": len(rows),
                "max_speed_mps": max(row[1] for row in rows),
                "max_abs_steering_rad": max(abs(row[2]) for row in rows),
                "description": definition.description,
                "csv_path": output_path.relative_to(output_root).as_posix(),
            }
        )

    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic 10 Hz Gazebo identification profiles."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "profiles"
        / "identification",
    )
    args = parser.parse_args()
    manifest = generate(args.output_root)
    for row in manifest:
        print(
            f"{row['split']}/{row['profile_name']}: "
            f"{row['duration_s']:.1f}s, {row['sample_count']} samples"
        )


if __name__ == "__main__":
    main()
