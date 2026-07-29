#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_ROOT = PACKAGE_ROOT / "profiles" / "identification"
DEFAULT_DATA_ROOT = Path("/workspace/data/identification")
DEFAULT_RESULTS_ROOT = Path("/workspace/results/identification")


def load_manifest(profile_root: Path) -> list[dict[str, str]]:
    manifest_path = profile_root / "manifest.csv"
    with manifest_path.open(newline="", encoding="ascii") as stream:
        return list(csv.DictReader(stream))


def select_profiles(
    manifest: list[dict[str, str]],
    profile_name: str | None,
    split: str | None,
) -> list[dict[str, str]]:
    if profile_name:
        selected = [
            row for row in manifest if row["profile_name"] == profile_name
        ]
        if not selected:
            raise ValueError(f"Unknown profile: {profile_name}")
        return selected
    if split:
        return [row for row in manifest if row["split"] == split]
    return manifest


def run_checked(
    command: list[str],
    *,
    output: Any | None = None,
) -> None:
    completed = subprocess.run(
        command,
        stdout=output,
        stderr=subprocess.STDOUT if output is not None else None,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: "
            + " ".join(command)
        )


def newest_bag(output_prefix: Path, earliest_mtime: float) -> Path:
    candidates = [
        path
        for path in output_prefix.parent.glob(f"{output_prefix.name}_*.bag")
        if path.stat().st_mtime >= earliest_mtime
    ]
    if not candidates:
        raise RuntimeError(f"No completed bag found for {output_prefix}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def collect_profile(
    row: dict[str, str],
    profile_root: Path,
    data_root: Path,
    results_root: Path,
) -> dict[str, Any]:
    name = row["profile_name"]
    split = row["split"]
    csv_path = profile_root / row["csv_path"]
    output_prefix = data_root / split / name
    result_dir = results_root / split / name
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = result_dir / "collection.log"

    print(f"[{split}] {name}: resetting vehicle", flush=True)
    run_checked(["rosservice", "call", "/gazebo/reset_world"])
    time.sleep(1.0)

    started_at = time.time()
    with log_path.open("w", encoding="ascii", errors="replace") as log:
        log.write(f"profile={name}\nsplit={split}\ncsv={csv_path}\n\n")
        log.flush()
        print(f"[{split}] {name}: collecting", flush=True)
        run_checked(
            [
                "roslaunch",
                "gem_sysid",
                "step0_collect.launch",
                f"profile_csv:={csv_path}",
                f"output_prefix:={output_prefix}",
            ],
            output=log,
        )

    bag_path = newest_bag(output_prefix, started_at)
    print(f"[{split}] {name}: analyzing", flush=True)
    with log_path.open("a", encoding="ascii", errors="replace") as log:
        run_checked(
            [
                sys.executable,
                str(PACKAGE_ROOT / "scripts" / "analyze_step0.py"),
                str(bag_path),
                "--output-dir",
                str(result_dir),
                "--artifact-prefix",
                "profile",
            ],
            output=log,
        )

    summary_path = result_dir / "profile_summary.json"
    summary = json.loads(summary_path.read_text(encoding="ascii"))
    expected_samples = int(row["sample_count"])
    actual_samples = int(
        summary["topic_timing"][
            "/gem_sysid/ackermann_cmd_stamped"
        ]["message_count"]
    )
    if actual_samples != expected_samples:
        raise RuntimeError(
            f"{name} recorded {actual_samples} profile commands; "
            f"expected {expected_samples}"
        )

    result = {
        "profile_name": name,
        "split": split,
        "duration_s": float(row["duration_s"]),
        "sample_count": actual_samples,
        "bag_path": str(bag_path),
        "bag_size_bytes": bag_path.stat().st_size,
        "commissioning_mean_delay_ms": summary[
            "commissioning_mean_delay_ms"
        ],
        "recorded_lower_batch_mean_delay_ms": summary[
            "ackermann_publish_to_next_lower_update"
        ]["mean_ms"],
        "recorded_lower_batch_p95_delay_ms": summary[
            "ackermann_publish_to_next_lower_update"
        ]["p95_ms"],
        "result_directory": str(result_dir),
    }
    (result_dir / "collection_result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="ascii",
    )
    print(
        f"[{split}] {name}: complete "
        f"(commissioning {result['commissioning_mean_delay_ms']:.2f} ms)",
        flush=True,
    )
    return result


def write_suite_summary(results: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reset, commission, collect, and analyze Gazebo identification "
            "profiles sequentially."
        )
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--profile")
    selection.add_argument(
        "--split",
        choices=("train", "validation", "test"),
    )
    parser.add_argument(
        "--profile-root",
        type=Path,
        default=DEFAULT_PROFILE_ROOT,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
    )
    args = parser.parse_args()

    manifest = load_manifest(args.profile_root)
    selected = select_profiles(manifest, args.profile, args.split)
    results = [
        collect_profile(
            row,
            args.profile_root,
            args.data_root,
            args.results_root,
        )
        for row in selected
    ]
    write_suite_summary(results, args.results_root / "suite_summary.csv")
    print(
        f"Completed {len(results)} profile(s); "
        f"summary: {args.results_root / 'suite_summary.csv'}"
    )


if __name__ == "__main__":
    main()
