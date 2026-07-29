#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from gem_sysid.dataset_preparation import prepare_suite


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Synchronize identification commands, odometry, and IMU data "
            "onto the commissioned 10 Hz command grid."
        )
    )
    parser.add_argument(
        "--suite-summary",
        type=Path,
        default=Path("/workspace/results/identification/suite_summary.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/workspace/data/processed"),
    )
    args = parser.parse_args()

    manifest_path, summary_path = prepare_suite(
        args.suite_summary,
        args.output_root,
    )
    print(f"Wrote {manifest_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
