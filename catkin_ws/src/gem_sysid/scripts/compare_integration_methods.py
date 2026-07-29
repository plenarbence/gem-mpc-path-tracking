#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from gem_sysid.integration_comparison import compare_test_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Euler and midpoint Euler pose integration on the "
            "synchronized test dataset."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/workspace/data/processed/test.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/workspace/results/model_identification/integration_comparison"
        ),
    )
    args = parser.parse_args()

    metrics_path, plot_path = compare_test_dataset(
        args.dataset,
        args.output_dir,
    )
    print(f"Wrote {metrics_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
