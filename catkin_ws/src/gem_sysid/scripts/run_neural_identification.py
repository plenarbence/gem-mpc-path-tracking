#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from gem_sysid.neural_experiment import (
    DEFAULT_HISTORIES,
    DEFAULT_SEEDS,
    finalize_existing_experiment,
    run_experiment_grid,
)
from gem_sysid.neural_training import OBJECTIVES, TrainingSettings


def comma_separated_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(","))


def comma_separated_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(","))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the complete two-model neural identification "
            "experiment grid."
        )
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("/workspace/data/processed"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/workspace/results/model_identification/neural_experiments"
        ),
    )
    parser.add_argument(
        "--histories",
        type=comma_separated_ints,
        default=DEFAULT_HISTORIES,
    )
    parser.add_argument(
        "--objectives",
        type=comma_separated_strings,
        default=OBJECTIVES,
    )
    parser.add_argument(
        "--seeds",
        type=comma_separated_ints,
        default=DEFAULT_SEEDS,
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--no-linear-baseline",
        action="store_true",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help=(
            "Regenerate validation summaries and evaluate the selected "
            "checkpoint once on test."
        ),
    )
    args = parser.parse_args()

    if args.report_only:
        summary_path, report_path = finalize_existing_experiment(
            args.processed_root,
            args.output_root
        )
        print(f"Wrote {summary_path}")
        print(f"Wrote {report_path}")
        return

    settings = TrainingSettings(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.patience,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip,
    )
    summary_path, report_path = run_experiment_grid(
        processed_root=args.processed_root,
        output_root=args.output_root,
        histories=args.histories,
        objectives=args.objectives,
        seeds=args.seeds,
        settings=settings,
        include_linear_baseline=not args.no_linear_baseline,
    )
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
