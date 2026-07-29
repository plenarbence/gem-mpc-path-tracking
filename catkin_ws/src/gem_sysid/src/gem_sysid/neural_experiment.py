from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from gem_sysid.neural_data import (
    EvaluationScales,
    OneStepArrays,
    ProfileSequence,
    RolloutArrays,
    StandardScaler,
    build_one_step_arrays,
    build_rollout_arrays,
    fit_evaluation_scales,
    fit_scalers,
    load_split,
    wrap_angle,
)
from gem_sysid.neural_evaluation import (
    PRIMARY_METRICS,
    evaluate_model,
    fit_linear_baseline,
    predict_one_step,
    predict_rollouts,
    validation_composite_score,
)
from gem_sysid.neural_model import (
    HIDDEN_LAYERS,
    HIDDEN_WIDTH,
    IdentifiedDynamicsModel,
    ResidualDynamicsPair,
    load_model_pair,
    parameter_count,
    save_model_pair,
)
from gem_sysid.neural_training import (
    OBJECTIVES,
    TrainingSettings,
    train_model_pair,
)


DEFAULT_HISTORIES = (0, 1, 2)
DEFAULT_SEEDS = (17, 43, 89)
ROLLOUT_HORIZON = 20


@dataclass(frozen=True)
class PreparedSplit:
    profiles: list[ProfileSequence]
    one_step: OneStepArrays
    rollout: RolloutArrays


@dataclass(frozen=True)
class PreparedHistory:
    history_depth: int
    train: PreparedSplit
    validation: PreparedSplit
    input_scaler: StandardScaler
    target_scaler: StandardScaler


def _prepare_history(
    split_profiles: dict[str, list[ProfileSequence]],
    history_depth: int,
) -> PreparedHistory:
    prepared: dict[str, PreparedSplit] = {}
    for split, profiles in split_profiles.items():
        prepared[split] = PreparedSplit(
            profiles=profiles,
            one_step=build_one_step_arrays(profiles, history_depth),
            rollout=build_rollout_arrays(
                profiles,
                history_depth,
                ROLLOUT_HORIZON,
            ),
        )
    input_scaler, target_scaler = fit_scalers(
        prepared["train"].one_step
    )
    return PreparedHistory(
        history_depth=history_depth,
        train=prepared["train"],
        validation=prepared["validation"],
        input_scaler=input_scaler,
        target_scaler=target_scaler,
    )


def _feature_names(history_depth: int) -> list[str]:
    names: list[str] = []
    for offset in range(history_depth + 1):
        suffix = "k" if offset == 0 else f"k-{offset}"
        names.extend(
            (
                f"speed_{suffix}",
                f"yaw_rate_{suffix}",
                f"speed_command_{suffix}",
                f"steering_command_{suffix}",
            )
        )
    return names


def _write_csv(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _evaluate_preselection_splits(
    model: ResidualDynamicsPair,
    prepared: PreparedHistory,
    evaluation_scales: EvaluationScales,
) -> tuple[dict[str, dict[str, float | int]], float]:
    metrics = {
        split_name: evaluate_model(
            model,
            split.one_step,
            split.rollout,
            prepared.input_scaler,
            prepared.target_scaler,
        )
        for split_name, split in (
            ("train", prepared.train),
            ("validation", prepared.validation),
        )
    }
    score = validation_composite_score(
        metrics["validation"],
        evaluation_scales,
    )
    return metrics, score


def _run_row(
    *,
    candidate_name: str,
    architecture: str,
    objective: str,
    history_depth: int,
    seed: int,
    parameter_total: int,
    best_epoch: int,
    final_epoch: int,
    best_validation_loss: float,
    termination_reason: str,
    duration_s: float,
    validation_score: float,
    metrics: dict[str, dict[str, float | int]],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "candidate_name": candidate_name,
        "architecture": architecture,
        "objective": objective,
        "history_depth": history_depth,
        "input_size": 4 * (history_depth + 1),
        "seed": seed,
        "parameter_count": parameter_total,
        "best_epoch": best_epoch,
        "final_epoch": final_epoch,
        "best_validation_objective": best_validation_loss,
        "termination_reason": termination_reason,
        "training_duration_s": duration_s,
        "validation_composite_score": validation_score,
    }
    for split, split_metrics in metrics.items():
        for key, value in split_metrics.items():
            row[f"{split}_{key}"] = value
    return row


def _write_run_artifacts(
    run_dir: Path,
    model: ResidualDynamicsPair,
    row: dict[str, Any],
    metrics: dict[str, dict[str, float | int]],
    training_history: list[dict[str, float | int]],
    prepared: PreparedHistory,
    evaluation_scales: EvaluationScales,
    settings: TrainingSettings,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    scalers = {
        "input": prepared.input_scaler.to_dict(),
        "target_delta": prepared.target_scaler.to_dict(),
        "evaluation": evaluation_scales.to_dict(),
    }
    (run_dir / "scalers.json").write_text(
        json.dumps(scalers, indent=2) + "\n",
        encoding="ascii",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "run": row,
                "split_metrics": metrics,
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    _write_csv(training_history, run_dir / "training_history.csv")
    metadata = {
        "candidate_name": row["candidate_name"],
        "architecture": row["architecture"],
        "objective": row["objective"],
        "history_depth": row["history_depth"],
        "feature_order": _feature_names(prepared.history_depth),
        "outputs": ["delta_speed", "delta_yaw_rate"],
        "hidden_width": (
            HIDDEN_WIDTH if row["architecture"] == "mlp" else 0
        ),
        "hidden_layers": (
            HIDDEN_LAYERS if row["architecture"] == "mlp" else 0
        ),
        "activation": (
            "tanh" if row["architecture"] == "mlp" else "none"
        ),
        "training": settings.to_dict(),
        "scalers_file": "scalers.json",
    }
    save_model_pair(model, run_dir, metadata)


def _run_mlp_candidate(
    prepared: PreparedHistory,
    objective: str,
    seed: int,
    output_root: Path,
    evaluation_scales: EvaluationScales,
    settings: TrainingSettings,
) -> dict[str, Any]:
    candidate_name = f"mlp_h{prepared.history_depth}_{objective}"
    run_dir = output_root / "runs" / candidate_name / f"seed_{seed}"
    print(f"{candidate_name}, seed {seed}: training", flush=True)

    torch.manual_seed(seed)
    model = ResidualDynamicsPair(
        input_size=prepared.train.one_step.features.shape[1]
    )
    started = time.perf_counter()
    result = train_model_pair(
        model=model,
        objective=objective,
        train_one_step=prepared.train.one_step,
        validation_one_step=prepared.validation.one_step,
        train_rollout=prepared.train.rollout,
        validation_rollout=prepared.validation.rollout,
        input_scaler=prepared.input_scaler,
        target_scaler=prepared.target_scaler,
        evaluation_scales=evaluation_scales,
        settings=settings,
        seed=seed,
    )
    metrics, score = _evaluate_preselection_splits(
        result.model,
        prepared,
        evaluation_scales,
    )
    duration = time.perf_counter() - started
    row = _run_row(
        candidate_name=candidate_name,
        architecture="mlp",
        objective=objective,
        history_depth=prepared.history_depth,
        seed=seed,
        parameter_total=parameter_count(result.model),
        best_epoch=result.best_epoch,
        final_epoch=result.final_epoch,
        best_validation_loss=result.best_validation_loss,
        termination_reason=result.termination_reason,
        duration_s=duration,
        validation_score=score,
        metrics=metrics,
    )
    _write_run_artifacts(
        run_dir,
        result.model,
        row,
        metrics,
        result.history,
        prepared,
        evaluation_scales,
        settings,
    )
    print(
        f"{candidate_name}, seed {seed}: complete "
        f"(epoch {result.best_epoch}, score {score:.6g}, {duration:.1f}s)",
        flush=True,
    )
    return row


def _run_linear_baseline(
    prepared: PreparedHistory,
    output_root: Path,
    evaluation_scales: EvaluationScales,
    settings: TrainingSettings,
) -> dict[str, Any]:
    candidate_name = "linear_h0_state_one_step"
    run_dir = output_root / "runs" / candidate_name / "deterministic"
    model = fit_linear_baseline(
        prepared.train.one_step,
        prepared.input_scaler,
        prepared.target_scaler,
    )
    metrics, score = _evaluate_preselection_splits(
        model,
        prepared,
        evaluation_scales,
    )
    row = _run_row(
        candidate_name=candidate_name,
        architecture="linear",
        objective="state_one_step",
        history_depth=0,
        seed=-1,
        parameter_total=parameter_count(model),
        best_epoch=0,
        final_epoch=0,
        best_validation_loss=0.0,
        termination_reason="closed_form_ridge",
        duration_s=0.0,
        validation_score=score,
        metrics=metrics,
    )
    _write_run_artifacts(
        run_dir,
        model,
        row,
        metrics,
        [{"epoch": 0, "train_loss": 0.0, "validation_loss": 0.0}],
        prepared,
        evaluation_scales,
        settings,
    )
    print(
        f"{candidate_name}: complete (score {score:.6g})",
        flush=True,
    )
    return row


def _aggregate_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["candidate_name"])].append(row)

    summary: list[dict[str, Any]] = []
    metric_keys = [
        f"{split}_{metric}"
        for split in ("train", "validation")
        for metric in PRIMARY_METRICS
    ]
    for candidate_name, group in grouped.items():
        first = group[0]
        aggregate: dict[str, Any] = {
            "candidate_name": candidate_name,
            "architecture": first["architecture"],
            "objective": first["objective"],
            "history_depth": first["history_depth"],
            "input_size": first["input_size"],
            "seed_count": len(group),
            "parameter_count": first["parameter_count"],
        }
        score_values = np.asarray(
            [float(row["validation_composite_score"]) for row in group]
        )
        aggregate["validation_composite_mean"] = float(
            np.mean(score_values)
        )
        aggregate["validation_composite_std"] = float(
            np.std(score_values)
        )
        for key in metric_keys:
            values = np.asarray([float(row[key]) for row in group])
            aggregate[f"{key}_mean"] = float(np.mean(values))
            aggregate[f"{key}_std"] = float(np.std(values))
        summary.append(aggregate)
    return sorted(
        summary,
        key=lambda row: float(row["validation_composite_mean"]),
    )


def _plot_candidate_summary(
    summary: list[dict[str, Any]],
    path: Path,
) -> None:
    labels = [str(row["candidate_name"]) for row in summary]
    figure, axes = plt.subplots(2, 3, figsize=(19, 11))
    titles = (
        ("state_speed_rmse", "One-Step Speed RMSE [m/s]"),
        ("state_yaw_rate_rmse", "One-Step Yaw-Rate RMSE [rad/s]"),
        ("pose_xy_rmse", "One-Step XY RMSE [m]"),
        ("pose_yaw_rmse", "One-Step Yaw RMSE [rad]"),
        ("rollout20_xy_rmse", "20-Step XY RMSE [m]"),
        ("rollout20_yaw_rmse", "20-Step Yaw RMSE [rad]"),
    )
    positions = np.arange(len(summary))
    for axis, (metric, title) in zip(axes.flat, titles):
        values = [
            float(row[f"validation_{metric}_mean"]) for row in summary
        ]
        errors = [
            float(row[f"validation_{metric}_std"]) for row in summary
        ]
        axis.barh(positions, values, xerr=errors, capsize=2)
        axis.set_yticks(positions)
        axis.set_yticklabels(labels, fontsize=8)
        axis.invert_yaxis()
        axis.set_title(title)
        axis.grid(True, axis="x")
    figure.suptitle(
        "Validation Metrics, Ordered by Validation Composite",
        fontsize=16,
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_report(
    summary: list[dict[str, Any]],
    selected_run: dict[str, Any],
    test_metrics: dict[str, float | int],
    parity: dict[str, float],
    path: Path,
) -> None:
    lines = [
        "# Neural Identification Experiment",
        "",
        "Candidate ranking and seed selection use only train and validation "
        "data. The selected checkpoint is evaluated once on the test split "
        "after selection.",
        "",
        "| Candidate | Seeds | Validation score | v RMSE | omega RMSE | "
        "1-step XY | 1-step yaw | 20-step XY | 20-step yaw |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['candidate_name']} | {row['seed_count']} | "
            f"{float(row['validation_composite_mean']):.6g} | "
            f"{float(row['validation_state_speed_rmse_mean']):.6g} | "
            f"{float(row['validation_state_yaw_rate_rmse_mean']):.6g} | "
            f"{float(row['validation_pose_xy_rmse_mean']):.6g} | "
            f"{float(row['validation_pose_yaw_rmse_mean']):.6g} | "
            f"{float(row['validation_rollout20_xy_rmse_mean']):.6g} | "
            f"{float(row['validation_rollout20_yaw_rmse_mean']):.6g} |"
        )
    lines.extend(
        [
            "",
            "## Selected Run",
            "",
            f"Candidate: `{selected_run['candidate_name']}`",
            "",
            f"Seed: `{selected_run['seed']}`",
            "",
            "The selected run is the seed with the lowest validation "
            "composite inside the highest-ranked candidate.",
            "",
            "| Evaluation | Validation | Test |",
            "| --- | ---: | ---: |",
            f"| One-step speed RMSE [m/s] | "
            f"{float(selected_run['validation_state_speed_rmse']):.6g} | "
            f"{float(test_metrics['state_speed_rmse']):.6g} |",
            f"| One-step yaw-rate RMSE [rad/s] | "
            f"{float(selected_run['validation_state_yaw_rate_rmse']):.6g} | "
            f"{float(test_metrics['state_yaw_rate_rmse']):.6g} |",
            f"| One-step XY RMSE [m] | "
            f"{float(selected_run['validation_pose_xy_rmse']):.6g} | "
            f"{float(test_metrics['pose_xy_rmse']):.6g} |",
            f"| One-step yaw RMSE [rad] | "
            f"{float(selected_run['validation_pose_yaw_rmse']):.6g} | "
            f"{float(test_metrics['pose_yaw_rmse']):.6g} |",
            f"| 20-step XY RMSE [m] | "
            f"{float(selected_run['validation_rollout20_xy_rmse']):.6g} | "
            f"{float(test_metrics['rollout20_xy_rmse']):.6g} |",
            f"| 20-step yaw RMSE [rad] | "
            f"{float(selected_run['validation_rollout20_yaw_rmse']):.6g} | "
            f"{float(test_metrics['rollout20_yaw_rmse']):.6g} |",
            "",
            "## Deployment Check",
            "",
            "The selected plain-JSON weights and scalers are loaded through "
            "the portable inference interface and compared with PyTorch on "
            "the complete test profile.",
            "",
            f"- Maximum one-step state difference: "
            f"`{parity['one_step_state_max_abs']:.3g}`.",
            f"- Maximum recursive state difference: "
            f"`{parity['rollout_state_max_abs']:.3g}`.",
            f"- Maximum recursive pose difference: "
            f"`{parity['rollout_pose_max_abs']:.3g}`.",
            "",
            "## Assignment Evidence",
            "",
            "![Selected model inputs, measured outputs, predictions, and "
            "RMSE](selected_model_test_evidence.png)",
            "",
            "The plotted values are stored in "
            "`selected_test_one_step_predictions.csv` and "
            "`selected_test_rollout20_predictions.csv`.",
            "",
            "## Interpretation",
            "",
            "- Direct state training with two previous samples gives the "
            "best mean validation composite.",
            "- Rollout-trained candidates with one or two previous samples "
            "are close and have lower seed variance.",
            "- Pure one-step pose training does not sufficiently constrain "
            "the internal speed prediction and performs poorly over 20 "
            "recursive steps.",
            "- The linear baseline predicts speed competitively but cannot "
            "represent yaw-rate dynamics or recursive pose behavior.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _strip_test_columns(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in row.items()
            if not key.startswith("test_")
        }
        for row in rows
    ]


def _strip_test_run_artifacts(output_root: Path) -> None:
    for path in (output_root / "runs").glob("*/*/metrics.json"):
        data = json.loads(path.read_text(encoding="ascii"))
        data["run"] = _strip_test_columns([data["run"]])[0]
        data["split_metrics"].pop("test", None)
        path.write_text(
            json.dumps(data, indent=2) + "\n",
            encoding="ascii",
        )


def _load_saved_scalers(
    run_dir: Path,
) -> tuple[StandardScaler, StandardScaler]:
    data = json.loads(
        (run_dir / "scalers.json").read_text(encoding="ascii")
    )
    return (
        StandardScaler.from_dict(data["input"]),
        StandardScaler.from_dict(data["target_delta"]),
    )


def _write_selected_test_artifacts(
    *,
    model: ResidualDynamicsPair,
    portable_model: IdentifiedDynamicsModel,
    test: PreparedSplit,
    input_scaler: StandardScaler,
    target_scaler: StandardScaler,
    test_metrics: dict[str, float | int],
    output_root: Path,
) -> dict[str, float]:
    if len(test.profiles) != 1:
        raise RuntimeError(
            "The assignment evidence plot requires one continuous test profile"
        )
    profile = test.profiles[0]
    one_step = predict_one_step(
        model,
        test.one_step,
        input_scaler,
        target_scaler,
    )
    rollout = predict_rollouts(
        model,
        test.rollout,
        input_scaler,
        target_scaler,
    )

    one_step_rows: list[dict[str, Any]] = []
    portable_one_step_states: list[np.ndarray] = []
    for row_index, sample_index in enumerate(test.one_step.sample_index):
        current_index = int(sample_index)
        history = test.one_step.features[row_index].reshape(
            portable_model.history_depth + 1,
            4,
        )
        portable_state = portable_model.predict_next_state(history)
        portable_one_step_states.append(portable_state)
        state_error = (
            one_step.state[row_index] - test.one_step.next_state[row_index]
        )
        pose_error = (
            one_step.pose[row_index] - test.one_step.next_pose[row_index]
        )
        one_step_rows.append(
            {
                "profile_name": profile.name,
                "time_s": (
                    profile.time[current_index + 1] - profile.time[0]
                ),
                "speed_command_mps": profile.command[current_index, 0],
                "steering_command_rad": profile.command[current_index, 1],
                "measured_speed_mps": test.one_step.next_state[
                    row_index, 0
                ],
                "predicted_speed_mps": one_step.state[row_index, 0],
                "speed_error_mps": state_error[0],
                "measured_yaw_rate_radps": test.one_step.next_state[
                    row_index, 1
                ],
                "predicted_yaw_rate_radps": one_step.state[row_index, 1],
                "yaw_rate_error_radps": state_error[1],
                "measured_x_m": test.one_step.next_pose[row_index, 0],
                "predicted_x_m": one_step.pose[row_index, 0],
                "measured_y_m": test.one_step.next_pose[row_index, 1],
                "predicted_y_m": one_step.pose[row_index, 1],
                "xy_error_m": float(np.linalg.norm(pose_error[:2])),
                "measured_yaw_rad": test.one_step.next_pose[row_index, 2],
                "predicted_yaw_rad": one_step.pose[row_index, 2],
                "yaw_error_rad": float(wrap_angle(pose_error[2])),
            }
        )
    _write_csv(
        one_step_rows,
        output_root / "selected_test_one_step_predictions.csv",
    )

    rollout_rows: list[dict[str, Any]] = []
    portable_rollout_states: list[np.ndarray] = []
    portable_rollout_poses: list[np.ndarray] = []
    horizon = test.rollout.commands.shape[1]
    for row_index, start_index in enumerate(test.rollout.start_index):
        portable_state, portable_pose = portable_model.rollout(
            test.rollout.initial_history_z[row_index],
            test.rollout.commands[row_index],
            test.rollout.dt[row_index],
            test.rollout.initial_pose[row_index],
        )
        portable_rollout_states.append(portable_state)
        portable_rollout_poses.append(portable_pose)
        measured_pose = test.rollout.target_poses[row_index, -1]
        predicted_pose = rollout.pose[row_index, -1]
        pose_error = predicted_pose - measured_pose
        rollout_rows.append(
            {
                "profile_name": profile.name,
                "window_start_time_s": (
                    profile.time[int(start_index)] - profile.time[0]
                ),
                "endpoint_time_s": (
                    profile.time[int(start_index) + horizon]
                    - profile.time[0]
                ),
                "measured_x_m": measured_pose[0],
                "predicted_x_m": predicted_pose[0],
                "measured_y_m": measured_pose[1],
                "predicted_y_m": predicted_pose[1],
                "xy_error_m": float(np.linalg.norm(pose_error[:2])),
                "measured_yaw_rad": measured_pose[2],
                "predicted_yaw_rad": predicted_pose[2],
                "yaw_error_rad": float(wrap_angle(pose_error[2])),
            }
        )
    _write_csv(
        rollout_rows,
        output_root / "selected_test_rollout20_predictions.csv",
    )

    portable_one_step = np.asarray(portable_one_step_states)
    portable_rollout_state = np.asarray(portable_rollout_states)
    portable_rollout_pose = np.asarray(portable_rollout_poses)
    parity = {
        "one_step_state_max_abs": float(
            np.max(np.abs(portable_one_step - one_step.state))
        ),
        "rollout_state_max_abs": float(
            np.max(np.abs(portable_rollout_state - rollout.state))
        ),
        "rollout_pose_max_abs": float(
            np.max(np.abs(portable_rollout_pose - rollout.pose))
        ),
    }
    parity_limits = {
        "one_step_state_max_abs": 1e-5,
        "rollout_state_max_abs": 1e-5,
        "rollout_pose_max_abs": 2e-4,
    }
    failures = [
        key
        for key, limit in parity_limits.items()
        if parity[key] > limit
    ]
    if failures:
        raise RuntimeError(
            "Portable inference parity failed for: "
            + ", ".join(failures)
        )

    command_time = profile.time - profile.time[0]
    one_step_time = np.asarray(
        [float(row["time_s"]) for row in one_step_rows]
    )
    rollout_time = np.asarray(
        [float(row["endpoint_time_s"]) for row in rollout_rows]
    )
    rollout_measured_pose = test.rollout.target_poses[:, -1, :]
    rollout_predicted_pose = rollout.pose[:, -1, :]

    figure, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes[0, 0].plot(
        command_time,
        profile.command[:, 0],
        color="#1f1f1f",
    )
    axes[0, 0].set_title("Input: Speed Command")
    axes[0, 0].set_ylabel("Speed [m/s]")
    axes[0, 1].plot(
        command_time,
        profile.command[:, 1],
        color="#d97706",
    )
    axes[0, 1].set_title("Input: Steering Command")
    axes[0, 1].set_ylabel("Steering [rad]")

    axes[1, 0].plot(
        one_step_time,
        test.one_step.next_state[:, 0],
        label="Measured",
        color="#1d4ed8",
    )
    axes[1, 0].plot(
        one_step_time,
        one_step.state[:, 0],
        label="Predicted",
        color="#dc2626",
        alpha=0.85,
    )
    axes[1, 0].set_title("One-Step Longitudinal Speed")
    axes[1, 0].set_ylabel("Speed [m/s]")
    axes[1, 0].legend()
    axes[1, 0].text(
        0.02,
        0.93,
        f"RMSE = {float(test_metrics['state_speed_rmse']):.6f} m/s",
        transform=axes[1, 0].transAxes,
        va="top",
    )

    axes[1, 1].plot(
        one_step_time,
        test.one_step.next_state[:, 1],
        label="Measured",
        color="#1d4ed8",
    )
    axes[1, 1].plot(
        one_step_time,
        one_step.state[:, 1],
        label="Predicted",
        color="#dc2626",
        alpha=0.85,
    )
    axes[1, 1].set_title("One-Step Yaw Rate")
    axes[1, 1].set_ylabel("Yaw rate [rad/s]")
    axes[1, 1].legend()
    axes[1, 1].text(
        0.02,
        0.93,
        f"RMSE = {float(test_metrics['state_yaw_rate_rmse']):.6f} rad/s",
        transform=axes[1, 1].transAxes,
        va="top",
    )

    axes[2, 0].plot(
        rollout_measured_pose[:, 0],
        rollout_measured_pose[:, 1],
        label="Measured endpoint",
        color="#1d4ed8",
    )
    axes[2, 0].plot(
        rollout_predicted_pose[:, 0],
        rollout_predicted_pose[:, 1],
        label="Predicted 20-step endpoint",
        color="#059669",
        alpha=0.85,
    )
    axes[2, 0].set_title("Recursive 20-Step XY Prediction")
    axes[2, 0].set_xlabel("x [m]")
    axes[2, 0].set_ylabel("y [m]")
    axes[2, 0].axis("equal")
    axes[2, 0].legend(loc="lower right")
    axes[2, 0].text(
        0.02,
        0.93,
        f"Endpoint RMSE = "
        f"{float(test_metrics['rollout20_xy_rmse']):.6f} m",
        transform=axes[2, 0].transAxes,
        va="top",
    )

    axes[2, 1].plot(
        rollout_time,
        rollout_measured_pose[:, 2],
        label="Measured endpoint",
        color="#1d4ed8",
    )
    axes[2, 1].plot(
        rollout_time,
        rollout_predicted_pose[:, 2],
        label="Predicted 20-step endpoint",
        color="#059669",
        alpha=0.85,
    )
    axes[2, 1].set_title("Recursive 20-Step Yaw Prediction")
    axes[2, 1].set_xlabel("Time [s]")
    axes[2, 1].set_ylabel("Yaw [rad]")
    axes[2, 1].legend()
    axes[2, 1].text(
        0.02,
        0.93,
        f"Endpoint RMSE = "
        f"{float(test_metrics['rollout20_yaw_rmse']):.6f} rad",
        transform=axes[2, 1].transAxes,
        va="top",
    )

    for axis in axes.flat:
        axis.grid(True, alpha=0.35)
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 1].set_xlabel("Time [s]")
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 1].set_xlabel("Time [s]")
    figure.suptitle(
        "Selected Dynamics Model on Held-Out Test Profile",
        fontsize=16,
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    figure.savefig(
        output_root / "selected_model_test_evidence.png",
        dpi=180,
    )
    plt.close(figure)
    return parity


def finalize_existing_experiment(
    processed_root: Path,
    output_root: Path,
) -> tuple[Path, Path]:
    run_metrics_path = output_root / "run_metrics.csv"
    with run_metrics_path.open(newline="", encoding="ascii") as stream:
        run_rows = list(csv.DictReader(stream))
    if not run_rows:
        raise RuntimeError(f"No run rows found in {run_metrics_path}")
    run_rows = _strip_test_columns(run_rows)
    _write_csv(run_rows, run_metrics_path)
    _strip_test_run_artifacts(output_root)
    config_path = output_root / "experiment_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="ascii"))
        config["candidate_evaluation_splits"] = ["train", "validation"]
        config["test_policy"] = (
            "Evaluate only the selected checkpoint after selection"
        )
        config_path.write_text(
            json.dumps(config, indent=2) + "\n",
            encoding="ascii",
        )

    summary = _aggregate_runs(run_rows)
    summary_path = output_root / "candidate_summary.csv"
    _write_csv(summary, summary_path)
    selected_candidate = str(summary[0]["candidate_name"])
    selected_run = min(
        (
            row for row in run_rows
            if row["candidate_name"] == selected_candidate
        ),
        key=lambda row: float(row["validation_composite_score"]),
    )
    seed = int(selected_run["seed"])
    selection = {
        "candidate_name": selected_candidate,
        "seed": seed,
        "validation_composite_score": float(
            selected_run["validation_composite_score"]
        ),
        "run_directory": (
            f"runs/{selected_candidate}/"
            + (f"seed_{seed}" if seed >= 0 else "deterministic")
        ),
        "selection_uses_test_metrics": False,
        "test_evaluated_after_selection": False,
    }
    (output_root / "selection.json").write_text(
        json.dumps(selection, indent=2) + "\n",
        encoding="ascii",
    )
    _plot_candidate_summary(
        summary,
        output_root / "validation_candidate_comparison.png",
    )
    run_dir = output_root / str(selection["run_directory"])
    input_scaler, target_scaler = _load_saved_scalers(run_dir)
    history_depth = int(selected_run["history_depth"])
    test_profiles = load_split(processed_root, "test")
    test = PreparedSplit(
        profiles=test_profiles,
        one_step=build_one_step_arrays(test_profiles, history_depth),
        rollout=build_rollout_arrays(
            test_profiles,
            history_depth,
            ROLLOUT_HORIZON,
        ),
    )
    model = load_model_pair(run_dir)
    test_metrics = evaluate_model(
        model,
        test.one_step,
        test.rollout,
        input_scaler,
        target_scaler,
    )
    portable_model = IdentifiedDynamicsModel.load(run_dir)
    parity = _write_selected_test_artifacts(
        model=model,
        portable_model=portable_model,
        test=test,
        input_scaler=input_scaler,
        target_scaler=target_scaler,
        test_metrics=test_metrics,
        output_root=output_root,
    )
    selection["test_evaluated_after_selection"] = True
    (output_root / "selection.json").write_text(
        json.dumps(selection, indent=2) + "\n",
        encoding="ascii",
    )
    (output_root / "selected_test_metrics.json").write_text(
        json.dumps(
            {
                "selection": selection,
                "test_metrics": test_metrics,
                "portable_vs_pytorch": parity,
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    report_path = output_root / "experiment_report.md"
    _write_report(
        summary,
        selected_run,
        test_metrics,
        parity,
        report_path,
    )
    return summary_path, report_path


def run_experiment_grid(
    processed_root: Path,
    output_root: Path,
    histories: tuple[int, ...] = DEFAULT_HISTORIES,
    objectives: tuple[str, ...] = OBJECTIVES,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    settings: TrainingSettings = TrainingSettings(),
    include_linear_baseline: bool = True,
) -> tuple[Path, Path]:
    if not histories or any(depth not in DEFAULT_HISTORIES for depth in histories):
        raise ValueError("histories must contain values from 0, 1, and 2")
    if not objectives or any(value not in OBJECTIVES for value in objectives):
        raise ValueError(f"objectives must be selected from {OBJECTIVES}")
    if not seeds:
        raise ValueError("at least one seed is required")
    settings.validate()

    torch.set_num_threads(min(4, torch.get_num_threads()))
    split_profiles = {
        split: load_split(processed_root, split)
        for split in ("train", "validation")
    }
    prepared = {
        depth: _prepare_history(split_profiles, depth)
        for depth in histories
    }
    global_train_one_step = build_one_step_arrays(
        split_profiles["train"],
        history_depth=0,
    )
    global_train_rollout = build_rollout_arrays(
        split_profiles["train"],
        history_depth=0,
        horizon=ROLLOUT_HORIZON,
    )
    evaluation_scales = fit_evaluation_scales(
        global_train_one_step,
        global_train_rollout,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "experiment_config.json").write_text(
        json.dumps(
            {
                "processed_root": str(processed_root),
                "histories": list(histories),
                "objectives": list(objectives),
                "seeds": list(seeds),
                "rollout_horizon": ROLLOUT_HORIZON,
                "two_separate_models": [
                    "delta_speed",
                    "delta_yaw_rate",
                ],
                "evaluation_scales_train_only": (
                    evaluation_scales.to_dict()
                ),
                "training": settings.to_dict(),
                "candidate_evaluation_splits": [
                    "train",
                    "validation",
                ],
                "test_policy": (
                    "Evaluate only the selected checkpoint after selection"
                ),
                "torch_version": torch.__version__,
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )

    run_rows: list[dict[str, Any]] = []
    if include_linear_baseline:
        if 0 not in prepared:
            baseline_prepared = _prepare_history(split_profiles, 0)
        else:
            baseline_prepared = prepared[0]
        run_rows.append(
            _run_linear_baseline(
                baseline_prepared,
                output_root,
                evaluation_scales,
                settings,
            )
        )

    for history_depth in histories:
        for objective in objectives:
            for seed in seeds:
                run_rows.append(
                    _run_mlp_candidate(
                        prepared[history_depth],
                        objective,
                        seed,
                        output_root,
                        evaluation_scales,
                        settings,
                    )
                )

    _write_csv(run_rows, output_root / "run_metrics.csv")
    return finalize_existing_experiment(processed_root, output_root)
