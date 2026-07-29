from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import rospkg
import yaml
from scipy.interpolate import PchipInterpolator, splev, splprep
from scipy.optimize import minimize_scalar

from gem_control.tracking_errors import lateral_error


PathLike = Union[str, Path]


@dataclass(frozen=True)
class PathPreprocessingConfig:
    duplicate_distance_m: float = 0.02
    startup_motion_step_m: float = 0.05
    minimum_lap_points: int = 500
    closure_search_tail_points: int = 160
    blend_length_m: float = 8.0
    dense_samples: int = 20000
    projection_samples: int = 5000
    smoothing_factor: float = 0.5

    def validate(self) -> None:
        for name in (
            "duplicate_distance_m",
            "startup_motion_step_m",
            "blend_length_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("{} must be positive and finite".format(name))
        for name in (
            "minimum_lap_points",
            "closure_search_tail_points",
            "dense_samples",
            "projection_samples",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError("{} must be positive".format(name))
        if not np.isfinite(self.smoothing_factor) or self.smoothing_factor < 0.0:
            raise ValueError("smoothing_factor must be finite and non-negative")


@dataclass(frozen=True)
class ReferencePathSettings:
    waypoint_package: str
    waypoint_relative_path: str
    preprocessing: PathPreprocessingConfig
    casadi_sample_count: int = 2500

    def validate(self) -> None:
        if not self.waypoint_package:
            raise ValueError("waypoint package must not be empty")
        relative = Path(self.waypoint_relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("waypoint relative path must stay inside its ROS package")
        self.preprocessing.validate()
        if self.casadi_sample_count < 20:
            raise ValueError("CasADi sample count must be at least 20")


@dataclass(frozen=True)
class PathPreprocessingDiagnostics:
    source_csv: str
    raw_point_count: int
    cleaned_point_count: int
    lap_point_count: int
    first_lap_raw_index: int
    last_lap_raw_index: int
    closure_error_before_m: float
    closure_error_after_m: float
    blend_start_lap_index: int
    blend_end_lap_index: int
    blend_length_m: float
    maximum_closure_correction_m: float


@dataclass(frozen=True)
class PathEvaluation:
    s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    curvature: np.ndarray


@dataclass(frozen=True)
class ProjectionResult:
    s: float
    s_wrapped: float
    x: float
    y: float
    distance: float
    signed_lateral_error: float
    yaw: float
    curvature: float


@dataclass(frozen=True)
class SeamDiagnostics:
    seam_position_gap_m: float
    seam_tangent_angle_gap_rad: float
    seam_first_derivative_gap: float
    seam_curvature_gap_1pm: float
    parameter_start: float
    parameter_end: float
    parameter_period: float
    epsilon_s: float


class ClosedReferencePath:
    """Closed, smoothed reference path parameterized by arc length."""

    def __init__(
        self,
        *,
        parameter_control: np.ndarray,
        x_control: np.ndarray,
        y_control: np.ndarray,
        s_to_parameter: PchipInterpolator,
        length: float,
        projection_samples: int,
        smoothing_factor: float,
        bspline_tck: Tuple[object, object, int],
    ) -> None:
        parameter = _as_1d_float("parameter_control", parameter_control)
        x = _as_1d_float("x_control", x_control)
        y = _as_1d_float("y_control", y_control)
        if not (len(parameter) == len(x) == len(y)):
            raise ValueError("parameter, x, and y control arrays must have equal length")
        if len(parameter) < 4:
            raise ValueError("at least four control points are required")
        if not np.all(np.diff(parameter) > 0.0):
            raise ValueError("parameter_control must be strictly increasing")
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError("path length must be positive and finite")
        if int(projection_samples) < 20:
            raise ValueError("projection_samples must be at least 20")

        self._parameter_control = parameter
        self._x_control = x
        self._y_control = y
        self._s_to_parameter = s_to_parameter
        self.length = float(length)
        self.smoothing_factor = float(smoothing_factor)
        self._bspline_tck = bspline_tck

        knots, _, degree = bspline_tck
        self._parameter_start = float(knots[degree])
        self._parameter_end = float(knots[-degree - 1])
        self._parameter_period = self._parameter_end - self._parameter_start
        if self._parameter_period <= 0.0:
            raise ValueError("path parameter period must be positive")

        self._projection_samples = int(projection_samples)
        self._build_projection_cache()

    @property
    def parameter_start(self) -> float:
        return self._parameter_start

    @property
    def parameter_end(self) -> float:
        return self._parameter_end

    @property
    def parameter_period(self) -> float:
        return self._parameter_period

    def evaluate(self, s: Union[np.ndarray, float]) -> PathEvaluation:
        scalar = np.isscalar(s)
        s_array = np.asarray(s, dtype=float)
        if not np.all(np.isfinite(s_array)):
            raise ValueError("s must contain only finite values")
        s_wrapped = np.mod(s_array, self.length)
        parameter = self._s_to_parameter(s_wrapped)
        return self._evaluate_parameter(parameter, s_array, scalar)

    def evaluate_unwrapped_progress(
        self, s: Union[np.ndarray, float]
    ) -> PathEvaluation:
        scalar = np.isscalar(s)
        s_array = np.asarray(s, dtype=float)
        if not np.all(np.isfinite(s_array)):
            raise ValueError("s must contain only finite values")
        parameter = self._s_to_parameter(s_array)
        return self._evaluate_parameter(parameter, s_array, scalar)

    def sample(self, s0: float, ds: float, count: int) -> PathEvaluation:
        if not np.isfinite(s0) or not np.isfinite(ds):
            raise ValueError("s0 and ds must be finite")
        if int(count) != count or count < 0:
            raise ValueError("count must be a non-negative integer")
        values = float(s0) + float(ds) * np.arange(int(count), dtype=float)
        return self.evaluate(values)

    def project(
        self,
        x: float,
        y: float,
        s_hint: Optional[float] = None,
    ) -> ProjectionResult:
        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError("x and y must be finite")
        point = np.array([float(x), float(y)])
        if s_hint is None:
            distances = np.hypot(
                self._projection_x - point[0],
                self._projection_y - point[1],
            )
            candidate_indices = _nearest_indices(distances, 8)
        else:
            if not np.isfinite(s_hint):
                raise ValueError("s_hint must be finite")
            hint_wrapped = float(s_hint) % self.length
            progress_delta = _signed_periodic_delta(
                self._projection_s,
                hint_wrapped,
                self.length,
            )
            mask = np.abs(progress_delta) <= max(3.0, 0.08 * self.length)
            if not np.any(mask):
                mask = np.ones_like(self._projection_s, dtype=bool)
            local_indices = np.flatnonzero(mask)
            local_distances = np.hypot(
                self._projection_x[mask] - point[0],
                self._projection_y[mask] - point[1],
            )
            candidate_indices = local_indices[
                _nearest_indices(local_distances, 8)
            ]

        best_s = 0.0
        best_distance_sq = float("inf")
        spacing = self.length / self._projection_samples
        for index in np.atleast_1d(candidate_indices):
            center = float(self._projection_s[int(index)])

            def objective(candidate_s: float) -> float:
                evaluation = self.evaluate(candidate_s)
                return (
                    (float(evaluation.x) - point[0]) ** 2
                    + (float(evaluation.y) - point[1]) ** 2
                )

            result = minimize_scalar(
                objective,
                bounds=(center - 2.5 * spacing, center + 2.5 * spacing),
                method="bounded",
                options={"xatol": 1e-9},
            )
            if result.fun < best_distance_sq:
                best_distance_sq = float(result.fun)
                best_s = float(result.x)

        if s_hint is not None:
            base_lap = np.floor(float(s_hint) / self.length)
            wrapped = best_s % self.length
            candidates = wrapped + self.length * np.array(
                [base_lap - 1.0, base_lap, base_lap + 1.0]
            )
            best_s = float(
                candidates[np.argmin(np.abs(candidates - float(s_hint)))]
            )

        wrapped_s = best_s % self.length
        evaluation = self.evaluate(wrapped_s)
        signed_error = lateral_error(
            point[0],
            point[1],
            float(evaluation.x),
            float(evaluation.y),
            float(evaluation.yaw),
        )
        return ProjectionResult(
            s=best_s,
            s_wrapped=wrapped_s,
            x=float(evaluation.x),
            y=float(evaluation.y),
            distance=float(np.sqrt(best_distance_sq)),
            signed_lateral_error=signed_error,
            yaw=float(evaluation.yaw),
            curvature=float(evaluation.curvature),
        )

    def project_local(
        self,
        x: float,
        y: float,
        s_hint: float,
        iteration_count: int = 4,
        maximum_step_m: float = 1.0,
    ) -> ProjectionResult:
        """Fast orthogonal projection near a continuous progress hint."""

        if (
            not np.isfinite(x)
            or not np.isfinite(y)
            or not np.isfinite(s_hint)
        ):
            raise ValueError("x, y, and s_hint must be finite")
        if iteration_count < 1 or maximum_step_m <= 0.0:
            raise ValueError("local projection settings are invalid")
        progress = float(s_hint)
        for _ in range(iteration_count):
            evaluation = self.evaluate(progress)
            dx = float(evaluation.x) - float(x)
            dy = float(evaluation.y) - float(y)
            tangent_x = math.cos(float(evaluation.yaw))
            tangent_y = math.sin(float(evaluation.yaw))
            normal_x = -tangent_y
            normal_y = tangent_x
            along_track = dx * tangent_x + dy * tangent_y
            normal_offset = dx * normal_x + dy * normal_y
            denominator = (
                1.0 + float(evaluation.curvature) * normal_offset
            )
            if abs(denominator) < 0.2:
                denominator = math.copysign(0.2, denominator)
            step = np.clip(
                along_track / denominator,
                -maximum_step_m,
                maximum_step_m,
            )
            progress -= float(step)

        wrapped = progress % self.length
        evaluation = self.evaluate(wrapped)
        signed_error = lateral_error(
            x,
            y,
            float(evaluation.x),
            float(evaluation.y),
            float(evaluation.yaw),
        )
        return ProjectionResult(
            s=progress,
            s_wrapped=wrapped,
            x=float(evaluation.x),
            y=float(evaluation.y),
            distance=float(
                np.hypot(float(evaluation.x) - x, float(evaluation.y) - y)
            ),
            signed_lateral_error=float(signed_error),
            yaw=float(evaluation.yaw),
            curvature=float(evaluation.curvature),
        )

    def seam_diagnostics(self, epsilon_s: float = 1e-3) -> SeamDiagnostics:
        if not np.isfinite(epsilon_s) or epsilon_s <= 0.0:
            raise ValueError("epsilon_s must be positive and finite")
        epsilon = min(float(epsilon_s), self.length * 1e-6)
        left_parameter = float(
            self._s_to_parameter(self.length - epsilon)
        )
        right_parameter = float(self._s_to_parameter(epsilon))
        left = self._raw_parameter_values(left_parameter, 0)
        right = self._raw_parameter_values(right_parameter, 0)
        left_d1 = self._raw_parameter_values(left_parameter, 1)
        right_d1 = self._raw_parameter_values(right_parameter, 1)
        left_d2 = self._raw_parameter_values(left_parameter, 2)
        right_d2 = self._raw_parameter_values(right_parameter, 2)
        left_yaw = np.arctan2(left_d1[1], left_d1[0])
        right_yaw = np.arctan2(right_d1[1], right_d1[0])
        yaw_gap = abs(
            np.arctan2(
                np.sin(right_yaw - left_yaw),
                np.cos(right_yaw - left_yaw),
            )
        )
        return SeamDiagnostics(
            seam_position_gap_m=float(np.linalg.norm(right - left)),
            seam_tangent_angle_gap_rad=float(yaw_gap),
            seam_first_derivative_gap=float(np.linalg.norm(right_d1 - left_d1)),
            seam_curvature_gap_1pm=float(
                abs(
                    _curvature(right_d1, right_d2)
                    - _curvature(left_d1, left_d2)
                )
            ),
            parameter_start=self.parameter_start,
            parameter_end=self.parameter_end,
            parameter_period=self.parameter_period,
            epsilon_s=epsilon,
        )

    def _evaluate_parameter(
        self,
        parameter: np.ndarray,
        s_values: np.ndarray,
        scalar: bool,
    ) -> PathEvaluation:
        x, y = (
            np.asarray(value, dtype=float)
            for value in splev(parameter, self._bspline_tck, der=0)
        )
        dx, dy = (
            np.asarray(value, dtype=float)
            for value in splev(parameter, self._bspline_tck, der=1)
        )
        ddx, ddy = (
            np.asarray(value, dtype=float)
            for value in splev(parameter, self._bspline_tck, der=2)
        )
        derivative_norm = np.hypot(dx, dy)
        if np.any(derivative_norm <= 1e-10):
            raise ValueError("path derivative is numerically singular")
        yaw = _wrap_to_pi(np.arctan2(dy, dx))
        curvature = (
            (dx * ddy - dy * ddx)
            / np.maximum(derivative_norm ** 3, 1e-12)
        )
        if scalar:
            return PathEvaluation(
                s=np.asarray(float(np.asarray(s_values))),
                x=np.asarray(float(x)),
                y=np.asarray(float(y)),
                yaw=np.asarray(float(yaw)),
                curvature=np.asarray(float(curvature)),
            )
        return PathEvaluation(
            s=np.asarray(s_values, dtype=float),
            x=x,
            y=y,
            yaw=yaw,
            curvature=curvature,
        )

    def _raw_parameter_values(
        self,
        parameter: float,
        derivative_order: int,
    ) -> np.ndarray:
        values = splev(parameter, self._bspline_tck, der=derivative_order)
        return np.array([float(values[0]), float(values[1])])

    def _build_projection_cache(self) -> None:
        self._projection_s = np.linspace(
            0.0,
            self.length,
            self._projection_samples,
            endpoint=False,
        )
        evaluation = self.evaluate(self._projection_s)
        self._projection_x = evaluation.x
        self._projection_y = evaluation.y


def load_reference_path_settings(
    config_path: Optional[PathLike] = None,
) -> ReferencePathSettings:
    path = (
        Path(config_path)
        if config_path is not None
        else resolve_package_file("gem_control", "config/reference_path.yaml")
    )
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError("reference path configuration must be a mapping")
    waypoint = document.get("waypoint", {})
    preprocessing = document.get("preprocessing", {})
    casadi = document.get("casadi", {})
    settings = ReferencePathSettings(
        waypoint_package=str(waypoint["package"]),
        waypoint_relative_path=str(waypoint["relative_path"]),
        preprocessing=PathPreprocessingConfig(
            duplicate_distance_m=float(
                preprocessing["duplicate_distance_m"]
            ),
            startup_motion_step_m=float(
                preprocessing["startup_motion_step_m"]
            ),
            minimum_lap_points=int(preprocessing["minimum_lap_points"]),
            closure_search_tail_points=int(
                preprocessing["closure_search_tail_points"]
            ),
            blend_length_m=float(preprocessing["blend_length_m"]),
            dense_samples=int(preprocessing["dense_samples"]),
            projection_samples=int(preprocessing["projection_samples"]),
            smoothing_factor=float(preprocessing["smoothing_factor"]),
        ),
        casadi_sample_count=int(casadi["sample_count"]),
    )
    settings.validate()
    return settings


def resolve_package_file(package_name: str, relative_path: PathLike) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("relative_path must stay inside the ROS package")
    try:
        package_root = Path(rospkg.RosPack().get_path(package_name))
        result = package_root.joinpath(relative)
        if result.exists():
            return result
    except rospkg.ResourceNotFound:
        pass

    # This fallback keeps direct source-tree tests useful before catkin is sourced.
    for parent in Path(__file__).resolve().parents:
        candidates = (
            parent / package_name / relative,
            parent
            / "POLARIS_GEM_e2"
            / "polaris_gem_drivers_sim"
            / package_name
            / relative,
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        "could not resolve {}/{} through ROS or the source tree".format(
            package_name,
            relative.as_posix(),
        )
    )


def build_configured_reference_path(
    config_path: Optional[PathLike] = None,
) -> Tuple[
    ClosedReferencePath,
    PathPreprocessingDiagnostics,
    ReferencePathSettings,
]:
    settings = load_reference_path_settings(config_path)
    waypoint_path = resolve_package_file(
        settings.waypoint_package,
        settings.waypoint_relative_path,
    )
    path, diagnostics = build_reference_path(
        waypoint_path,
        settings.preprocessing,
    )
    return path, diagnostics, settings


def build_reference_path(
    csv_path: PathLike,
    config: Optional[PathPreprocessingConfig] = None,
) -> Tuple[ClosedReferencePath, PathPreprocessingDiagnostics]:
    cfg = config or PathPreprocessingConfig()
    cfg.validate()
    source_path = Path(csv_path)
    raw = load_waypoint_csv(source_path)
    raw_indices = np.arange(len(raw), dtype=int)
    cleaned, cleaned_indices = _remove_consecutive_close_points(
        raw,
        raw_indices,
        cfg.duplicate_distance_m,
    )
    lap, lap_indices = _extract_one_lap(cleaned, cleaned_indices, cfg)
    closure_error_before = float(np.linalg.norm(lap[-1] - lap[0]))
    blended, blend_start, corrections = _blend_closure(
        lap,
        cfg.blend_length_m,
    )
    closure_error_after = float(np.linalg.norm(blended[-1] - blended[0]))
    closed_xy = np.vstack([blended, blended[0]])
    path = _fit_smoothed_arc_length_path(closed_xy, cfg)
    diagnostics = PathPreprocessingDiagnostics(
        source_csv=str(source_path.resolve()),
        raw_point_count=len(raw),
        cleaned_point_count=len(cleaned),
        lap_point_count=len(lap),
        first_lap_raw_index=int(lap_indices[0]),
        last_lap_raw_index=int(lap_indices[-1]),
        closure_error_before_m=closure_error_before,
        closure_error_after_m=closure_error_after,
        blend_start_lap_index=int(blend_start),
        blend_end_lap_index=int(len(lap) - 1),
        blend_length_m=cfg.blend_length_m,
        maximum_closure_correction_m=float(
            np.max(np.linalg.norm(corrections, axis=1))
        ),
    )
    return path, diagnostics


def load_waypoint_csv(csv_path: PathLike) -> np.ndarray:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError("waypoint CSV does not exist: {}".format(path))
    values = np.loadtxt(
        str(path),
        delimiter=",",
        usecols=(0, 1),
        ndmin=2,
    )
    if len(values) < 4:
        raise ValueError("waypoint CSV must contain at least four points")
    if not np.all(np.isfinite(values)):
        raise ValueError("waypoint CSV contains NaN or infinity")
    return np.asarray(values, dtype=float)


def _remove_consecutive_close_points(
    points: np.ndarray,
    source_indices: np.ndarray,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    keep = [0]
    for index in range(1, len(points)):
        if np.linalg.norm(points[index] - points[keep[-1]]) >= threshold:
            keep.append(index)
    keep_array = np.asarray(keep, dtype=int)
    return points[keep_array], source_indices[keep_array]


def _extract_one_lap(
    points: np.ndarray,
    source_indices: np.ndarray,
    cfg: PathPreprocessingConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    step_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    moving_candidates = (
        np.flatnonzero(step_lengths >= cfg.startup_motion_step_m) + 1
    )
    if len(moving_candidates) == 0:
        raise ValueError("could not identify a moving waypoint section")
    start = int(moving_candidates[0])
    tail_start = max(
        start + cfg.minimum_lap_points,
        len(points) - cfg.closure_search_tail_points,
    )
    if tail_start >= len(points):
        raise ValueError("not enough points to extract one full lap")
    tail_indices = np.arange(tail_start, len(points), dtype=int)
    closure_distances = np.linalg.norm(
        points[tail_indices] - points[start],
        axis=1,
    )
    end = int(tail_indices[int(np.argmin(closure_distances))])
    if end <= start + cfg.minimum_lap_points:
        raise ValueError("lap closure candidate is too close to the start")
    return (
        points[start : end + 1].copy(),
        source_indices[start : end + 1].copy(),
    )


def _blend_closure(
    points: np.ndarray,
    blend_length_m: float,
) -> Tuple[np.ndarray, int, np.ndarray]:
    residual = points[-1] - points[0]
    segment_lengths = np.hypot(
        np.diff(points[:, 0]),
        np.diff(points[:, 1]),
    )
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    start_distance = max(0.0, cumulative[-1] - blend_length_m)
    blend_start = int(
        np.searchsorted(cumulative, start_distance, side="left")
    )
    if blend_start >= len(points) - 2:
        blend_start = max(0, len(points) - 3)
    blend_progress = (
        (cumulative[blend_start:] - cumulative[blend_start])
        / max(cumulative[-1] - cumulative[blend_start], 1e-12)
    )
    corrections = np.zeros_like(points)
    corrections[blend_start:] = (
        -residual[None, :] * _smoothstep(blend_progress)[:, None]
    )
    blended = points + corrections
    blended[-1] = blended[0]
    corrections[-1] = blended[-1] - points[-1]
    return blended, blend_start, corrections


def _fit_smoothed_arc_length_path(
    closed_xy: np.ndarray,
    cfg: PathPreprocessingConfig,
) -> ClosedReferencePath:
    chord_progress = _chord_lengths(closed_xy)
    # The blended lap already ends at its first point, and closed_xy appends the
    # same point once more. Exclude both terminal duplicates for periodic fitting.
    points = closed_xy[:-2]
    progress = chord_progress[:-2]
    parameter_guess = progress / chord_progress[-1]
    tck, fitted_parameter = splprep(
        [points[:, 0], points[:, 1]],
        u=parameter_guess,
        per=True,
        s=float(cfg.smoothing_factor),
        k=3,
        quiet=2,
    )
    knots, _, degree = tck
    parameter_start = float(knots[degree])
    parameter_end = float(knots[-degree - 1])
    if parameter_end <= parameter_start:
        raise ValueError("invalid periodic smoothing-spline parameter domain")

    dense_parameter = np.linspace(
        parameter_start,
        parameter_end,
        cfg.dense_samples + 1,
    )
    dense_x, dense_y = splev(dense_parameter, tck, der=0)
    dense_lengths = np.hypot(np.diff(dense_x), np.diff(dense_y))
    dense_s = np.concatenate([[0.0], np.cumsum(dense_lengths)])
    keep = np.concatenate([[True], np.diff(dense_s) > 1e-12])
    s_to_parameter = PchipInterpolator(
        dense_s[keep],
        dense_parameter[keep],
        extrapolate=True,
    )

    fitted = np.asarray(fitted_parameter, dtype=float)
    if np.isclose(fitted[-1], parameter_end, rtol=0.0, atol=1e-12):
        parameter_control = fitted
        x_control = points[:, 0]
        y_control = points[:, 1]
    else:
        parameter_control = np.r_[fitted, parameter_end]
        x_control = np.r_[points[:, 0], points[0, 0]]
        y_control = np.r_[points[:, 1], points[0, 1]]
    return ClosedReferencePath(
        parameter_control=parameter_control,
        x_control=x_control,
        y_control=y_control,
        s_to_parameter=s_to_parameter,
        length=float(dense_s[-1]),
        projection_samples=cfg.projection_samples,
        smoothing_factor=cfg.smoothing_factor,
        bspline_tck=tck,
    )


def _chord_lengths(points: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            [0.0],
            np.cumsum(
                np.hypot(
                    np.diff(points[:, 0]),
                    np.diff(points[:, 1]),
                )
            ),
        ]
    )


def _smoothstep(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=float), 0.0, 1.0)
    return clipped ** 3 * (
        10.0 - 15.0 * clipped + 6.0 * clipped ** 2
    )


def _nearest_indices(values: np.ndarray, count: int) -> np.ndarray:
    selected_count = min(int(count), len(values))
    if selected_count == len(values):
        return np.arange(len(values), dtype=int)
    return np.argpartition(values, selected_count - 1)[:selected_count]


def _signed_periodic_delta(
    values: np.ndarray,
    reference: float,
    period: float,
) -> np.ndarray:
    return (values - reference + 0.5 * period) % period - 0.5 * period


def _curvature(first: np.ndarray, second: np.ndarray) -> float:
    dx, dy = first
    ddx, ddy = second
    speed_squared = dx * dx + dy * dy
    return float(
        (dx * ddy - dy * ddx)
        / max(speed_squared ** 1.5, 1e-12)
    )


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _as_1d_float(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError("{} must be one-dimensional".format(name))
    if not np.all(np.isfinite(array)):
        raise ValueError("{} must contain only finite values".format(name))
    return array
