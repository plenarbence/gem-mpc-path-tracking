from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Optional

import casadi as ca
import numpy as np

from gem_control.reference_path import (
    ClosedReferencePath,
    build_configured_reference_path,
)


@dataclass(frozen=True)
class PathParityResult:
    sample_count: int
    maximum_position_difference_m: float
    mean_position_difference_m: float
    maximum_yaw_difference_rad: float
    maximum_curvature_difference_1pm: float
    mean_curvature_difference_1pm: float


class CasadiReferencePath:
    """Differentiable CasADi approximation of a ClosedReferencePath."""

    _instance_ids = itertools.count()

    def __init__(
        self,
        path: Optional[ClosedReferencePath] = None,
        sample_count: int = 2500,
    ) -> None:
        if int(sample_count) < 20:
            raise ValueError("sample_count must be at least 20")
        self.python_path = path or build_configured_reference_path()[0]
        self.length = float(self.python_path.length)
        self.sample_count = int(sample_count)
        suffix = str(next(self._instance_ids))

        grid = np.linspace(0.0, self.length, self.sample_count + 1)
        evaluation = self.python_path.evaluate(grid)
        yaw_unwrapped = np.unwrap(np.asarray(evaluation.yaw, dtype=float))
        self._x_interpolant = ca.interpolant(
            "x_ref_" + suffix,
            "bspline",
            [grid],
            np.asarray(evaluation.x, dtype=float),
        )
        self._y_interpolant = ca.interpolant(
            "y_ref_" + suffix,
            "bspline",
            [grid],
            np.asarray(evaluation.y, dtype=float),
        )
        self._yaw_interpolant = ca.interpolant(
            "yaw_ref_" + suffix,
            "bspline",
            [grid],
            yaw_unwrapped,
        )
        self._curvature_interpolant = ca.interpolant(
            "curvature_ref_" + suffix,
            "bspline",
            [grid],
            np.asarray(evaluation.curvature, dtype=float),
        )

        progress = ca.MX.sym("s")
        values = self.evaluate_symbolic(progress)
        self.function = ca.Function(
            "reference_path_" + suffix,
            [progress],
            [
                values["x"],
                values["y"],
                values["yaw"],
                values["curvature"],
                values["tangent_x"],
                values["tangent_y"],
                values["normal_x"],
                values["normal_y"],
            ],
        )

    def wrap_s(self, s):
        return s - self.length * ca.floor(s / self.length)

    def x(self, s):
        return self._x_interpolant(self.wrap_s(s))

    def y(self, s):
        return self._y_interpolant(self.wrap_s(s))

    def yaw(self, s):
        return self._yaw_interpolant(self.wrap_s(s))

    def curvature(self, s):
        return self._curvature_interpolant(self.wrap_s(s))

    def evaluate_symbolic(self, s):
        yaw = self.yaw(s)
        return {
            "x": self.x(s),
            "y": self.y(s),
            "yaw": yaw,
            "curvature": self.curvature(s),
            "tangent_x": ca.cos(yaw),
            "tangent_y": ca.sin(yaw),
            "normal_x": -ca.sin(yaw),
            "normal_y": ca.cos(yaw),
        }

    def evaluate_numpy(self, s_values: np.ndarray) -> np.ndarray:
        values = np.atleast_1d(np.asarray(s_values, dtype=float))
        rows = []
        for value in values:
            output = self.function(float(value))
            rows.append(
                [
                    float(np.asarray(item).reshape(-1)[0])
                    for item in output
                ]
            )
        return np.asarray(rows, dtype=float)

    def parity_check(self, s_values: np.ndarray) -> PathParityResult:
        progress = np.atleast_1d(np.asarray(s_values, dtype=float))
        casadi_values = self.evaluate_numpy(progress)
        python_values = self.python_path.evaluate(progress)
        position_difference = np.hypot(
            casadi_values[:, 0] - python_values.x,
            casadi_values[:, 1] - python_values.y,
        )
        yaw_difference = np.arctan2(
            np.sin(casadi_values[:, 2] - python_values.yaw),
            np.cos(casadi_values[:, 2] - python_values.yaw),
        )
        curvature_difference = np.abs(
            casadi_values[:, 3] - python_values.curvature
        )
        return PathParityResult(
            sample_count=len(progress),
            maximum_position_difference_m=float(
                np.max(position_difference)
            ),
            mean_position_difference_m=float(
                np.mean(position_difference)
            ),
            maximum_yaw_difference_rad=float(
                np.max(np.abs(yaw_difference))
            ),
            maximum_curvature_difference_1pm=float(
                np.max(curvature_difference)
            ),
            mean_curvature_difference_1pm=float(
                np.mean(curvature_difference)
            ),
        )
