from __future__ import annotations

from typing import Tuple, Union

import numpy as np


NumericResult = Union[float, np.ndarray]


def lateral_error(
    x,
    y,
    reference_x,
    reference_y,
    reference_yaw,
) -> NumericResult:
    """Return signed cross-track error, positive left of the path tangent."""

    values = _finite_broadcast(
        x,
        y,
        reference_x,
        reference_y,
        reference_yaw,
    )
    x_value, y_value, ref_x, ref_y, ref_yaw = values
    error = (
        -np.sin(ref_yaw) * (x_value - ref_x)
        + np.cos(ref_yaw) * (y_value - ref_y)
    )
    return _scalar_or_array(error)


def yaw_error(yaw, reference_yaw) -> NumericResult:
    """Return measured-minus-reference yaw wrapped to [-pi, pi)."""

    yaw_value, ref_yaw = _finite_broadcast(yaw, reference_yaw)
    difference = yaw_value - ref_yaw
    error = np.arctan2(np.sin(difference), np.cos(difference))
    return _scalar_or_array(error)


def lateral_error_symbolic(
    x,
    y,
    reference_x,
    reference_y,
    reference_yaw,
):
    """CasADi expression matching lateral_error."""

    import casadi as ca

    return (
        -ca.sin(reference_yaw) * (x - reference_x)
        + ca.cos(reference_yaw) * (y - reference_y)
    )


def yaw_error_symbolic(yaw, reference_yaw):
    """CasADi expression matching yaw_error."""

    import casadi as ca

    difference = yaw - reference_yaw
    return ca.atan2(ca.sin(difference), ca.cos(difference))


def _finite_broadcast(*values) -> Tuple[np.ndarray, ...]:
    arrays = tuple(
        np.asarray(value, dtype=float)
        for value in np.broadcast_arrays(*values)
    )
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("tracking-error inputs must be finite")
    return arrays


def _scalar_or_array(value: np.ndarray) -> NumericResult:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return float(array)
    return array
