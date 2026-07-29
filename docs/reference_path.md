# Reference Path

The `gem_control` package provides the common closed reference path used by
future path-tracking controllers. It ports the validated path processing from
`v1` without depending on the old project tree.

## Source And Configuration

The source is the simulator's
`package://gem_pure_pursuit_sim/waypoints/wps.csv`. ROS package discovery
resolves the file on Linux and inside Docker; no host-specific path is stored.
Runtime settings are in
`catkin_ws/src/gem_control/config/reference_path.yaml`.

The deterministic preprocessing:

1. Removes consecutive points closer than `0.02 m`.
2. Removes the stationary prefix and extracts one complete lap.
3. Blends the final `8 m` to close the recorded `0.2724 m` endpoint gap.
4. Fits a periodic cubic parametric B-spline with smoothing factor `0.5`.
5. Builds a monotonic numerical map from arc length `s` to spline parameter.

The resulting lap length is `831.718009 m`. The path evaluates
`x(s)`, `y(s)`, wrapped yaw, and curvature. Progress may continue beyond one
lap; geometry evaluation wraps it by the lap length. Position projection accepts
an optional progress hint so the returned `s` remains continuous between laps.

`gem_control.tracking_errors` provides matching NumPy and CasADi helpers for
the controller convention:

```text
e_y   = -sin(yaw_ref) (x - x_ref) + cos(yaw_ref) (y - y_ref)
e_yaw = atan2(sin(yaw - yaw_ref), cos(yaw - yaw_ref))
```

Positive lateral error is to the left of the path tangent.

## MPC Interface

`ClosedReferencePath` is the numerical Python implementation used for
projection and validation. `CasadiReferencePath` samples the same geometry and
provides differentiable symbolic expressions for position, yaw, curvature,
tangent, and normal vectors. The configured `5000`-point lookup keeps its
maximum validation differences below:

- position: `0.0001 m`
- yaw: `0.001 rad`
- curvature: `0.001 1/m`

## Validation

After building and sourcing the catkin workspace, run:

```bash
rosrun gem_control validate_reference_path.py
```

The command writes a sampled reference CSV, machine-readable summary, general
path plot, and dedicated waypoint-to-spline difference plot to
`results/reference_path/`. `waypoint_projection_errors.csv` contains the
projected location and signed and Euclidean error for every original waypoint.
The command exits with an error if closure or CasADi parity limits are exceeded.
