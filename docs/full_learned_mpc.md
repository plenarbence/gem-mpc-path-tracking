# Full Learned MPC

`gem_control` contains the nonlinear MPC that directly optimizes Ackermann
speed and steering commands, its guarded ROS execution node, and offline and
Gazebo validation tools.

## Frozen Dynamics Model

The selected identification result is copied into
`catkin_ws/src/gem_control/models/selected/`. `source_metadata.json` records
the source run, seed, and SHA-256 hashes. Runtime control therefore does not
depend on mutable training-result directories.

The physical state is

```text
[x, y, yaw, longitudinal_speed, yaw_rate]
```

At every prediction step, separate residual MLPs predict speed and yaw-rate
increments from:

```text
[z_k, z_k-1, z_k-2]
z = [speed, yaw_rate, speed_command, steering_command]
```

The model uses the selected H2 architecture: two 32-unit `tanh` layers in each
network. Pose is propagated with midpoint Euler using the current and predicted
speed and yaw rate. NumPy and CasADi inference have regression and parity tests.

## Timing

The timing preparation is deliberately causal:

1. Snapshot the latest odometry callback before command publication.
2. Extrapolate this measurement to the commissioned application anchor,
   `command publication + 5 ms`, while holding measured speed and yaw rate.
3. Predict one complete `0.1 s` step with the command that is already active.
4. Start the optimization at the next application anchor. Its first optimized
   command is therefore the command expected to take effect there.

The sensor header drives extrapolation while a separate callback-availability
time enforces causality, accommodating small Gazebo header-clock skew. The
`5 ms` value is replaced by the measured commissioning mean for every run.
Commissioning anchors each attempt to the observed lower-controller update.
An attempt that wakes after the intended update and measures the next `30 Hz`
cycle is discarded and retried instead of contaminating the mean.

The execution node publishes the closed CSV reference as a latched
`nav_msgs/Path` on `/gem_control/reference_path`. It publishes a decimated
driven trajectory on `/gem_control/vehicle_path`. The project RViz
configuration uses the ground-truth `world` frame and displays both paths
together with the moving vehicle model.

## Optimization

The decision variables are the state trajectory, direct speed and steering
commands, and continuous path progress. Dynamics and orthogonal path
projection are hard equalities; progress is constrained to be monotonic.

The cost retains the tuned `v1` terms:

```text
8.0        * lateral_error^2
28.2682787 * (1 - cos(yaw_error))
2.4        * (progress_speed - reference_speed)^2
0.25908285 * (steering_command_change / 0.06)^2
0.2061432  * (speed_command_change / 0.2)^2
```

Commands stay inside the identified domain: `0..5.5 m/s` and
`-0.3..0.3 rad`. IPOPT has a `60` iteration limit and a `65 ms` CPU limit,
leaving time for model rollout and constraint checking inside the complete
`80 ms` calculation budget. A CPU-limited iterate is accepted only if it is
finite, feasible within `0.002`, and still inside that complete budget. The
safe fallback holds the previously active command.

After every accepted solve, the next primal guess is:

```text
[u_1, u_2, ..., u_N-1, u_N-1]
```

States are not copied from the old solution. They are re-rolled from the new
delayed initial condition through the learned model. Solver construction and
three stationary solves are performed before enabling command publication; the
last stationary primal solution is retained for the first moving solve. The
wall-time deadline is intentionally not enforced during this disabled-output
warmup. Its separate startup solver has a `0.5 s` CPU cap; the operational
solver retains the strict `65 ms` CPU and `80 ms` complete-calculation limits.

## Horizon Benchmark

Run:

```bash
rosrun gem_control benchmark_full_mpc.py
```

The deterministic benchmark starts `0.15 m` off the path with a `4 degree`
yaw error and runs model-consistent closed-loop steps. Results are stored in
`results/mpc/offline_solver_benchmark/`.

On the development laptop, 20 perturbed solves per horizon produced:

| Horizon | Accepted | Mean | p95 | Maximum |
| ---: | ---: | ---: | ---: | ---: |
| 20 | 0/20 | 87.5 ms | 90.9 ms | 91.0 ms |
| 15 | 0/20 | 83.2 ms | 86.7 ms | 87.4 ms |
| 12 | 20/20 | 47.7 ms | 66.8 ms | 67.1 ms |
| 10 | 20/20 | 41.3 ms | 56.8 ms | 63.6 ms |

Every horizon completed three startup solves before measurement. Horizon 15
exceeded the `80 ms` criterion, while horizon 12 passed every solve. Horizon 12
is therefore the configured `1.2 s` prediction horizon.

## Gazebo Execution

Launch the uniform-asphalt simulator, commissioning, recorder, and controller:

```bash
roslaunch gem_control full_mpc_sim.launch
```

The launch starts at the reference-path origin and caps this validation run at
`2.0 m/s`. It drives for `12 s`, ramps the reference to zero over `3 s`, then
holds zero until odometry confirms the vehicle is stationary. Lateral error,
yaw error, stale odometry, repeated solver failure, and command limits are
guarded.

Analyze a run with:

```bash
rosrun gem_control analyze_full_mpc_run.py \
  /workspace/results/mpc/simulator_h12_run6
```

The analyzer writes the combined `validation.png` and a dedicated
`cross_track_error.png`. The dedicated plot shows the assignment's `+/- 1 m`
constraint, a detailed signed smoothed-path error view, and the unsigned
distance from each logged vehicle position to the nearest original CSV
waypoint in the selected one-lap range.

The completed validation covered 151 control cycles and `27.63 m`. Lateral
error was `0.00725 m` RMS, `0.02034 m` p95 absolute, and `0.02568 m` maximum.
Complete MPC computation was `38.4 ms` mean and `74.7 ms` p95. Two isolated
cycles exceeded `80 ms`; both held the prior feasible command. The vehicle
settled to `0.065 m/s` under zero command. Evidence is stored in
`results/mpc/simulator_h12_run6/`.

Run the final complete lap at `19.5 km/h` (`5.4167 m/s`) with:

```bash
roslaunch gem_control full_mpc_sim.launch \
  gui:=true use_rviz:=true \
  reference_speed_mps:=5.4166666667 target_laps:=1.0 \
  output_directory:=/workspace/results/mpc/simulator_h12_19p5_kmh
```

Lap mode ramps the reference during the first `3 s`, uses a speed-scaled
slowdown near the finish, detects completion from continuous unwrapped path
progress, and then holds zero until stationary. The recorded final lap covered
`831.68 m` in 1,648 controlled cycles. Lateral error was `0.00776 m` RMS,
`0.01301 m` p95 absolute, and `0.05426 m` maximum. Yaw error was
`0.00322 rad` RMS and `0.02743 rad` maximum. Complete MPC computation was
`28.49 ms` mean, `36.63 ms` p95, and `76.71 ms` maximum, with zero `80 ms`
deadline misses. All solves were accepted, so the horizon-10 fallback was not
required and horizon 12 remains selected. Commissioned takeover delay was
`4.50 ms`; the vehicle stopped stationary in `1.60 s`. Evidence is stored in
`results/mpc/simulator_h12_19p5_kmh/`.
