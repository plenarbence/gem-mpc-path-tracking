# Cascaded P Controller

## Purpose

The cascaded P controller is a deterministic baseline separate from the full
learned MPC. It reuses the same smoothed reference path, 10 Hz commissioned
grid, odometry input, safety limits, visualization topics, stopping logic, and
result format.

It does not use the learned dynamics model or an optimizer.

## Control Law

Positive lateral error is left of the path tangent. The fixed baseline gains are

```text
K_y   = 0.27 rad/m
K_psi = 0.90 rad/rad
```

The two proportional loops are

```text
yaw_compensation = clip(-K_y * lateral_error, +/- 30 deg)
desired_yaw      = path_yaw + yaw_compensation
inner_yaw_error  = wrap(desired_yaw - measured_yaw)
steering_command = clip(K_psi * inner_yaw_error, +/- 0.3 rad)
```

The requested speed is clipped to `[0, 5.5] m/s`.

## Timing

Commissioning aligns command takeover to approximately 5 ms after the 10 Hz
publication grid. At tick `k`, the node publishes the command calculated during
the preceding interval, then calculates a new command from the current
measurement:

```text
measured state x_k -> calculate u_(k+1) -> publish at next grid point
```

The command therefore has an intentional one-step delay. No state prediction
is used. The log records `command_delay_steps=1`,
`state_prediction_enabled=false`, and measurement age at command takeover.

## Run

```bash
roslaunch gem_control cascaded_p_sim.launch \
  gui:=true use_rviz:=true \
  reference_speed_mps:=5.4166666667 target_laps:=1.0 \
  output_directory:=/workspace/results/cascaded_p/simulator_19p5_kmh
```

Analyze the recorded CSV:

```bash
rosrun gem_control analyze_cascaded_p_run.py \
  /workspace/results/cascaded_p/simulator_19p5_kmh
```

## Validated Lap

The commissioned Gazebo run completed one lap at a requested `19.5 km/h`.

| Metric | Result |
|---|---:|
| Progress | 831.70 m |
| Lateral-error RMS | 0.0812 m |
| Lateral-error p95 absolute | 0.1532 m |
| Lateral-error maximum absolute | 0.2971 m |
| Maximum measured speed | 19.76 km/h |
| Mean controller calculation | 0.533 ms |
| Maximum controller calculation | 1.914 ms |
| Deadline misses | 0 |
| Commissioned takeover delay | 4.567 ms |
| Final stationary state | passed |

The original frozen gains were retained without Gazebo-specific retuning.
