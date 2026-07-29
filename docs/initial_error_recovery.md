# Initial-Error Recovery Comparison

## Purpose

This stress test compares both retained controllers from the same deliberately
misaligned pose. It is kept separate from the nominal assignment lap results.
No controller gains, weights, models, or horizons are changed for this test.

## Scenario

- Reference path-progression rate: `10 km/h`
- Target path progress: `50 m`
- Initial lateral displacement: `0.5 m` along the left path normal
- Initial yaw error: `+40 deg`
- Controller period: `100 ms`
- Timing commissioning: enabled before every run
- Gazebo and the smoothed reference path: identical for all controllers

The initial Gazebo pose is derived from the reference-path origin:

```text
x = 0.1286947801 m
y = -1.5034045379 m
yaw = 0.6961865301 rad
```

The dedicated launch file uses wider `5 m` lateral and `3 rad` yaw safety
guards so the stress response can be observed. The normal controller launch
files retain the assignment guards.

## Reproduction

Run each controller independently:

```bash
roslaunch gem_control initial_error_test.launch \
  controller:=full_learned_mpc

roslaunch gem_control initial_error_test.launch \
  controller:=cascaded_p
```

Generate the common comparison after the two runs:

```bash
rosrun gem_control analyze_initial_error_comparison.py \
  /workspace/results/initial_error_0p5m_40deg_10kmh
```

The comparison defines recovery as the first continuous `1.0 s` for which
both `|lateral error| <= 0.10 m` and `|yaw error| <= 5 deg`.

## Results

| Controller | 50 m completed | Peak lateral error | Settled time | Settled distance | Max compute |
|---|---:|---:|---:|---:|---:|
| Full learned MPC | yes | `1.816 m` | `7.8 s` | `13.75 m` | `79.07 ms` |
| Cascaded P | yes | `1.822 m` | `7.2 s` | `11.11 m` | `0.92 ms` |

The full learned MPC and cascaded P controller both recovered and completed
the requested distance.

For the full learned MPC, `10 km/h` is the requested path-progression rate,
not a vehicle-speed reference. Its vehicle speed can therefore be higher
while the vehicle is strongly misaligned with the path. Ignoring curvature,
the initial `40 deg` yaw error requires approximately
`10 / cos(40 deg) = 13.1 km/h` vehicle speed to obtain `10 km/h` progress
along the path. The comparison plot therefore shows measured `ds/dt` against
the requested progression instead of comparing vehicle speed to a false
speed reference. The derivative uses the matching odometry timestamps before
a five-sample display average; command-publication timestamps are not used to
differentiate odometry-based progress.

Both completing controllers temporarily exceeded the assignment's
`+/-1 m` lateral band during the initial transient. This does not affect the
nominal-lap assignment evidence, but it means this stress scenario is not a
successful `+/-1 m` constrained recovery.

![Initial-error controller comparison](../results/initial_error_0p5m_40deg_10kmh/comparison.png)

Machine-readable metrics are stored in
`results/initial_error_0p5m_40deg_10kmh/comparison_summary.json`.
