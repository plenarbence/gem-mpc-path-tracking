# Simplified Hierarchical MPC

## Purpose

This controller is a separate alternative to the direct-command learned MPC
and the cascaded-P baseline. It reproduces the hierarchical kinematic MPC from
`v1` while retaining the commissioned 10 Hz command grid, the smoothed closed
reference path, safety checks, visualization, logging, and result analysis.

The learned neural model is not used inside the optimization horizon. It is
used once per control cycle to compensate the intentional one-command-period
execution delay.

## Upper Model

The upper state and input are

```text
z_k = [X_k, Y_k, psi_k, v_k, s_k]
u_k = [Delta v_k, Delta psi_k]
```

The configured horizon is 12 steps at 0.1 s, or 1.2 s. Position uses Euler
propagation. The requested yaw offset is passed through the modeled lower
yaw-P loop and bicycle yaw equation:

```text
X_(k+1)   = X_k + dt v_k cos(psi_k)
Y_(k+1)   = Y_k + dt v_k sin(psi_k)
delta_raw = K_steer wrap(Delta psi_k)
delta     = 0.5 tanh(delta_raw / 0.5)
omega_k   = v_k / L tan(delta)
psi_(k+1) = psi_k + dt omega_k
v_(k+1)   = v_k + Delta v_k
```

The smooth saturation keeps the symbolic optimization differentiable. Runtime
steering uses exact clipping at `+/-0.5 rad`.

## Frenet Progress

`s` is the continuous distance travelled along the smoothed reference path.
The initial value is obtained by numerical closest-point projection. The
horizon advances it with the local Frenet approximation:

```text
e_y       = signed lateral error
e_psi     = psi - psi_ref(s)
s_(k+1)   = s_k + dt v_k cos(e_psi) / (1 - kappa(s_k) e_y)
```

The denominator has a smooth positive floor of `0.25`. All predicted reference
positions, headings, and curvatures are evaluated at the optimized `s_k`.

## Cost And Lower Command

The final weights retain the robust `v1` path/yaw values and use the
Gazebo-validated longitudinal penalty:

```text
8.0               * e_y^2
212.7565413928612 * (1 - cos(e_psi))
0.5580760220602   * (progress_speed - reference_speed)^2
0.05              * (Delta v / 0.2)^2
7.5062518207342   * (Delta psi / 0.1735)^2
```

The original robust `v1` value for the normalized `Delta v` term was `4.0`.
That value made recovery from a Gazebo speed disturbance too expensive: the
first full run fell toward `4.0-4.3 m/s`. Partial high-speed runs tested `0.1`,
`0.2`, and `0.05`. The final `0.05` value recovered speed without exceeding
the measured `20 km/h` limit and was retained for the replacement full lap.

The first optimized input becomes the next Ackermann command:

```text
speed_command    = clip(v_start + Delta v_0, 0.0, 5.5)
steering_command = clip(1.3151152438 * wrap(Delta psi_0), -0.5, 0.5)
```

This matches the `v1` hierarchy. There is no additional lower longitudinal
controller.

## Timing

At each commissioned grid point the node:

1. Publishes the command prepared during the preceding 100 ms interval.
2. Aligns the latest causal odometry to the measured command-takeover anchor.
3. Uses the selected H2 learned model to predict one 100 ms step under the
   command that is already active.
4. Projects small learned speed noise into the physical nonnegative state
   domain.
5. Starts the simplified 12-step MPC from that predicted state.
6. Stores its first command for publication at the next grid point.

Raw and physically projected learned start speeds are both logged. Accepted
solutions shift the previous input sequence by one stage for the next warm
start. A rejected or late solve shifts the last accepted sequence as a safe
fallback. IPOPT receives a 65 ms CPU limit inside the complete 80 ms
calculation budget.

## Run

```bash
roslaunch gem_control simplified_mpc_sim.launch \
  gui:=true use_rviz:=true \
  reference_speed_mps:=5.4166666667 target_laps:=1.0 \
  output_directory:=/workspace/results/simplified_mpc/simulator_h12_19p5_kmh
```

Analyze the recorded CSV:

```bash
rosrun gem_control analyze_simplified_mpc_run.py \
  /workspace/results/simplified_mpc/simulator_h12_19p5_kmh
```

## Validated Lap

The requested `19.5 km/h` Gazebo run completed one lap.

| Metric | Result |
|---|---:|
| Progress | 831.71 m |
| Accepted solves | 1,620 / 1,620 |
| Lateral-error RMS | 0.0122 m |
| Lateral-error p95 absolute | 0.0228 m |
| Lateral-error maximum absolute | 0.0529 m |
| Cruise mean speed | 19.16 km/h |
| Cruise speed range | 18.34 to 19.99 km/h |
| Cruise speed RMSE | 0.1544 m/s |
| Maximum measured speed | 19.99 km/h |
| Mean complete calculation | 16.77 ms |
| p95 complete calculation | 23.60 ms |
| Maximum complete calculation | 60.60 ms |
| Deadline misses | 0 |
| Maximum steering command | 0.0581 rad |
| Steering saturations | 0 |
| Commissioned takeover delay | 4.483 ms |
| Final stationary state | passed |

The path result is well inside the assignment's `+/-1 m` lateral-error limit.
Cruise statistics exclude the first 10 s acceleration period and the
finish-reference slowdown. The final tuning changes only the normalized
`Delta v` penalty; progression, lateral/yaw weights, steering behavior,
command limits, timing, and horizon remain unchanged.
