# System Identification Notes

## Step 0: Measurement Pipeline

Step 0 uses the same 24-second, 10 Hz command profile for controlled
comparisons:

- Ramp from rest to 10 km/h.
- Hold speed while steering slowly left and right.
- Return steering and speed commands to zero.

The excitation node publishes an additional
`/gem_sysid/ackermann_cmd_stamped` message because the simulator's
`AckermannDrive` input has no header.

## Surface Decision

The first Step 0 run left the road and crossed onto the grass plane. The
original world defines grass friction as `mu=100` and `mu2=50`, so the surface
transition visibly disturbed speed, steering, and IMU measurements.

System-identification runs therefore use `sysid_flat.world`: a 500 by 500 metre
asphalt plate with explicit uniform friction (`mu=1`, `mu2=1`). The original track
world remains unchanged and will be used for path-tracking evaluation.

The second Step 0 run used the identical command profile on the flat world.
The surface-related disturbances disappeared.

The identification world retains the upstream ODE physics settings: a 1 kHz
update rate with a 1 ms step.

## Joint-State Source

The upstream simulator launch publishes `/gem/joint_states` from two nodes:

- Gazebo's `joint_state_ctrlr` at approximately 30 Hz.
- `joint_state_publisher` at 0.1 Hz.

The second source injects synthetic zero joint positions every 10 seconds.
The identification launch does not start that node, leaving Gazebo as the only
joint-state source during data collection.

## Ackermann Takeover Timing

For each stamped Ackermann publication, the Step 0 analyzer finds the next
message on `/gem/left_rear_wheel_ctrlr/command`. The difference is an observed
upper-bound estimate of takeover delay because the lower-controller message is
not stamped.

Both Step 0 runs produced a delay near 22 ms. This is repeatable within a run
because the command profile uses simulation time at 10 Hz and the lower
controller runs at 30 Hz. It is not guaranteed at startup: without explicit
phase alignment, the fixed delay may land anywhere within one lower-controller
period.

Every excitation run now performs a one-time stationary commissioning step:

- Vehicle speed remains zero.
- Ten alternating `+/-0.02 rad` steering markers are published.
- A lower batch is complete after one message arrives from each of the six
  steering and wheel controller command topics.
- Batch time is the mean receipt time of those six messages.
- The node adjusts only the 10 Hz phase to target a 5 ms delay, then holds that
  phase fixed for the complete profile.
- Steering returns to zero before profile playback.

The bag records `/gem_sysid/profile_start` and
`/gem_sysid/commissioning_delay_ms`. Calibration markers remain auditable, while
analysis uses the profile-start timestamp to exclude them from identification
data.

A stationary smoke test measured a 4.75 ms commissioning mean. Subsequent
fixed-phase profile commands measured 4.88 ms to the next lower update. This is
an observed receipt-time estimate because the lower `Float64` commands have no
headers or timestamps.

The commissioned full Step 0 run is stored in
`data/step0_commissioned_2026-07-29-08-14-26.bag`. Commissioning measured
4.85 ms in the excitation node. During the fixed-phase profile, rosbag measured
the mean receipt time of all six lower topics at 5.98 ms (5.00 ms minimum,
6.83 ms maximum, 6.33 ms p95). The approximately 1 ms difference is subscriber
receipt and recording overhead; neither value is a stamped controller callback
time.

The corresponding plot and JSON summary are in
`results/step0_commissioned/`.

## Runtime Note

The WSL D3D12 path caused `gzserver` to crash inside `libgazebo_ode` when the
vehicle was spawned. The identification launch therefore isolates rendering:
`gzserver` uses stable software GL for physics and sensors, while `gzclient` and
RViz inherit the Docker GPU settings and use the NVIDIA GPU for visualization.

The upstream vehicle also enables front and stereo image cameras at 30 Hz.
These cameras are not identification inputs and make software server rendering
expensive. `generate_sysid_urdf.py` removes only camera and multicamera sensor
blocks from the generated identification URDF. IMU, odometry, joint dynamics,
and vehicle geometry remain unchanged.

With the original 1 kHz physics rate restored, the complete Docker environment
(server, Gazebo client, RViz, ROS, and noVNC) measured about 174% CPU and 661 MiB
RAM. Before removing the unused cameras, it used about 600% CPU.
