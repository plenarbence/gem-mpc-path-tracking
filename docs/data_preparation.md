# Data Preparation

`prepare_dataset.py` converts the recorded identification bags into
synchronized 10 Hz CSV datasets.

For command sample \(k\), the state anchor is

\[
t_k = t_{\mathrm{Ackermann},k}
      + \frac{\Delta t_{\mathrm{commissioned}}}{1000}.
\]

The commissioned delay is read from the bag separately for every profile.
Odometry values are linearly interpolated to \(t_k\) using the odometry header
timestamp. Extrapolation is forbidden and causes preparation to fail.

Yaw is extracted from the odometry quaternion and unwrapped before
interpolation. Both wrapped and unwrapped yaw are stored. Odometry velocity is
in the world frame, yaw rate comes from `twist.twist.angular.z`, and
longitudinal vehicle speed is calculated as

\[
v = \cos(\psi)v_x + \sin(\psi)v_y.
\]

## Outputs

The combined files are:

- `data/processed/train.csv`
- `data/processed/validation.csv`
- `data/processed/test.csv`

Each profile also has an individual CSV and metadata JSON below its split
directory. `dataset_manifest.csv` maps profiles to their source bags and output
files. `dataset_summary.json` records interpolation quality and sample counts.

Every row stores the profile name, split, sample index, nominal 10 Hz time,
actual simulator anchor, command publication time, commissioned delay, command
inputs, position, world velocity, longitudinal speed, yaw, and yaw rate.

The combined CSV files retain `profile_name` and reset `sample_index` for every
run. Identification code must form transitions within a profile only; the last
row of one profile must never be paired with the first row of another.
