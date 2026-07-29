# Identification Profile Suite

The suite contains six training profiles, three independent validation
profiles, and one longer untouched test profile. All commands use a 10 Hz grid.

| Split | Profile | Duration | Purpose |
| --- | --- | ---: | --- |
| Train | `train_speed_multistep` | 39.0 s | Straight speed steps |
| Train | `train_speed_ramps` | 53.0 s | Speed ramps and operating points |
| Train | `train_steering_multistep` | 70.5 s | Steering steps at several speeds |
| Train | `train_steering_sine` | 78.0 s | Persistent smooth steering |
| Train | `train_steering_chirp` | 68.0 s | Broad-band steering excitation |
| Train | `train_combined_random` | 84.0 s | Coupled deterministic random holds |
| Validation | `validation_speed_unseen` | 37.1 s | Unseen speed levels and durations |
| Validation | `validation_steering_unseen` | 66.0 s | Unseen steering amplitudes and frequencies |
| Validation | `validation_stop_go_turn` | 56.0 s | Stop, restart, and turning transitions |
| Test | `test_long_mixed` | 118.0 s | Longer mixed final evaluation |

The generator is deterministic. Validation and test inputs are separately
defined and must not be used to fit the model.

## Collection Procedure

`run_identification_suite.py` performs the same sequence for every profile:

1. Reset the Gazebo world.
2. Keep the vehicle stationary and commission the 10 Hz command phase.
3. Run the CSV profile and record the measurement topics.
4. Verify the recorded command count and produce signal and timing reports.

The CSV definitions and manifest are in
`catkin_ws/src/gem_sysid/profiles/identification/`. Bags are written below
`data/identification/`; reports are written below `results/identification/`.

## Collected Run

The complete suite was collected on 2026-07-29. All 10 profiles completed with
the expected 6,706 total command samples and no incomplete bag files.
Commissioning means were 4.87 to 4.98 ms against the 5 ms target. The 10 bags
occupy 81.3 MB.

The largest observed world-axis displacement was 200.55 m, inside the uniform
asphalt world's 250 m half-width. The vehicle was reset before every profile.
The small reverse-speed residual at the end of the two straight training runs
was retained because it is measured simulator dynamics, not a data-processing
error.

Detailed per-run paths and timing values are recorded in
`results/identification/suite_summary.csv`.
