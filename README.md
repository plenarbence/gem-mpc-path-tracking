# GEM MPC Path Tracking

System identification and model predictive path tracking for the
POLARIS GEM e2 simulator.

## Development Environment

The project is developed on Windows using Docker Desktop with the WSL2
backend. ROS Noetic, Gazebo 11, and the project dependencies will run
inside an Ubuntu 20.04 Docker container.

## Run the Environment

Build the image:

```bash
docker compose build
```

Start the portable CPU-rendered environment:

```bash
docker compose up -d
```

On Windows with Docker Desktop, WSL2, and an NVIDIA GPU, start with the
optional GPU override:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu-wsl.yml up -d
```

Open the container desktop at
<http://localhost:6080/vnc.html?autoconnect=true&resize=scale>.

Open a shell in the running container:

```bash
docker compose exec dev bash
```

Stop the environment:

```bash
docker compose down
```

The GPU override is specific to Docker Desktop's WSL2 backend. The
standard Compose file does not require an NVIDIA GPU.

## System Identification

Launch the uniform-asphalt identification simulator:

```bash
roslaunch gem_sysid sysid_sim.launch
```

Collect the Step 0 profile in a second terminal:

```bash
roslaunch gem_sysid step0_collect.launch \
  output_prefix:=/workspace/data/step0
```

Before every profile, the excitation node keeps the vehicle stopped and
commissions the 10 Hz Ackermann publication phase against all six 30 Hz lower
controller command topics. It targets a 5 ms handoff delay, returns steering to
zero, then holds the resulting phase fixed for the complete run.

Generate the deterministic identification profiles:

```bash
rosrun gem_sysid generate_identification_profiles.py
```

Collect the complete train, validation, and test suite:

```bash
rosrun gem_sysid run_identification_suite.py
```

The runner resets the vehicle, commissions the Ackermann phase, records a bag,
and produces a timing and signal-quality report for every profile. Use
`--split train` or `--profile train_speed_multistep` to collect a subset.
Profile definitions are listed in
`catkin_ws/src/gem_sysid/profiles/identification/manifest.csv`.

Prepare the synchronized 10 Hz datasets:

```bash
rosrun gem_sysid prepare_dataset.py
```

For each command, the synchronization anchor is its stamped publication time
plus that run's commissioned delay. Position, velocity, yaw, and yaw rate are
linearly interpolated from the odometry header timestamps to that anchor. The
generated train, validation, and test CSV files are written to
`data/processed/`.

Compare Euler and midpoint Euler pose integration on the long test profile:

```bash
rosrun gem_sysid compare_integration_methods.py
```

This comparison uses only odometry longitudinal speed and odometry yaw rate.
Results are written to
`results/model_identification/integration_comparison/`.

Train and evaluate the neural identification grid:

```bash
rosrun gem_sysid run_neural_identification.py
```

The grid trains separate residual MLPs for speed and yaw rate using three input
histories, three objectives, and three deterministic seeds. Candidate selection
uses train and validation data only. The selected checkpoint is evaluated once
on the held-out test profile. PyTorch checkpoints and portable JSON weights
are written below `results/model_identification/neural_experiments/`.

The learned state is longitudinal speed \(v\) and yaw rate \(\omega\). For

\[
z_k=[v_k,\omega_k,v_{\mathrm{cmd},k},\delta_{\mathrm{cmd},k}],
\]

two independent residual networks predict

\[
v_{k+1}=v_k+f(z_k,z_{k-1},z_{k-2}),\qquad
\omega_{k+1}=\omega_k+g(z_k,z_{k-1},z_{k-2}).
\]

Position and yaw are propagated with midpoint Euler. The selected model uses
two 32-unit `tanh` layers in each network and was chosen from 27 neural runs
using validation RMSE only.

| Selected-model metric | Validation | Test |
| --- | ---: | ---: |
| One-step speed RMSE | 0.003201 m/s | 0.004015 m/s |
| One-step yaw-rate RMSE | 0.001513 rad/s | 0.003046 rad/s |
| One-step XY RMSE | 0.000206 m | 0.000253 m |
| One-step yaw RMSE | 0.000111 rad | 0.000202 rad |
| Recursive 20-step XY RMSE | 0.047471 m | 0.054778 m |
| Recursive 20-step yaw RMSE | 0.012369 rad | 0.009451 rad |

![Selected model inputs, measured outputs, predictions, and RMSE](results/model_identification/neural_experiments/selected_model_test_evidence.png)

The detailed method, experiment comparison, prediction CSVs, and portable
inference contract are documented in
[`docs/neural_identification.md`](docs/neural_identification.md).

## Status

Docker environment, GEM simulator, uniform identification world, reusable CSV
excitation, timing commissioning, Step 0 analysis, and the complete
identification profile suite, synchronized datasets, and neural dynamics
identification experiment are implemented. The system-identification stage is
complete; MPC path tracking remains to be implemented.
