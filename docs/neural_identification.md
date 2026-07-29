# Neural Dynamics Identification

The identification model consists of two separate residual MLPs:

\[
\Delta v_k=f(z_k),\qquad
\Delta\omega_k=g(z_k),
\]

\[
v_{k+1}=v_k+\Delta v_k,\qquad
\omega_{k+1}=\omega_k+\Delta\omega_k.
\]

The base input is

\[
z_k=[v_k,\omega_k,v_{\mathrm{cmd},k},\delta_{\mathrm{cmd},k}].
\]

Three input histories are evaluated:

- \(z_k\), with 4 inputs.
- \([z_k,z_{k-1}]\), with 8 inputs.
- \([z_k,z_{k-1},z_{k-2}]\), with 12 inputs.

Each network has two 32-unit `tanh` hidden layers and one scalar linear output.
The two networks have independent parameters but are optimized together for
pose-based objectives.

## Training Objectives

Every input history is trained with three objectives:

1. `state_one_step`: normalized one-step speed and yaw-rate error.
2. `pose_one_step`: one-step XY and wrapped-yaw error after midpoint Euler.
3. `pose_rollout20`: recursive 20-step XY and wrapped-yaw error.

For recursive training and evaluation, each window begins with measured pose,
state, and required history. Later states and poses use only model predictions.
Future command samples remain known inputs. Windows never cross profile
boundaries.

All input and target normalization is fitted on training data only. Models use
Adam with learning rate \(10^{-3}\), batch size 128, weight decay \(10^{-5}\),
gradient clipping at 1.0, a 150-epoch ceiling, and validation early stopping
with patience 20. Seeds 17, 43, and 89 are trained for every neural candidate.
A deterministic linear residual model is included as a baseline.

## Validation and Final Test

All candidates are compared on validation data with the same metrics:

- One-step speed RMSE.
- One-step yaw-rate RMSE.
- One-step XY RMSE using predicted next state and midpoint Euler.
- One-step wrapped-yaw RMSE.
- Recursive 20-step endpoint XY RMSE.
- Recursive 20-step endpoint wrapped-yaw RMSE.

The validation composite is the mean of these six RMSE values after
normalization by characteristic scales calculated from training data.
Validation metrics alone determine the selected candidate and seed. Candidate
run files contain no test metrics. After selection is written to
`selection.json`, only the selected checkpoint is evaluated on
`test_long_mixed`.

## Results

The best mean validation candidate is `mlp_h2_state_one_step`. Its selected
seed is 89.

| Metric | Validation | Test |
| --- | ---: | ---: |
| One-step speed RMSE | 0.003201 m/s | 0.004015 m/s |
| One-step yaw-rate RMSE | 0.001513 rad/s | 0.003046 rad/s |
| One-step XY RMSE | 0.000206 m | 0.000253 m |
| One-step yaw RMSE | 0.000111 rad | 0.000202 rad |
| 20-step XY RMSE | 0.047471 m | 0.054778 m |
| 20-step yaw RMSE | 0.012369 rad | 0.009451 rad |

The best candidate uses two previous samples, indicating that command and state
history helps represent lower-controller dynamics. Rollout-trained candidates
with one and two previous samples rank closely behind and show lower
seed-to-seed variation.

Pure one-step pose training performs poorly in recursive rollout because pose
loss alone does not sufficiently constrain the internal speed prediction. The
linear baseline predicts one-step speed competitively but cannot represent the
yaw-rate and recursive pose dynamics.

## MPC Handoff

`IdentifiedDynamicsModel` is the portable inference contract. It loads the
plain-JSON weights and saved train-only scalers, verifies the history shape and
order, and provides next-state and recursive midpoint-rollout methods:

```python
from pathlib import Path

from gem_sysid.neural_model import IdentifiedDynamicsModel

model = IdentifiedDynamicsModel.load(Path("path/to/selected/run"))
next_state = model.predict_next_state(history_z)
states, poses = model.rollout(history_z, commands, dt, initial_pose)
```

`history_z` has shape `[3, 4]` and is ordered current to oldest. Its columns are
speed, yaw rate, speed command, and steering command. The exported interface is
tested against PyTorch. On the complete test profile, the largest difference
is \(2.4\times10^{-7}\) for one-step state, \(1.2\times10^{-6}\) for recursive
state, and \(8.3\times10^{-5}\) for recursive pose.

## Artifacts

Run the complete experiment with:

```bash
rosrun gem_sysid run_neural_identification.py
```

Aggregate results are in
`results/model_identification/neural_experiments/candidate_summary.csv`.
`selection.json` identifies the selected run. `selected_test_metrics.json`
contains its one-time final test evaluation. The assignment evidence and its
source values are:

- `selected_model_test_evidence.png`
- `selected_test_one_step_predictions.csv`
- `selected_test_rollout20_predictions.csv`

Every run directory contains scalers, training history, metrics, separate speed
and yaw-rate PyTorch checkpoints, and plain JSON network weights for later C++
or CasADi use.
