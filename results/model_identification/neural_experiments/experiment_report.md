# Neural Identification Experiment

Candidate ranking and seed selection use only train and validation data. The selected checkpoint is evaluated once on the test split after selection.

| Candidate | Seeds | Validation score | v RMSE | omega RMSE | 1-step XY | 1-step yaw | 20-step XY | 20-step yaw |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mlp_h2_state_one_step | 3 | 0.0206699 | 0.00313224 | 0.0021545 | 0.00020858 | 0.000138806 | 0.0472425 | 0.0185123 |
| mlp_h1_pose_rollout20 | 3 | 0.0217437 | 0.00458228 | 0.0029424 | 0.000259925 | 0.000168113 | 0.0571122 | 0.0175432 |
| mlp_h2_pose_rollout20 | 3 | 0.0222686 | 0.0038833 | 0.00334366 | 0.000250155 | 0.000180389 | 0.049721 | 0.0177359 |
| mlp_h1_state_one_step | 3 | 0.0265123 | 0.00334739 | 0.0026711 | 0.000229166 | 0.000161467 | 0.0504986 | 0.0246456 |
| mlp_h0_pose_rollout20 | 3 | 0.0278729 | 0.00458627 | 0.00503202 | 0.000255905 | 0.000259385 | 0.0615724 | 0.0203835 |
| mlp_h0_state_one_step | 3 | 0.0296056 | 0.00335906 | 0.00433443 | 0.000228003 | 0.000240524 | 0.0483651 | 0.0247117 |
| mlp_h2_pose_one_step | 3 | 0.0448269 | 0.0313384 | 0.00256707 | 0.00157488 | 0.000135734 | 0.60042 | 0.0241585 |
| mlp_h1_pose_one_step | 3 | 0.04772 | 0.0313094 | 0.00265846 | 0.00157329 | 0.000144448 | 0.599921 | 0.0276067 |
| mlp_h0_pose_one_step | 3 | 0.049143 | 0.031281 | 0.00419801 | 0.00157129 | 0.000226574 | 0.599083 | 0.0256399 |
| linear_h0_state_one_step | 1 | 0.125572 | 0.00288648 | 0.0174706 | 0.000213641 | 0.000902424 | 0.218487 | 0.109855 |

## Selected Run

Candidate: `mlp_h2_state_one_step`

Seed: `89`

The selected run is the seed with the lowest validation composite inside the highest-ranked candidate.

| Evaluation | Validation | Test |
| --- | ---: | ---: |
| One-step speed RMSE [m/s] | 0.00320051 | 0.00401496 |
| One-step yaw-rate RMSE [rad/s] | 0.00151343 | 0.00304642 |
| One-step XY RMSE [m] | 0.00020623 | 0.000253115 |
| One-step yaw RMSE [rad] | 0.000111322 | 0.000202199 |
| 20-step XY RMSE [m] | 0.0474707 | 0.0547781 |
| 20-step yaw RMSE [rad] | 0.0123691 | 0.00945073 |

## Deployment Check

The selected plain-JSON weights and scalers are loaded through the portable inference interface and compared with PyTorch on the complete test profile.

- Maximum one-step state difference: `2.36e-07`.
- Maximum recursive state difference: `1.14e-06`.
- Maximum recursive pose difference: `8.22e-05`.

## Assignment Evidence

![Selected model inputs, measured outputs, predictions, and RMSE](selected_model_test_evidence.png)

The plotted values are stored in `selected_test_one_step_predictions.csv` and `selected_test_rollout20_predictions.csv`.

## Interpretation

- Direct state training with two previous samples gives the best mean validation composite.
- Rollout-trained candidates with one or two previous samples are close and have lower seed variance.
- Pure one-step pose training does not sufficiently constrain the internal speed prediction and performs poorly over 20 recursive steps.
- The linear baseline predicts speed competitively but cannot represent yaw-rate dynamics or recursive pose behavior.
