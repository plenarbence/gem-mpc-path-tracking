# Euler Integration Comparison

The fixed Euler and midpoint Euler definitions were evaluated once on
`test_long_mixed`. No training or validation samples were used.

For interval \(\Delta t_k=t_{k+1}-t_k\), Euler uses

\[
\begin{aligned}
x_{k+1} &= x_k + \Delta t_k v_k \cos(\psi_k),\\
y_{k+1} &= y_k + \Delta t_k v_k \sin(\psi_k),\\
\psi_{k+1} &= \psi_k + \Delta t_k \omega_k.
\end{aligned}
\]

Midpoint Euler uses

\[
\bar v_k=\frac{v_k+v_{k+1}}{2},\qquad
\bar\omega_k=\frac{\omega_k+\omega_{k+1}}{2},
\]

\[
\psi_{k+\frac12}
=\psi_k+\frac{\Delta t_k}{2}\bar\omega_k,
\]

\[
\begin{aligned}
x_{k+1} &= x_k + \Delta t_k \bar v_k
           \cos(\psi_{k+\frac12}),\\
y_{k+1} &= y_k + \Delta t_k \bar v_k
           \sin(\psi_{k+\frac12}),\\
\psi_{k+1} &= \psi_k + \Delta t_k \bar\omega_k.
\end{aligned}
\]

The midpoint calculation is an offline, noncausal integration baseline because
it uses the measured values at \(k+1\).

## Results

| Evaluation | Method | XY RMSE | Yaw RMSE |
| --- | --- | ---: | ---: |
| One step | Euler | 0.003118 m | 0.06483 deg |
| One step | Midpoint Euler | 0.000133 m | 0.01019 deg |
| Full 118 s rollout | Euler | 0.735642 m | 0.70202 deg |
| Full 118 s rollout | Midpoint Euler | 0.634701 m | 0.44528 deg |

Midpoint Euler reduces one-step XY RMSE by 95.7% and yaw RMSE by 84.3%.
For the complete open-loop rollout, the reductions are 13.7% and 36.6%.

One-step evaluation starts each prediction from the measured pose. Full rollout
starts only once from the initial measured pose and recursively integrates
odometry speed and odometry yaw rate. The accumulated rollout error therefore
includes small measurement and interpolation biases.
