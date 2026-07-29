from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import casadi as ca
import numpy as np
import yaml

from gem_control.casadi_reference_path import CasadiReferencePath
from gem_control.reference_path import (
    ClosedReferencePath,
    build_configured_reference_path,
)


@dataclass(frozen=True)
class SimplifiedMpcWeights:
    lateral: float = 8.0
    yaw: float = 212.7565413928612
    progression: float = 0.5580760220602027
    delta_yaw: float = 7.506251820734166
    delta_speed: float = 4.0


@dataclass(frozen=True)
class SimplifiedMpcConfig:
    period_s: float = 0.1
    horizon_steps: int = 12
    computation_budget_s: float = 0.08
    ipopt_cpu_limit_s: float = 0.065
    ipopt_max_iterations: int = 80
    wheelbase_m: float = 1.75
    steering_gain_rad_per_rad: float = 1.3151152437925338
    maximum_steering_command_rad: float = 0.5
    minimum_speed_command_mps: float = 0.0
    maximum_speed_command_mps: float = 5.5
    maximum_predicted_speed_mps: float = 5.56
    delta_speed_scale_mps: float = 0.2
    delta_yaw_scale_rad: float = 0.1735
    frenet_minimum_denominator: float = 0.25
    path_sample_count: int = 2500
    maximum_odometry_age_s: float = 0.12
    weights: SimplifiedMpcWeights = field(
        default_factory=SimplifiedMpcWeights
    )

    def validate(self) -> None:
        positive = (
            self.period_s,
            self.computation_budget_s,
            self.ipopt_cpu_limit_s,
            self.wheelbase_m,
            self.steering_gain_rad_per_rad,
            self.maximum_steering_command_rad,
            self.maximum_speed_command_mps,
            self.maximum_predicted_speed_mps,
            self.delta_speed_scale_mps,
            self.delta_yaw_scale_rad,
            self.frenet_minimum_denominator,
            self.maximum_odometry_age_s,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("positive simplified-MPC values must be finite")
        if self.horizon_steps < 2:
            raise ValueError("horizon_steps must be at least 2")
        if self.ipopt_max_iterations < 1:
            raise ValueError("ipopt_max_iterations must be positive")
        if self.ipopt_cpu_limit_s > self.computation_budget_s:
            raise ValueError("IPOPT CPU limit must fit the computation budget")
        if (
            self.minimum_speed_command_mps < 0.0
            or self.minimum_speed_command_mps
            >= self.maximum_speed_command_mps
        ):
            raise ValueError("invalid speed-command limits")
        for name, value in self.weights.__dict__.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError("{} weight must be non-negative".format(name))


@dataclass(frozen=True)
class SimplifiedMpcDiagnostics:
    solver_success: bool
    solution_accepted: bool
    return_status: str
    iteration_count: int
    solve_time_s: float
    total_compute_time_s: float
    deadline_met: bool
    warm_start_source: str
    maximum_dynamics_residual: float
    objective: float
    fallback_reason: str


@dataclass(frozen=True)
class SimplifiedMpcResult:
    success: bool
    first_command: np.ndarray
    states: np.ndarray
    controls: np.ndarray
    diagnostics: SimplifiedMpcDiagnostics
    first_delta_speed_mps: float
    first_delta_yaw_rad: float
    desired_next_speed_mps: float
    desired_next_yaw_rad: float
    raw_steering_command_rad: float
    steering_command_saturated: bool


def load_simplified_mpc_config(
    path: Path | str | None = None,
) -> SimplifiedMpcConfig:
    config_path = (
        Path(path)
        if path is not None
        else _control_package_path() / "config" / "simplified_mpc.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="ascii"))
    weights = payload["weights"]
    solver = payload["solver"]
    model = payload["model"]
    limits = payload["limits"]
    timing = payload["timing"]
    config = SimplifiedMpcConfig(
        period_s=float(payload["period_s"]),
        horizon_steps=int(payload["horizon_steps"]),
        computation_budget_s=float(solver["computation_budget_s"]),
        ipopt_cpu_limit_s=float(solver["ipopt_cpu_limit_s"]),
        ipopt_max_iterations=int(solver["ipopt_max_iterations"]),
        wheelbase_m=float(model["wheelbase_m"]),
        steering_gain_rad_per_rad=float(
            model["lower_yaw_p_gain_rad_per_rad"]
        ),
        maximum_steering_command_rad=float(
            limits["maximum_steering_command_rad"]
        ),
        minimum_speed_command_mps=float(
            limits["minimum_speed_command_mps"]
        ),
        maximum_speed_command_mps=float(
            limits["maximum_speed_command_mps"]
        ),
        maximum_predicted_speed_mps=float(
            limits["maximum_predicted_speed_mps"]
        ),
        delta_speed_scale_mps=float(model["delta_speed_scale_mps"]),
        delta_yaw_scale_rad=float(model["delta_yaw_scale_rad"]),
        frenet_minimum_denominator=float(
            model["frenet_minimum_denominator"]
        ),
        maximum_odometry_age_s=float(
            timing["maximum_odometry_age_s"]
        ),
        weights=SimplifiedMpcWeights(
            lateral=float(weights["lateral"]),
            yaw=float(weights["yaw"]),
            progression=float(weights["progression"]),
            delta_yaw=float(weights["delta_yaw"]),
            delta_speed=float(weights["delta_speed"]),
        ),
    )
    config.validate()
    return config


def _control_package_path() -> Path:
    try:
        import rospkg
    except ImportError:
        return Path(__file__).resolve().parents[2]
    try:
        return Path(rospkg.RosPack().get_path("gem_control"))
    except rospkg.ResourceNotFound:
        return Path(__file__).resolve().parents[2]


def project_learned_start_state(
    predicted_vehicle_state: np.ndarray,
    config: SimplifiedMpcConfig,
) -> np.ndarray:
    """Project learned speed noise into the MPC's physical state domain."""

    state = np.asarray(predicted_vehicle_state, dtype=float)
    if state.shape != (5,) or not np.isfinite(state).all():
        raise ValueError(
            "predicted state must be [x, y, yaw, speed, yaw_rate]"
        )
    projected = state.copy()
    projected[3] = np.clip(
        projected[3],
        config.minimum_speed_command_mps,
        config.maximum_predicted_speed_mps,
    )
    return projected


class SimplifiedKinematicMpc:
    """Kinematic upper MPC with a modeled lower yaw-P steering loop."""

    _instance_ids = itertools.count()

    def __init__(
        self,
        config: SimplifiedMpcConfig | None = None,
        reference_path: ClosedReferencePath | None = None,
    ) -> None:
        self.config = config or load_simplified_mpc_config()
        self.config.validate()
        self.reference_path = (
            reference_path or build_configured_reference_path()[0]
        )
        self.casadi_path = CasadiReferencePath(
            self.reference_path,
            sample_count=self.config.path_sample_count,
        )
        self.N = self.config.horizon_steps
        self.nx = 5
        self.nu = 2
        self._previous_controls: np.ndarray | None = None
        build_start = time.perf_counter()
        self._build_solver()
        self.solver_construction_time_s = time.perf_counter() - build_start

    def reset_warm_start(self) -> None:
        self._previous_controls = None

    def stage_dynamics_numpy(
        self, state: np.ndarray, control: np.ndarray
    ) -> np.ndarray:
        value = np.asarray(state, dtype=float)
        action = np.asarray(control, dtype=float)
        if value.shape != (5,) or action.shape != (2,):
            raise ValueError("state/control shapes must be (5,) and (2,)")
        geometry = self.reference_path.evaluate(value[4])
        lateral = (
            -math.sin(float(geometry.yaw))
            * (value[0] - float(geometry.x))
            + math.cos(float(geometry.yaw))
            * (value[1] - float(geometry.y))
        )
        denominator = 1.0 - float(geometry.curvature) * lateral
        safe_denominator = self._smooth_positive_floor(denominator)
        wrapped_delta_yaw = math.atan2(
            math.sin(action[1]), math.cos(action[1])
        )
        raw_steering = (
            self.config.steering_gain_rad_per_rad * wrapped_delta_yaw
        )
        steering = self._smooth_steering(raw_steering)
        yaw_rate = value[3] / self.config.wheelbase_m * math.tan(steering)
        return np.asarray(
            (
                value[0]
                + self.config.period_s * value[3] * math.cos(value[2]),
                value[1]
                + self.config.period_s * value[3] * math.sin(value[2]),
                value[2] + self.config.period_s * yaw_rate,
                value[3] + action[0],
                value[4]
                + self.config.period_s
                * value[3]
                * math.cos(value[2] - float(geometry.yaw))
                / safe_denominator,
            ),
            dtype=float,
        )

    def solve(
        self,
        *,
        predicted_vehicle_state: np.ndarray,
        previous_progress_m: float,
        reference_speed_mps: float,
        warm_start_allowed: bool = True,
        enforce_deadline: bool = True,
    ) -> SimplifiedMpcResult:
        total_start = time.perf_counter()
        vehicle = project_learned_start_state(
            predicted_vehicle_state, self.config
        )
        if not np.isfinite(reference_speed_mps):
            raise ValueError("reference_speed_mps must be finite")
        projection = self.reference_path.project_local(
            vehicle[0],
            vehicle[1],
            previous_progress_m,
        )
        state0 = np.asarray(
            (vehicle[0], vehicle[1], vehicle[2], vehicle[3], projection.s)
        )
        parameters = np.r_[state0, float(reference_speed_mps)]
        controls, warm_source = self._initial_controls(
            state0,
            reference_speed_mps,
            warm_start_allowed,
        )
        states = self._rollout(state0, controls)
        guess = self._pack(states, controls)

        solve_start = time.perf_counter()
        try:
            solution = self._solver(
                x0=guess,
                p=parameters,
                lbx=self._lower_bounds,
                ubx=self._upper_bounds,
                lbg=self._constraint_bounds,
                ubg=self._constraint_bounds,
            )
            solve_time = time.perf_counter() - solve_start
            candidate = np.asarray(solution["x"], dtype=float).reshape(-1)
            stats = self._solver.stats()
            status = str(stats.get("return_status", "unknown"))
            solver_success = bool(stats.get("success", False))
            iterations = int(stats.get("iter_count", 0))
        except Exception as error:
            solve_time = time.perf_counter() - solve_start
            candidate = guess
            status = "exception:{}".format(error)
            solver_success = False
            iterations = 0

        candidate_states, candidate_controls = self._unpack(candidate)
        finite = bool(
            np.isfinite(candidate_states).all()
            and np.isfinite(candidate_controls).all()
        )
        residual = (
            self._maximum_dynamics_residual(candidate, parameters)
            if finite
            else math.inf
        )
        objective = (
            float(self._objective_function(candidate, parameters))
            if finite
            else math.inf
        )
        total_compute = time.perf_counter() - total_start
        deadline_met = total_compute <= self.config.computation_budget_s
        accepted = (
            solver_success
            and status in ("Solve_Succeeded", "Solved_To_Acceptable_Level")
            and finite
            and residual <= 1e-4
            and (deadline_met or not enforce_deadline)
        )
        fallback_reason = ""
        if accepted:
            used_states = candidate_states
            used_controls = candidate_controls
            self._previous_controls = candidate_controls.copy()
        else:
            if not solver_success:
                fallback_reason = "solver failure:{}".format(status)
            elif not finite:
                fallback_reason = "nonfinite candidate"
            elif residual > 1e-4:
                fallback_reason = "dynamics residual"
            else:
                fallback_reason = "computation deadline"
            used_controls = self._fallback_controls()
            used_states = self._rollout(state0, used_controls)
            objective = float(
                self._objective_function(
                    self._pack(used_states, used_controls), parameters
                )
            )

        first_delta_speed = float(used_controls[0, 0])
        first_delta_yaw = float(used_controls[0, 1])
        desired_speed = float(state0[3] + first_delta_speed)
        desired_yaw = float(state0[2] + first_delta_yaw)
        raw_steering = (
            self.config.steering_gain_rad_per_rad
            * math.atan2(
                math.sin(first_delta_yaw), math.cos(first_delta_yaw)
            )
        )
        steering = float(
            np.clip(
                raw_steering,
                -self.config.maximum_steering_command_rad,
                self.config.maximum_steering_command_rad,
            )
        )
        speed_command = float(
            np.clip(
                desired_speed,
                self.config.minimum_speed_command_mps,
                self.config.maximum_speed_command_mps,
            )
        )
        return SimplifiedMpcResult(
            success=accepted,
            first_command=np.asarray((speed_command, steering)),
            states=used_states,
            controls=used_controls,
            diagnostics=SimplifiedMpcDiagnostics(
                solver_success=solver_success,
                solution_accepted=accepted,
                return_status=status,
                iteration_count=iterations,
                solve_time_s=solve_time,
                total_compute_time_s=total_compute,
                deadline_met=deadline_met,
                warm_start_source=warm_source,
                maximum_dynamics_residual=residual,
                objective=objective,
                fallback_reason=fallback_reason,
            ),
            first_delta_speed_mps=first_delta_speed,
            first_delta_yaw_rad=first_delta_yaw,
            desired_next_speed_mps=desired_speed,
            desired_next_yaw_rad=desired_yaw,
            raw_steering_command_rad=float(raw_steering),
            steering_command_saturated=not math.isclose(
                raw_steering, steering, rel_tol=0.0, abs_tol=1e-12
            ),
        )

    def _build_solver(self) -> None:
        suffix = str(next(self._instance_ids))
        states = ca.MX.sym("simplified_states_" + suffix, self.nx, self.N + 1)
        controls = ca.MX.sym(
            "simplified_controls_" + suffix, self.nu, self.N
        )
        parameters = ca.MX.sym("simplified_parameters_" + suffix, 6)
        constraints = [states[:, 0] - parameters[:5]]
        objective = 0
        weights = self.config.weights

        for stage in range(self.N):
            state = states[:, stage]
            control = controls[:, stage]
            geometry = self.casadi_path.evaluate_symbolic(state[4])
            lateral = (
                -ca.sin(geometry["yaw"]) * (state[0] - geometry["x"])
                + ca.cos(geometry["yaw"]) * (state[1] - geometry["y"])
            )
            yaw_error = state[2] - geometry["yaw"]
            next_state = self._stage_dynamics_symbolic(
                state, control, geometry, lateral
            )
            constraints.append(states[:, stage + 1] - next_state)
            progress_speed = (
                states[4, stage + 1] - states[4, stage]
            ) / self.config.period_s
            objective += (
                weights.lateral * lateral**2
                + weights.yaw * (1.0 - ca.cos(yaw_error))
                + weights.progression
                * (progress_speed - parameters[5]) ** 2
                + weights.delta_speed
                * (control[0] / self.config.delta_speed_scale_mps) ** 2
                + weights.delta_yaw
                * (control[1] / self.config.delta_yaw_scale_rad) ** 2
            )

        decision = ca.vertcat(
            ca.reshape(states, -1, 1),
            ca.reshape(controls, -1, 1),
        )
        constraint_vector = ca.vertcat(*constraints)
        nlp = {
            "x": decision,
            "p": parameters,
            "f": objective,
            "g": constraint_vector,
        }
        options = {
            "ipopt": {
                "print_level": 0,
                "max_iter": self.config.ipopt_max_iterations,
                "max_cpu_time": self.config.ipopt_cpu_limit_s,
                "tol": 1e-5,
                "acceptable_tol": 1e-4,
                "warm_start_init_point": "yes",
            },
            "print_time": False,
        }
        self._solver = ca.nlpsol(
            "simplified_mpc_" + suffix, "ipopt", nlp, options
        )
        self._constraint_function = ca.Function(
            "simplified_constraints_" + suffix,
            [decision, parameters],
            [constraint_vector],
        )
        self._objective_function = ca.Function(
            "simplified_objective_" + suffix,
            [decision, parameters],
            [objective],
        )
        decision_size = self.nx * (self.N + 1) + self.nu * self.N
        self._lower_bounds = np.full(decision_size, -np.inf)
        self._upper_bounds = np.full(decision_size, np.inf)
        for stage in range(self.N + 1):
            speed_index = self.nx * stage + 3
            self._lower_bounds[speed_index] = 0.0
            self._upper_bounds[speed_index] = (
                self.config.maximum_predicted_speed_mps
            )
        self._constraint_bounds = np.zeros(self.nx * (self.N + 1))

    def _stage_dynamics_symbolic(self, state, control, geometry, lateral):
        denominator = 1.0 - geometry["curvature"] * lateral
        difference = denominator - self.config.frenet_minimum_denominator
        safe_denominator = (
            self.config.frenet_minimum_denominator
            + 0.5 * (difference + ca.sqrt(difference**2 + 1e-8))
        )
        wrapped_delta_yaw = ca.atan2(
            ca.sin(control[1]), ca.cos(control[1])
        )
        raw_steering = (
            self.config.steering_gain_rad_per_rad * wrapped_delta_yaw
        )
        steering = (
            self.config.maximum_steering_command_rad
            * ca.tanh(
                raw_steering / self.config.maximum_steering_command_rad
            )
        )
        yaw_rate = (
            state[3] / self.config.wheelbase_m * ca.tan(steering)
        )
        return ca.vertcat(
            state[0]
            + self.config.period_s * state[3] * ca.cos(state[2]),
            state[1]
            + self.config.period_s * state[3] * ca.sin(state[2]),
            state[2] + self.config.period_s * yaw_rate,
            state[3] + control[0],
            state[4]
            + self.config.period_s
            * state[3]
            * ca.cos(state[2] - geometry["yaw"])
            / safe_denominator,
        )

    def _initial_controls(
        self,
        state0: np.ndarray,
        reference_speed_mps: float,
        warm_start_allowed: bool,
    ) -> tuple[np.ndarray, str]:
        if warm_start_allowed and self._previous_controls is not None:
            return (
                np.vstack(
                    (
                        self._previous_controls[1:],
                        self._previous_controls[-1:],
                    )
                ),
                "shifted_warm_start",
            )
        controls = np.zeros((self.N, self.nu))
        states = np.zeros((self.N + 1, self.nx))
        states[0] = state0
        for stage in range(self.N):
            reference = self.reference_path.evaluate(states[stage, 4])
            lateral = (
                -math.sin(float(reference.yaw))
                * (states[stage, 0] - float(reference.x))
                + math.cos(float(reference.yaw))
                * (states[stage, 1] - float(reference.y))
            )
            desired_yaw = float(reference.yaw) - 0.3 * lateral
            controls[stage, 1] = math.atan2(
                math.sin(desired_yaw - states[stage, 2]),
                math.cos(desired_yaw - states[stage, 2]),
            )
            controls[stage, 0] = np.clip(
                reference_speed_mps - states[stage, 3],
                -self.config.delta_speed_scale_mps,
                self.config.delta_speed_scale_mps,
            )
            states[stage + 1] = self.stage_dynamics_numpy(
                states[stage], controls[stage]
            )
        return controls, "cascaded_p_cold_start"

    def _fallback_controls(self) -> np.ndarray:
        if self._previous_controls is None:
            return np.zeros((self.N, self.nu))
        return np.vstack(
            (self._previous_controls[1:], self._previous_controls[-1:])
        )

    def _rollout(
        self, state0: np.ndarray, controls: np.ndarray
    ) -> np.ndarray:
        states = np.zeros((self.N + 1, self.nx))
        states[0] = state0
        for stage in range(self.N):
            states[stage + 1] = self.stage_dynamics_numpy(
                states[stage], controls[stage]
            )
        return states

    def _pack(
        self, states: np.ndarray, controls: np.ndarray
    ) -> np.ndarray:
        return np.r_[
            states.T.reshape(-1, order="F"),
            controls.T.reshape(-1, order="F"),
        ]

    def _unpack(
        self, decision: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        state_end = self.nx * (self.N + 1)
        states = decision[:state_end].reshape(
            (self.nx, self.N + 1), order="F"
        ).T
        controls = decision[state_end:].reshape(
            (self.nu, self.N), order="F"
        ).T
        return states, controls

    def _maximum_dynamics_residual(
        self, decision: np.ndarray, parameters: np.ndarray
    ) -> float:
        residual = np.asarray(
            self._constraint_function(decision, parameters), dtype=float
        ).reshape(-1)
        return float(np.max(np.abs(residual), initial=0.0))

    def _smooth_positive_floor(self, value: float) -> float:
        difference = value - self.config.frenet_minimum_denominator
        return float(
            self.config.frenet_minimum_denominator
            + 0.5 * (difference + math.sqrt(difference**2 + 1e-8))
        )

    def _smooth_steering(self, raw_steering: float) -> float:
        limit = self.config.maximum_steering_command_rad
        return float(limit * math.tanh(raw_steering / limit))
