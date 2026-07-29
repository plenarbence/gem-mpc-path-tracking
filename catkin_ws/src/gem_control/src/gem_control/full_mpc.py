from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import casadi as ca
import numpy as np
import yaml

from gem_control.casadi_reference_path import CasadiReferencePath
from gem_control.learned_dynamics import (
    LearnedDynamics,
    midpoint_pose_step_numpy,
    midpoint_pose_step_symbolic,
    selected_model_directory,
)
from gem_control.reference_path import (
    ClosedReferencePath,
    build_configured_reference_path,
)
from gem_control.timing_compensation import (
    DelayedMpcStart,
    TimedVehicleState,
    prepare_delayed_mpc_start,
)


@dataclass(frozen=True)
class MpcWeights:
    lateral: float = 8.0
    yaw: float = 28.2682787
    progress_speed: float = 2.4
    steering_command_change: float = 0.25908285
    speed_command_change: float = 0.2061432


@dataclass(frozen=True)
class MpcLimits:
    minimum_speed_command_mps: float = 0.0
    maximum_speed_command_mps: float = 5.5
    maximum_steering_command_rad: float = 0.3
    minimum_predicted_speed_mps: float = -0.1
    maximum_predicted_speed_mps: float = 5.56
    maximum_predicted_yaw_rate_radps: float = 1.2
    progress_bound_margin_m: float = 4.0


@dataclass(frozen=True)
class FullMpcConfig:
    period_s: float = 0.1
    horizon_steps: int = 12
    computation_budget_s: float = 0.08
    ipopt_cpu_limit_s: float = 0.065
    ipopt_max_iterations: int = 60
    ipopt_tolerance: float = 1e-4
    ipopt_acceptable_tolerance: float = 1e-3
    reference_speed_mps: float = 5.0
    wheelbase_m: float = 1.75
    speed_command_change_scale_mps: float = 0.2
    steering_command_change_scale_rad: float = 0.06
    path_sample_count: int = 2500
    commissioned_takeover_delay_s: float = 0.005
    maximum_odometry_age_s: float = 0.12
    stationary_solve_count: int = 3
    startup_warmup_cpu_limit_s: float = 0.5
    weights: MpcWeights = field(default_factory=MpcWeights)
    limits: MpcLimits = field(default_factory=MpcLimits)

    def validate(self) -> None:
        if self.period_s <= 0.0:
            raise ValueError("period_s must be positive")
        if self.horizon_steps < 2:
            raise ValueError("horizon_steps must be at least 2")
        if self.computation_budget_s <= 0.0:
            raise ValueError("computation_budget_s must be positive")
        if self.ipopt_max_iterations <= 0:
            raise ValueError("ipopt_max_iterations must be positive")
        if not 0.0 < self.ipopt_cpu_limit_s <= self.computation_budget_s:
            raise ValueError(
                "IPOPT CPU limit must be in the computation budget"
            )
        if not (
            self.limits.minimum_speed_command_mps
            <= self.reference_speed_mps
            <= self.limits.maximum_speed_command_mps
        ):
            raise ValueError("reference speed is outside model command limits")
        if (
            self.speed_command_change_scale_mps <= 0.0
            or self.steering_command_change_scale_rad <= 0.0
        ):
            raise ValueError("command-change scales must be positive")
        if self.commissioned_takeover_delay_s < 0.0:
            raise ValueError("commissioned takeover delay must be non-negative")
        if self.maximum_odometry_age_s <= 0.0:
            raise ValueError("maximum odometry age must be positive")
        if self.stationary_solve_count < 1:
            raise ValueError("stationary solve count must be positive")
        if self.startup_warmup_cpu_limit_s < self.computation_budget_s:
            raise ValueError(
                "startup warmup CPU limit must cover computation budget"
            )


@dataclass(frozen=True)
class FullMpcInitialCondition:
    state: np.ndarray
    fixed_history_z: np.ndarray
    previous_command: np.ndarray
    previous_progress_m: Optional[float] = None


@dataclass(frozen=True)
class FullMpcDiagnostics:
    solver_success: bool
    solution_accepted: bool
    return_status: str
    iteration_count: int
    solve_time_s: float
    total_compute_time_s: float
    computation_budget_s: float
    deadline_enforced: bool
    deadline_met: bool
    warm_start_source: str
    maximum_constraint_violation: float
    objective: float
    fallback_reason: str


@dataclass(frozen=True)
class FullMpcResult:
    success: bool
    first_command: np.ndarray
    controls: np.ndarray
    states: np.ndarray
    progress: np.ndarray
    diagnostics: FullMpcDiagnostics


def load_full_mpc_config(
    path: Path | str | None = None,
) -> FullMpcConfig:
    if path is not None:
        config_path = Path(path)
    else:
        config_path = _control_package_path() / "config" / "full_mpc.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="ascii"))
    ipopt = payload["ipopt"]
    model_limits = payload["model_limits"]
    state_limits = payload["state_limits"]
    weights = payload["weights"]
    scales = payload["command_change_scales"]
    timing = payload["timing"]
    startup = payload["startup"]
    config = FullMpcConfig(
        period_s=float(payload["period_s"]),
        horizon_steps=int(payload["horizon_steps"]),
        computation_budget_s=float(payload["computation_budget_s"]),
        ipopt_cpu_limit_s=float(ipopt["cpu_limit_s"]),
        ipopt_max_iterations=int(ipopt["max_iterations"]),
        ipopt_tolerance=float(ipopt["tolerance"]),
        ipopt_acceptable_tolerance=float(ipopt["acceptable_tolerance"]),
        reference_speed_mps=float(payload["reference_speed_mps"]),
        wheelbase_m=float(payload["wheelbase_m"]),
        speed_command_change_scale_mps=float(scales["speed_mps"]),
        steering_command_change_scale_rad=float(scales["steering_rad"]),
        commissioned_takeover_delay_s=float(
            timing["commissioned_takeover_delay_s"]
        ),
        maximum_odometry_age_s=float(
            timing["maximum_odometry_age_s"]
        ),
        stationary_solve_count=int(startup["stationary_solve_count"]),
        startup_warmup_cpu_limit_s=float(startup["cpu_limit_s"]),
        weights=MpcWeights(
            lateral=float(weights["lateral"]),
            yaw=float(weights["yaw"]),
            progress_speed=float(weights["progress_speed"]),
            steering_command_change=float(
                weights["steering_command_change"]
            ),
            speed_command_change=float(weights["speed_command_change"]),
        ),
        limits=MpcLimits(
            minimum_speed_command_mps=float(
                model_limits["minimum_speed_command_mps"]
            ),
            maximum_speed_command_mps=float(
                model_limits["maximum_speed_command_mps"]
            ),
            maximum_steering_command_rad=float(
                model_limits["maximum_steering_command_rad"]
            ),
            minimum_predicted_speed_mps=float(
                state_limits["minimum_predicted_speed_mps"]
            ),
            maximum_predicted_speed_mps=float(
                state_limits["maximum_predicted_speed_mps"]
            ),
            maximum_predicted_yaw_rate_radps=float(
                state_limits["maximum_predicted_yaw_rate_radps"]
            ),
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


class FullLearnedMpc:
    """Direct-command nonlinear MPC using the frozen learned H2 model."""

    _instance_ids = itertools.count()

    def __init__(
        self,
        config: FullMpcConfig | None = None,
        model: LearnedDynamics | None = None,
        reference_path: ClosedReferencePath | None = None,
    ) -> None:
        self.config = config or load_full_mpc_config()
        self.config.validate()
        self.model = model or LearnedDynamics(selected_model_directory())
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
        self._previous_progress: np.ndarray | None = None
        build_start = time.perf_counter()
        self._build_solver()
        self.solver_construction_time_s = time.perf_counter() - build_start

    def reset_warm_start(self) -> None:
        self._previous_controls = None
        self._previous_progress = None

    def prepare_delayed_initial_condition(
        self,
        *,
        odometry: TimedVehicleState,
        command_publish_timestamp_s: float,
        applied_history_z: np.ndarray,
        previous_progress_m: float | None = None,
    ) -> tuple[FullMpcInitialCondition, DelayedMpcStart]:
        delayed = prepare_delayed_mpc_start(
            model=self.model,
            odometry=odometry,
            command_publish_timestamp_s=command_publish_timestamp_s,
            commissioned_takeover_delay_s=(
                self.config.commissioned_takeover_delay_s
            ),
            controller_period_s=self.config.period_s,
            applied_history_z=applied_history_z,
            maximum_odometry_age_s=self.config.maximum_odometry_age_s,
        )
        return (
            FullMpcInitialCondition(
                state=delayed.predicted_state,
                fixed_history_z=delayed.fixed_history_z,
                previous_command=delayed.active_command,
                previous_progress_m=previous_progress_m,
            ),
            delayed,
        )

    @property
    def decision_variable_count(self) -> int:
        return self.nx * (self.N + 1) + self.nu * self.N + self.N + 1

    def solve(
        self,
        initial_condition: FullMpcInitialCondition,
        reference_speed_mps: float | None = None,
        *,
        enforce_deadline: bool = True,
    ) -> FullMpcResult:
        total_start = time.perf_counter()
        state0, history, previous_command = self._validate_initial_condition(
            initial_condition
        )
        reference_speed = (
            self.config.reference_speed_mps
            if reference_speed_mps is None
            else float(reference_speed_mps)
        )
        if not (
            self.config.limits.minimum_speed_command_mps
            <= reference_speed
            <= self.config.limits.maximum_speed_command_mps
        ):
            raise ValueError("reference speed is outside command limits")
        if initial_condition.previous_progress_m is None:
            projection = self.reference_path.project(state0[0], state0[1])
        else:
            projection = self.reference_path.project_local(
                state0[0],
                state0[1],
                initial_condition.previous_progress_m,
            )
        s0 = float(projection.s)

        controls, progress, warm_source = self._initial_guess(
            s0, reference_speed
        )
        states = self._rollout_states(state0, controls, history)
        x0 = self._pack(states, controls, progress)
        lbx, ubx = self._variable_bounds(state0, s0, progress)
        parameters = np.r_[history.reshape(-1), reference_speed, previous_command]

        solver_start = time.perf_counter()
        solver_success = False
        return_status = "not_run"
        iteration_count = -1
        fallback_reason = ""
        try:
            active_solver = (
                self.solver if enforce_deadline else self.warmup_solver
            )
            solution = active_solver(
                x0=x0,
                lbx=lbx,
                ubx=ubx,
                lbg=self.lbg,
                ubg=self.ubg,
                p=parameters,
            )
            solver_elapsed = time.perf_counter() - solver_start
            stats = active_solver.stats()
            solver_success = bool(stats.get("success", False))
            return_status = str(stats.get("return_status", "unknown"))
            iteration_count = int(stats.get("iter_count", -1))
            candidate = np.asarray(solution["x"], dtype=float).reshape(-1)
            objective = float(solution["f"])
        except RuntimeError as error:
            solver_elapsed = time.perf_counter() - solver_start
            candidate = x0
            objective = float(self.objective_function(x0, parameters))
            return_status = "solver_exception: {}".format(error)
            fallback_reason = return_status

        violation = self._maximum_constraint_violation(
            candidate, parameters, lbx, ubx
        )
        total_elapsed = time.perf_counter() - total_start
        deadline_met = (
            total_elapsed <= self.config.computation_budget_s
        )
        accepted_status = return_status in (
            "Solve_Succeeded",
            "Solved_To_Acceptable_Level",
        )
        feasible_time_limit = (
            return_status == "Maximum_CpuTime_Exceeded"
        )
        accepted = (
            ((solver_success and accepted_status) or feasible_time_limit)
            and (deadline_met or not enforce_deadline)
            and np.isfinite(candidate).all()
            and np.isfinite(objective)
            and violation <= 2e-3
        )
        if not accepted:
            if not fallback_reason:
                if enforce_deadline and not deadline_met:
                    fallback_reason = "computation_deadline_missed"
                elif not solver_success or not accepted_status:
                    fallback_reason = "solver_not_successful"
                elif violation > 2e-3:
                    fallback_reason = "constraint_violation"
                else:
                    fallback_reason = "nonfinite_solution"
            candidate = x0
            objective = float(self.objective_function(x0, parameters))

        states_out, controls_out, progress_out = self._unpack(candidate)
        if accepted:
            self._previous_controls = controls_out.copy()
            self._previous_progress = progress_out.copy()
        first_command = (
            controls_out[0].copy()
            if accepted
            else previous_command.copy()
        )
        diagnostics = FullMpcDiagnostics(
            solver_success=solver_success,
            solution_accepted=accepted,
            return_status=return_status,
            iteration_count=iteration_count,
            solve_time_s=float(solver_elapsed),
            total_compute_time_s=float(total_elapsed),
            computation_budget_s=self.config.computation_budget_s,
            deadline_enforced=enforce_deadline,
            deadline_met=deadline_met,
            warm_start_source=warm_source,
            maximum_constraint_violation=violation,
            objective=objective,
            fallback_reason=fallback_reason,
        )
        return FullMpcResult(
            success=accepted,
            first_command=first_command,
            controls=controls_out,
            states=states_out,
            progress=progress_out,
            diagnostics=diagnostics,
        )

    def stationary_warmup(
        self,
        initial_condition: FullMpcInitialCondition,
        solve_count: int | None = None,
    ) -> list[FullMpcDiagnostics]:
        solve_count = (
            self.config.stationary_solve_count
            if solve_count is None
            else solve_count
        )
        if solve_count < 1:
            raise ValueError("solve_count must be positive")
        diagnostics = []
        for _ in range(solve_count):
            result = self.solve(
                initial_condition,
                reference_speed_mps=0.0,
                enforce_deadline=False,
            )
            diagnostics.append(result.diagnostics)
        return diagnostics

    def _build_solver(self) -> None:
        suffix = str(next(self._instance_ids))
        X = ca.MX.sym("X", self.nx, self.N + 1)
        U = ca.MX.sym("U", self.nu, self.N)
        S = ca.MX.sym("S", self.N + 1)
        P = ca.MX.sym("P", 11)
        fixed_history = ca.reshape(P[0:8], 4, 2).T
        reference_speed = P[8]
        previous_command = P[9:11]

        constraints = []
        objective = 0
        for k in range(self.N):
            feature_history = self._feature_history_symbolic(
                X, U, fixed_history, k
            )
            next_dynamic = self.model.predict_next_state_symbolic(
                feature_history
            )
            next_pose = midpoint_pose_step_symbolic(
                X[0:3, k],
                X[3:5, k],
                next_dynamic,
                self.config.period_s,
            )
            constraints.append(
                X[:, k + 1] - ca.vertcat(next_pose, next_dynamic)
            )

            prior_command = previous_command if k == 0 else U[:, k - 1]
            speed_change = (
                U[0, k] - prior_command[0]
            ) / self.config.speed_command_change_scale_mps
            steering_change = (
                U[1, k] - prior_command[1]
            ) / self.config.steering_command_change_scale_rad
            objective += (
                self.config.weights.speed_command_change * speed_change**2
                + self.config.weights.steering_command_change
                * steering_change**2
            )

        for k in range(self.N + 1):
            reference = self.casadi_path.evaluate_symbolic(S[k])
            dx = X[0, k] - reference["x"]
            dy = X[1, k] - reference["y"]
            projection_residual = (
                reference["tangent_x"] * dx
                + reference["tangent_y"] * dy
            )
            constraints.append(projection_residual)
            if k > 0:
                lateral = (
                    reference["normal_x"] * dx
                    + reference["normal_y"] * dy
                )
                yaw_difference = X[2, k] - reference["yaw"]
                progress_speed = (S[k] - S[k - 1]) / self.config.period_s
                objective += (
                    self.config.weights.lateral * lateral**2
                    + self.config.weights.yaw
                    * (1.0 - ca.cos(yaw_difference))
                    + self.config.weights.progress_speed
                    * (progress_speed - reference_speed) ** 2
                )
                constraints.append(S[k] - S[k - 1])

        variables = ca.vertcat(
            ca.reshape(X, -1, 1),
            ca.reshape(U, -1, 1),
            S,
        )
        constraint_vector = ca.vertcat(*constraints)
        options = {
            "ipopt.print_level": 0,
            "print_time": False,
            "ipopt.max_iter": self.config.ipopt_max_iterations,
            "ipopt.max_cpu_time": self.config.ipopt_cpu_limit_s,
            "ipopt.tol": self.config.ipopt_tolerance,
            "ipopt.acceptable_tol": self.config.ipopt_acceptable_tolerance,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.mu_strategy": "adaptive",
        }
        self.solver = ca.nlpsol(
            "full_learned_mpc_" + suffix,
            "ipopt",
            {"x": variables, "f": objective, "g": constraint_vector, "p": P},
            options,
        )
        warmup_options = dict(options)
        warmup_options["ipopt.max_cpu_time"] = (
            self.config.startup_warmup_cpu_limit_s
        )
        self.warmup_solver = ca.nlpsol(
            "full_learned_mpc_warmup_" + suffix,
            "ipopt",
            {"x": variables, "f": objective, "g": constraint_vector, "p": P},
            warmup_options,
        )
        self.objective_function = ca.Function(
            "full_mpc_objective_" + suffix,
            [variables, P],
            [objective],
        )
        self.constraint_function = ca.Function(
            "full_mpc_constraints_" + suffix,
            [variables, P],
            [constraint_vector],
        )

        lbg = []
        ubg = []
        for _ in range(self.N):
            lbg.extend([0.0] * self.nx)
            ubg.extend([0.0] * self.nx)
        for k in range(self.N + 1):
            lbg.append(0.0)
            ubg.append(0.0)
            if k > 0:
                lbg.append(0.0)
                ubg.append(np.inf)
        self.lbg = np.asarray(lbg)
        self.ubg = np.asarray(ubg)

    def _feature_history_symbolic(self, X, U, fixed_history, k: int):
        row0 = ca.horzcat(
            X[3, k], X[4, k], U[0, k], U[1, k]
        )
        if k == 0:
            row1 = fixed_history[0, :]
            row2 = fixed_history[1, :]
        elif k == 1:
            row1 = ca.horzcat(
                X[3, 0], X[4, 0], U[0, 0], U[1, 0]
            )
            row2 = fixed_history[0, :]
        else:
            row1 = ca.horzcat(
                X[3, k - 1],
                X[4, k - 1],
                U[0, k - 1],
                U[1, k - 1],
            )
            row2 = ca.horzcat(
                X[3, k - 2],
                X[4, k - 2],
                U[0, k - 2],
                U[1, k - 2],
            )
        return ca.vertcat(row0, row1, row2)

    def _feature_history_numpy(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        fixed_history: np.ndarray,
        k: int,
    ) -> np.ndarray:
        row0 = np.r_[states[k, 3:5], controls[k]]
        if k == 0:
            return np.vstack((row0, fixed_history[0], fixed_history[1]))
        row1 = np.r_[states[k - 1, 3:5], controls[k - 1]]
        row2 = (
            fixed_history[0]
            if k == 1
            else np.r_[states[k - 2, 3:5], controls[k - 2]]
        )
        return np.vstack((row0, row1, row2))

    def _rollout_states(
        self,
        state0: np.ndarray,
        controls: np.ndarray,
        fixed_history: np.ndarray,
    ) -> np.ndarray:
        states = np.zeros((self.N + 1, self.nx))
        states[0] = state0
        for k in range(self.N):
            feature = self._feature_history_numpy(
                states, controls, fixed_history, k
            )
            next_dynamic = self.model.predict_next_state_numpy(feature)
            states[k + 1, 0:3] = midpoint_pose_step_numpy(
                states[k, 0:3],
                states[k, 3:5],
                next_dynamic,
                self.config.period_s,
            )
            states[k + 1, 3:5] = next_dynamic
        return states

    def _initial_guess(
        self, start_progress: float, reference_speed: float
    ) -> tuple[np.ndarray, np.ndarray, str]:
        if (
            self._previous_controls is not None
            and self._previous_progress is not None
        ):
            controls = np.vstack(
                (
                    self._previous_controls[1:],
                    self._previous_controls[-1],
                )
            )
            final_step = max(
                0.0,
                self._previous_progress[-1]
                - self._previous_progress[-2],
            )
            progress = np.r_[
                self._previous_progress[1:],
                self._previous_progress[-1] + final_step,
            ]
            progress += start_progress - progress[0]
            progress[0] = start_progress
            return (
                controls,
                np.maximum.accumulate(progress),
                "shifted_previous_solution",
            )
        progress = (
            start_progress
            + self.config.period_s
            * reference_speed
            * np.arange(self.N + 1)
        )
        reference = self.reference_path.evaluate(progress[:-1])
        steering = np.arctan(
            self.config.wheelbase_m * reference.curvature
        )
        limits = self.config.limits
        controls = np.column_stack(
            (
                np.full(self.N, reference_speed),
                np.clip(
                    steering,
                    -limits.maximum_steering_command_rad,
                    limits.maximum_steering_command_rad,
                ),
            )
        )
        return controls, progress, "path_feedforward_cold_start"

    def _validate_initial_condition(
        self, initial: FullMpcInitialCondition
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        state = np.asarray(initial.state, dtype=float)
        history = np.asarray(initial.fixed_history_z, dtype=float)
        previous = np.asarray(initial.previous_command, dtype=float)
        if state.shape != (5,) or not np.isfinite(state).all():
            raise ValueError("state must be finite [x, y, yaw, speed, yaw_rate]")
        if history.shape != (2, 4) or not np.isfinite(history).all():
            raise ValueError("fixed_history_z must be finite with shape (2, 4)")
        if previous.shape != (2,) or not np.isfinite(previous).all():
            raise ValueError("previous_command must be finite with shape (2,)")
        return state, history, previous

    def _variable_bounds(
        self,
        state0: np.ndarray,
        s0: float,
        progress_guess: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        lower = np.full(self.decision_variable_count, -np.inf)
        upper = np.full(self.decision_variable_count, np.inf)
        state_end = self.nx * (self.N + 1)
        control_end = state_end + self.nu * self.N

        state_lower = np.full((self.nx, self.N + 1), -np.inf)
        state_upper = np.full((self.nx, self.N + 1), np.inf)
        limits = self.config.limits
        state_lower[3, :] = limits.minimum_predicted_speed_mps
        state_upper[3, :] = limits.maximum_predicted_speed_mps
        state_lower[4, :] = -limits.maximum_predicted_yaw_rate_radps
        state_upper[4, :] = limits.maximum_predicted_yaw_rate_radps
        state_lower[:, 0] = state0
        state_upper[:, 0] = state0

        control_lower = np.vstack(
            (
                np.full(self.N, limits.minimum_speed_command_mps),
                np.full(self.N, -limits.maximum_steering_command_rad),
            )
        )
        control_upper = np.vstack(
            (
                np.full(self.N, limits.maximum_speed_command_mps),
                np.full(self.N, limits.maximum_steering_command_rad),
            )
        )
        progress_lower = (
            progress_guess - limits.progress_bound_margin_m
        )
        progress_upper = (
            progress_guess + limits.progress_bound_margin_m
        )
        progress_lower[0] = s0
        progress_upper[0] = s0

        lower[:state_end] = state_lower.reshape(-1, order="F")
        upper[:state_end] = state_upper.reshape(-1, order="F")
        lower[state_end:control_end] = control_lower.reshape(
            -1, order="F"
        )
        upper[state_end:control_end] = control_upper.reshape(
            -1, order="F"
        )
        lower[control_end:] = progress_lower
        upper[control_end:] = progress_upper
        return lower, upper

    def _pack(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        progress: np.ndarray,
    ) -> np.ndarray:
        return np.r_[
            states.T.reshape(-1, order="F"),
            controls.T.reshape(-1, order="F"),
            progress,
        ]

    def _unpack(
        self, variables: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        state_end = self.nx * (self.N + 1)
        control_end = state_end + self.nu * self.N
        states = variables[:state_end].reshape(
            (self.nx, self.N + 1), order="F"
        ).T
        controls = variables[state_end:control_end].reshape(
            (self.nu, self.N), order="F"
        ).T
        return states, controls, variables[control_end:]

    def _maximum_constraint_violation(
        self,
        variables: np.ndarray,
        parameters: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> float:
        constraints = np.asarray(
            self.constraint_function(variables, parameters),
            dtype=float,
        ).reshape(-1)
        constraint_violation = np.maximum(
            self.lbg - constraints, 0.0
        ) + np.maximum(constraints - self.ubg, 0.0)
        bound_violation = np.maximum(
            lower - variables, 0.0
        ) + np.maximum(variables - upper, 0.0)
        return float(
            max(
                np.max(constraint_violation, initial=0.0),
                np.max(bound_violation, initial=0.0),
            )
        )
