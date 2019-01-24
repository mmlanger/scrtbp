import numpy as np
import numba as nb

import scrtbp.exceptions as exceptions
from scrtbp.taylor import expansion
from scrtbp.taylor import steppers


def generate_fixed_step_integrator(taylor_params, step, order=20):
    state_dim = taylor_params[1]

    TaylorExpansion = expansion.generate_taylor_expansion(*taylor_params)
    Stepper = steppers.generate_fixed_stepper(TaylorExpansion)

    @nb.njit
    def fixed_step_integration(init_cond, n_points, init_t0=0.0):
        points = np.empty((n_points, state_dim))
        points[0] = init_cond

        stepper = Stepper(init_cond, init_t0, step, order)

        for i in range(1, n_points):
            stepper.advance()
            points[i] = stepper.expansion.state

        return points

    return fixed_step_integration


def generate_adaptive_dense_integrator(
    taylor_params,
    order=20,
    tol_abs=1e-16,
    tol_rel=1e-16,
    max_intermediate_steps=1000000,
    max_steps=1000000000,
):
    state_dim = taylor_params[1]

    TaylorExpansion = expansion.generate_taylor_expansion(*taylor_params)
    Stepper = steppers.generate_adaptive_stepper(TaylorExpansion)
    StepLimiterProxy = steppers.generate_step_limiter_proxy(Stepper)

    @nb.njit
    def dense_integration(init_cond, times, init_t0=0.0):
        n_points = times.shape[0]
        points = np.empty((n_points, state_dim))

        stepper = Stepper(init_cond, init_t0, order, tol_abs, tol_rel)
        limiter = StepLimiterProxy(stepper, max_intermediate_steps, max_steps)

        if times[0] < init_t0:
            raise ValueError("First time smaller than init_t0!")

        i = 0
        while limiter.valid():
            while times[i] < stepper.next_t:
                # times[i] is in [stepper.t, stepper.next_t)
                if stepper.t == times[i]:
                    points[i] = stepper.expansion.state
                else:
                    target_step = times[i] - stepper.t
                    stepper.expansion.eval(target_step, points[i])

                i += 1
                if i < n_points:
                    limiter.reset_constraint()
                else:
                    return points

            limiter.advance()

        raise exceptions.MaxStepsExceeded

    return dense_integration


def generate_fixed_step_adaptive_integrator(
    taylor_params,
    step,
    order=20,
    tol_abs=1e-16,
    tol_rel=1e-16,
    max_event_steps=1000000,
    max_steps=1000000000,
    py_exit_condition=None,
    array_cast=True,
):
    state_dim = taylor_params[1]

    TaylorExpansion = expansion.generate_taylor_expansion(*taylor_params)
    Stepper = steppers.generate_adaptive_stepper(TaylorExpansion)
    StepLimiterProxy = steppers.generate_step_limiter_proxy(Stepper)

    if py_exit_condition is None:

        @nb.njit
        def exit_condition(state):
            return False

    else:
        exit_condition = nb.njit(py_exit_condition)

    @nb.njit
    def to_array(state_list):
        n_points = len(state_list)
        points = np.empty((n_points, state_dim))

        for i in range(n_points):
            state = state_list[i]
            for j in range(state_dim):
                points[i, j] = state[j]

        return points

    @nb.njit
    def fixed_adaptive_integration(init_cond, n_points, init_t0=0.0):
        points = [init_cond]
        temp_point = np.zeros(state_dim)

        stepper = Stepper(init_cond, init_t0, order, tol_abs, tol_rel)
        limiter = StepLimiterProxy(stepper, max_event_steps, max_steps)

        i = 1
        target_t = init_t0 + step
        while limiter.valid():
            while target_t < stepper.next_t:
                # target_t is in [stepper.t, stepper.next_t)
                if stepper.t == target_t:
                    temp_point[:] = stepper.expansion.state
                else:
                    target_step = target_t - stepper.t
                    stepper.expansion.eval(target_step, temp_point)

                if exit_condition(temp_point):
                    return to_array(points)
                else:
                    points.append(temp_point.copy())
                    i += 1
                    target_t = init_t0 + i * step

                    if i < n_points:
                        limiter.reset_constraint()
                    else:
                        return to_array(points)

            limiter.advance()

        raise exceptions.MaxStepsExceeded

    return fixed_adaptive_integration
