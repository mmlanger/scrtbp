import numpy as np
import numba as nb

from scrtbp.taylor import expansion
from scrtbp.taylor import steppers


def generate_fixed_step_integrator(
    taylor_coeff_func, state_dim, extra_dim, step, order=30
):
    TaylorExpansion = expansion.generate_taylor_expansion(
        taylor_coeff_func, state_dim, extra_dim
    )
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


def generate_dense_integrator(
    taylor_coeff_func,
    state_dim,
    extra_dim,
    step,
    order=30,
    max_event_steps=1000000,
    max_steps=1000000000,
):
    TaylorExpansion = expansion.generate_taylor_expansion(
        taylor_coeff_func, state_dim, extra_dim
    )
    Stepper = steppers.generate_fixed_stepper(TaylorExpansion)
    StepLimiterProxy = steppers.generate_step_limter_proxy(Stepper)

    @nb.njit
    def dense_integration(init_cond, times, init_t0=0.0):
        n_points = times.shape[0]
        points = np.empty((n_points, state_dim))

        stepper = Stepper(init_cond, init_t0, step, order)
        limiter = StepLimiterProxy(stepper, max_event_steps, max_steps)

        if times[0] < init_t0:
            return ValueError("First time smaller than init_t0!")

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
                    break

            if not i < n_points:
                break

            limiter.advance()

        return points

    return dense_integration
