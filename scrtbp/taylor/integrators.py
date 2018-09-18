from . import rules
import numpy as np
from numba import jit


@jit(nopython=True)
def sum_taylor_series(coeffs, delta_t, output):
    state_dim = coeffs.shape[0]

    for k in range(state_dim):
        output[k] = rules.CompHorner(coeffs[k], delta_t)


def generate_fixed_step_integrator(taylor_coeff_func,
                                   state_dim,
                                   extra_dim,
                                   dt,
                                   order=30):
    @jit
    def fixed_step_integration(input_state, times):
        n_points = times.shape[0]

        points = np.empty((n_points, state_dim))
        extra_coeffs = np.empty((extra_dim, order))
        taylor_coeffs = np.empty((state_dim, order + 1))

        points[0] = input_state

        for i in range(n_points - 1):
            taylor_coeff_func(points[i], taylor_coeffs, extra_coeffs)

            # dt = times[i+1] - times[i]
            sum_taylor_series(taylor_coeffs, dt, points[i + 1])

        return points

    return fixed_step_integration


def generate_dense_integrator(taylor_coeff_func,
                              state_dim,
                              extra_dim,
                              step,
                              order=30):
    @jit
    def dense_integration(input_state, times, init_t0=None):
        """calculates phase space points at each t in 'times' with an integration step size for the taylor coeff of 'step'"""
        # times needs to be monotonely increasing!
        # times can be choosen independently of step

        n_points = times.shape[0]

        points = np.empty(
            (n_points,
             state_dim))  # phase space points, will be evaluated at times
        extra_coeffs = np.empty((extra_dim, order))
        taylor_coeffs = np.empty((state_dim, order + 1))

        # checks, if initital time should be first entry of times or init_t0
        if init_t0 is None:
            init_t0 = times[0]

        cur_t = init_t0  # current time step (taylor is expanded around these)
        next_t = cur_t + step

        cur_state = np.empty(
            state_dim
        )  # state at current time step; used to estimate state in [cur_t, next_t] via taylor expansion
        cur_state[:] = input_state

        i = 0  # phase space point counter
        step_num = 1  # time step counter (step_num = 0 is initital time)
        while i < n_points:
            # taylor coeff for current time step
            taylor_coeff_func(cur_state, taylor_coeffs, extra_coeffs)

            # checks for all times in the interval [cur_t, next_t]
            while times[i] < next_t:
                # if time lies on current time step
                if cur_t == times[i]:
                    # current state is a phase space point
                    points[i, :] = cur_state
                    i += 1
                # if time lies in interval
                elif cur_t < times[i]:
                    target_step = times[i] - cur_t
                    # sums up taylor series around t0 = cur_t with dt = target_step
                    for k in range(state_dim):
                        points[i, k] = CompHorner(taylor_coeffs[k], target_step)
                    i += 1

                # if the last time in times lies in the interval [cur_t, next_t], break
                if not i < n_points:
                    break

            # advance a step to next time step
            cur_t = next_t
            # sums up taylor series to get curr_state at new time step
            for k in range(state_dim):
                cur_state[k] = CompHorner(taylor_coeffs[k], step)
            step_num += 1
            next_t = init_t0 + step_num * step  # not next_t = cur_t + step to avoid error

        return points

    return dense_integration
