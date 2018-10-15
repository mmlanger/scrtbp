from scrtbp.taylor import expansion
import numpy as np
from numba import njit


def generate_fixed_step_integrator(taylor_coeff_func,
                                   state_dim,
                                   extra_dim,
                                   dt,
                                   order=30):
    @njit
    def fixed_step_integration(input_state, times):
        n_points = times.shape[0]

        points = np.empty((n_points, state_dim))
        # TODO?

        return points

    return fixed_step_integration


def generate_dense_integrator(taylor_coeff_func,
                              state_dim,
                              extra_dim,
                              step,
                              order=30):
    @njit
    def dense_integration(input_state, times, init_t0=None):
        """
            calculates phase space points at each t in 'times' with an integration 
            step size for the taylor coeff of 'step'

            - times needs to be monotonely increasing!
            - times can be choosen independently of step
        """
        n_points = times.shape[0]

        # phase space points, will be evaluated at times
        points = np.empty((n_points, state_dim))

        # # coefficient arrays
        # extra_coeffs = np.empty((extra_dim, order))
        # taylor_coeffs = np.empty((state_dim, order + 1))

        # series = expansion.TaylorExpansion(taylor_coeffs)

        # # checks, if initital time should be first entry of times or init_t0
        # if init_t0 is None:
        #     init_t0 = times[0]

        # cur_t = init_t0  # current time step (taylor is expanded around these)
        # next_t = cur_t + step

        # # state at current time step; used to estimate state in [cur_t, next_t] via taylor expansion
        # cur_state = np.empty(state_dim)
        # cur_state = input_state

        # # phase space point counter
        # i = 0

        # # time step counter (step_num = 0 is initital time)
        # step_num = 1

        # while i < n_points:
        #     # taylor coeff for current time step
        #     taylor_coeff_func(cur_state, taylor_coeffs, extra_coeffs)

        #     # checks for all times in the interval [cur_t, next_t]
        #     while times[i] < next_t:
        #         # if time lies on current time step
        #         if cur_t == times[i]:
        #             # current state is a phase space point
        #             points[i, :] = cur_state
        #             i += 1
        #         # if time lies in interval
        #         elif cur_t < times[i]:
        #             target_step = times[i] - cur_t
        #             # sums up taylor series around t0 = cur_t with dt = target_step
        #             series.eval(target_step, points[i])
        #             i += 1

        #         # if the last time in times lies in the interval [cur_t, next_t], break
        #         if not i < n_points:
        #             break

        #     # advance a step to next time step
        #     cur_t = next_t

        #     # sums up taylor series to get curr_state at new time step
        #     series.eval(step, cur_state)
        #     step_num += 1

        #     # FIXME: not next_t = cur_t + step to avoid error
        #     next_t = init_t0 + step_num * step

        return points

    return dense_integration
