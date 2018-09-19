import numpy as np
from numba import njit

from scrtbp.taylor.expansion import TaylorExpansion, generate_func_adapter


def generate_event_solver(taylor_coeff_func,
                          poincare_char_func,
                          state_dim,
                          extra_dim,
                          step=0.01,
                          order=30,
                          max_steps=1000000,
                          one_way_mode=True):
    FuncAdapter = generate_func_adapter(poincare_char_func)

    if one_way_mode:

        def py_root_condition(fa, fb):
            return fa < 0.0 and 0.0 < fb

    else:

        def py_root_condition(fa, fb):
            return fa * fb < 0.0

    root_condition = njit(py_root_condition)

    @njit
    def solve_points(input_state, n_points, t0=0.0):
        """
            solves trajectory step wise, looks if its trajectory 
            crosses the poincare section during a step via 
            root search, saves n_points many section points 
            and their time of section

            n_points = number of points inside poincare section to be calculated
        """
        points = np.empty((n_points, state_dim))
        t_points = np.empty(n_points)

        extra_coeffs = np.empty((extra_dim, order))
        series = TaylorExpansion(np.empty((state_dim, order + 1)))
        event_func = FuncAdapter(series)

        state_now = np.empty(state_dim)
        state_next = np.empty(state_dim)
        brackets = np.empty((2, 2))

        state_next = input_state
        t_next = t0
        f_next = event_func.eval_state(state_next)
        step_num = 0

        for i in range(n_points):
            # calculates trajectory steps until intersection is found
            # (or unfavourably max_steps is reached)
            for j in range(max_steps):
                state_now = state_next
                t_now = t_next
                f_now = f_next

                # compute taylor series around state_now
                taylor_coeff_func(state_now, series.coeffs, extra_coeffs)

                # previously j, which is reset after each new point !
                t_next = t0 + (step_num + 1) * step
                f_next = event_func.eval(step)
                state_next = event_func.state_cache

                step_num = step_num + 1

                # searches for root in current step interval t_now, t_next
                if root_condition(f_now, f_next):
                    # 0 instead of t_now, since taylor stutzstelle already is t_now
                    brackets[0, 0] = 0.0
                    brackets[0, 1] = f_now
                    brackets[1, 0] = step
                    brackets[1, 1] = f_next

                    # bracket size
                    diff_curr = abs(brackets[0, 0] - brackets[1, 0])
                    # hack: just need a worse diff (only used once)
                    diff_prev = 2.0 * diff_curr

                    # until bracket size doesn't change anymore, do ...
                    while diff_prev != diff_curr:
                        root_step = bracket_iteration(event_func, brackets)
                        diff_prev = diff_curr
                        diff_curr = abs(brackets[0, 0] - brackets[1, 0])

                    # uses taylor series around t_now to approximate point[i]
                    series.eval(root_step, points[i])
                    t_points[i] = t_now + root_step
                    break
                elif f_now == 0.0:
                    # immediatly found a root
                    points[i] = state_now
                    t_points[i] = t_now
                    break
                elif f_next == 0.0:
                    # ignore upcoming root, will be caught in next step
                    pass
                else:
                    # no root in interval, continue steps
                    pass
            # previously: if max_steps is exceeded, point[i] will be random and make no sense!
            # if max_steps is exceeded, print error, return points
            # ATTENTION: points from i onward will be random remnants from np.empty!
            if j == (max_steps - 1):
                # raise Exception('ERROR: max_steps exceeded')
                print("ERROR: max_steps ", max_steps, " exceeded")
                return points, t_points

        return points, t_points

    return solve_points
