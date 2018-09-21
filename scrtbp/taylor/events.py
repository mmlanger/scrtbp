import numpy as np
import numba as nb

from scrtbp.taylor.expansion import TaylorExpansion, generate_func_adapter
from scrtbp.util import root


class MaxStepsExceededException(Exception):
    pass


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

    root_condition = nb.njit(py_root_condition)

    step_cache_spec = [("state", nb.float64[:]), ("t", nb.float64),
                       ("f", nb.float64)]

    @nb.jitclass(step_cache_spec)
    class EventTracker:
        def __init__(self, state0, t0, f0, expansion):
            self.state = state0
            self.t = t0
            self.f = f0

        def read_from_cache(self, cache):
            self.state = cache.state
            self.t = cache.t
            self.f = chache.f

        def advance(self):
            pass

    @nb.njit
    def advance_chaches(chache_now, chache_next):
        pass

    @nb.njit
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

        state_now = input_state.copy()
        f_now = event_func.eval_from_state(state_now)

        state_next = np.empty(state_dim)

        stepper = FixedStepper(t0, step, max_steps)

        while stepper.valid():
            taylor_coeff_func(state_now, series.coeffs, extra_coeffs)

            f_next = event_func.eval(step)
            state_next = event_func.state_cache

            # searches for root in current step interval t_now, t_next
            if root_condition(f_now, f_next):
                brackets = root.Brackets(0.0, f_now, step, f_next)
                root_step = root.solve(event_func, brackets)

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

            stepper.advance()
            state_now = state_next
        else:
            raise MaxStepsExceededException

        t_next = t0
        f_next = event_func.eval_state(state_next)
        step_num = 0

        for i in range(n_points):
            # compute trajectory steps until intersection is found
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
                    brackets = root.Brackets(0.0, f_now, step, f_next)

                    root_step = root.solve(event_func, brackets)
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

            if j == (max_steps - 1):
                raise MaxStepsExceededException

        return points, t_points

    return solve_points
