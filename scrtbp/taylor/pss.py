import numpy as np
import numba as nb

import scrtbp.exceptions as exceptions
from scrtbp.util import root

from . import expansion
from . import steppers
from . import events


def generate_adaptive_pss_solver(
    taylor_params,
    event_func,
    py_slice_condition,
    order=20,
    tol_abs=1e-16,
    tol_rel=1e-10,
    max_pss_steps=1000000,
    max_steps=1000000000,
    one_way_mode=True,
):
    TaylorExpansion = expansion.generate_taylor_expansion(*taylor_params)
    _, state_dim, _ = taylor_params
    
    FuncAdapter = expansion.generate_func_adapter(event_func)
    Stepper = steppers.generate_adaptive_stepper(TaylorExpansion)
    StepLimiterProxy = steppers.generate_step_limiter_proxy(Stepper)
    EventObserver = events.generate_event_observer(Stepper, FuncAdapter, one_way_mode)

    slice_condition = nb.njit(py_slice_condition)

    @nb.njit
    def solve_pss(input_state, n_points, t0=0.0):
        points = np.empty((n_points, state_dim))
        times = np.empty(n_points)

        event_state = np.empty(state_dim)
        event_time = t0

        stepper = Stepper(input_state, t0, order, tol_abs, tol_rel)
        observer = EventObserver(stepper)
        limiter = StepLimiterProxy(stepper, max_pss_steps, max_steps)

        i = 0
        while limiter.valid():
            if observer.event_detected():
                event_time = observer.extract_event(event_state)
                if slice_condition(event_state):
                    times[i] = event_time
                    points[i] = event_state
                    i += 1
                    if i < n_points:
                        limiter.reset_constraint()
                    else:
                        break
            limiter.advance()
        else:
            raise exceptions.MaxStepsExceeded

        return points, times

    return solve_pss
