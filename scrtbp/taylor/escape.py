import numpy as np
import numba as nb

import scrtbp.exceptions as exceptions
from scrtbp.util import root

from . import expansion
from . import steppers
from . import events


def generate_poincare_escape_solver(
    taylor_params,
    py_event_func,
    py_escape_condition,
    order: int = 20,
    tol_abs: float = 1e-16,
    tol_rel: float = 1e-10,
    max_event_steps: int = 1000000,
    max_steps: int = 1000000000,
    one_way_mode: bool = True,
):
    TaylorExpansion = expansion.generate_taylor_expansion(*taylor_params)
    _, state_dim, _ = taylor_params

    FuncAdapter = expansion.generate_func_adapter(py_event_func)
    Stepper = steppers.generate_adaptive_stepper(TaylorExpansion)
    StepLimiterProxy = steppers.generate_step_limiter_proxy(Stepper)
    EventObserver = events.generate_event_observer(Stepper, FuncAdapter, one_way_mode)

    escape_condition = nb.njit(py_escape_condition)

    @nb.njit
    def solve_escape(input_state, t0=0.0):
        event_state = np.empty(state_dim)
        event_time = t0

        stepper = Stepper(input_state, t0, order, tol_abs, tol_rel)
        observer = EventObserver(stepper)
        limiter = StepLimiterProxy(stepper, max_event_steps, max_steps)

        while limiter.valid():
            if observer.event_detected():
                event_time = observer.extract_event(event_state)
                if escape_condition(event_state):
                    return event_state, event_time
            limiter.advance()
        else:
            raise exceptions.MaxStepsExceeded

    return solve_escape


def generate_adaptive_escape_solver(
    taylor_params,
    py_exit_function,
    max_time: float,
    order: int = 20,
    tol_abs: float = 1e-16,
    tol_rel: float = 1e-16,
    max_steps: int = 1000000000,
    one_way_mode: bool = False,
):
    TaylorExpansion = expansion.generate_taylor_expansion(*taylor_params)
    Stepper = steppers.generate_adaptive_stepper(TaylorExpansion)
    FuncAdapter = expansion.generate_func_adapter(py_exit_function)
    EventObserver = events.generate_event_observer(Stepper, FuncAdapter, one_way_mode)

    if one_way_mode:

        @nb.njit
        def root_condition(fa, fb):
            return fa > 0.0 > fb  # only + to - roots are detected

    else:

        @nb.njit
        def root_condition(fa, fb):
            return fa * fb < 0.0

    # @nb.njit
    def solve_escape(init_cond, n_points, init_t0=0.0):
        state = init_cond.copy()

        stepper = Stepper(init_cond, init_t0, order, tol_abs, tol_rel)
        observer = EventObserver(stepper)

        i = 0
        while stepper.valid() and i < max_steps:
            if observer.event_detected():
                time = observer.extract_event(state)
                break
            stepper.advance()
            i += 1
        else:
            raise exceptions.MaxStepsExceeded

        return state, time

    return solve_escape
