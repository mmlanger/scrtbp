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
    order=20,
    tol_abs=1e-16,
    tol_rel=1e-10,
    max_event_steps=1000000,
    max_steps=1000000000,
    one_way_mode=True,
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
    max_time,
    order=20,
    tol_abs=1e-16,
    tol_rel=1e-16,
    max_steps=1000000000,
    one_way_mode=False,
):
    state_dim = taylor_params[1]

    TaylorExpansion = expansion.generate_taylor_expansion(*taylor_params)
    Stepper = steppers.generate_adaptive_stepper(TaylorExpansion)
    FuncAdapter = expansion.generate_func_adapter(py_exit_function)

    if one_way_mode:

        @nb.njit
        def root_condition(fa, fb):
            return fa > 0.0 > fb  # only + to - roots are detected

    else:

        @nb.njit
        def root_condition(fa, fb):
            return fa * fb < 0.0

    @nb.njit
    def escape_integration(init_cond, n_points, step, init_t0=0.0):
        state = [init_cond]
        temp_point = np.zeros(state_dim)

        stepper = Stepper(init_cond, init_t0, order, tol_abs, tol_rel)
        exit_func = FuncAdapter(stepper.expansion.series)

        f = exif_func(init_cond)
        while stepper.valid():
            next_t = stepper.next_t
            next_f = exit_func.eval(stepper.step)

            if root_condition(f, next_f):
                pass
            
            if stepper.next_t > max_time:
                if next_ofli > max_ofli:
                    root.Brackets(s)
                    target_step = root.solve_root(
                        func_adapter, 0.0, ofli_stepper.next_t - ofli_stepper.t
                    )
                else:
                    target_step = max_time - ofli_stepper.t
                ofli_val = ofli_stepper.compute_ofli(target_step)
                ofli_time = ofli_stepper.t + target_step
                break

            stepper.advance()

            if exit_condition(ofli_stepper.stepper.expansion.state):
                ofli_val = ofli_stepper.compute_ofli(0.0)
                ofli_time = ofli_stepper.t
                break

            return state, time

        raise exceptions.MaxStepsExceeded

    return escape_integration
