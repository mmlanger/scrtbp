import numpy as np
import numba as nb

import scrtbp.exceptions as exceptions

from scrtbp.util import root

from . import expansion
from . import steppers


def generate_event_observer(StepperClass, FuncAdapter, one_way_mode=True):
    if one_way_mode:
        # only - to + roots are detected
        def py_root_condition(fa, fb):
            return fa < 0.0 and 0.0 < fb

    else:
        # all sign crossings are detected as roots
        def py_root_condition(fa, fb):
            return fa * fb < 0.0

    root_condition = nb.njit(py_root_condition)

    event_observer_spec = dict(
        stepper=StepperClass.class_type.instance_type,
        func=FuncAdapter.class_type.instance_type,
        t=nb.float64,
        f=nb.float64,
        next_t=nb.float64,
        next_f=nb.float64,
    )

    @nb.jitclass(event_observer_spec)
    class EventObserver:
        def __init__(self, stepper):
            self.stepper = stepper
            self.func = FuncAdapter(stepper.expansion.series)

            self.update()

        def update(self):
            self.t = self.stepper.t
            self.f = self.func.eval_from_state(self.stepper.expansion.state)
            self.next_t = self.stepper.next_t
            self.next_f = self.func.eval(self.stepper.step)

        def cached_update(self):
            if self.next_t == self.stepper.t:
                self.t = self.next_t
                self.f = self.next_f
                self.next_t = self.stepper.next_t
                self.next_f = self.func.eval(self.stepper.step)
            else:
                self.update()

        def event_detected(self):
            if self.t != self.stepper.t:
                self.cached_update()

            return self.f == 0.0 or root_condition(self.f, self.next_f)

        def get_brackets(self):
            return root.Brackets(0.0, self.f, self.stepper.step, self.next_f)

        def resolve_event(self):
            if self.f == 0.0:
                return 0.0
            else:
                brackets = self.get_brackets()
                return root.solve(self.func, brackets)

        def extract_event(self, output):
            if self.f == 0.0:
                output = self.stepper.expansion.state
                return self.t
            else:
                root_step = self.resolve_event()
                self.stepper.expansion.eval(root_step, output)
                return self.t + root_step

    return EventObserver


def generate_event_solver(
    taylor_params,
    event_func,
    step=0.01,
    order=20,
    max_event_steps=1000000,
    max_steps=1000000000,
    one_way_mode=True,
):
    state_dim = taylor_params[1]

    TaylorExpansion = expansion.generate_taylor_expansion(*taylor_params)
    FuncAdapter = expansion.generate_func_adapter(event_func)
    Stepper = steppers.generate_fixed_stepper(TaylorExpansion)
    StepLimiterProxy = steppers.generate_step_limter_proxy(Stepper)
    EventObserver = generate_event_observer(Stepper, FuncAdapter, one_way_mode)

    @nb.njit
    def solve_points(input_state, n_points, t0=0.0):
        points = np.empty((n_points, state_dim))
        times = np.empty(n_points)

        stepper = Stepper(input_state, t0, step, order)
        observer = EventObserver(stepper)
        limiter = StepLimiterProxy(stepper, max_event_steps, max_steps)

        i = 0
        while limiter.valid():
            if observer.event_detected():
                times[i] = observer.extract_event(points[i])
                i += 1
                if i < n_points:
                    limiter.reset_constraint()
                else:
                    break
            limiter.advance()
        else:
            raise exceptions.MaxStepsExceeded

        return points, times

    return solve_points


def generate_adaptive_event_solver(
    taylor_params,
    event_func,
    order=20,
    tol_abs=1e-16,
    tol_rel=1e-10,
    max_event_steps=1000000,
    max_steps=1000000000,
    one_way_mode=True,
):
    state_dim = taylor_params[1]

    TaylorExpansion = expansion.generate_taylor_expansion(*taylor_params)
    FuncAdapter = expansion.generate_func_adapter(event_func)
    Stepper = steppers.generate_adaptive_stepper(TaylorExpansion)
    StepLimiterProxy = steppers.generate_step_limter_proxy(Stepper)
    EventObserver = generate_event_observer(Stepper, FuncAdapter, one_way_mode)

    @nb.njit
    def solve_points(input_state, n_points, t0=0.0):
        points = np.empty((n_points, state_dim))
        times = np.empty(n_points)

        stepper = Stepper(input_state, t0, order, tol_abs, tol_rel)
        observer = EventObserver(stepper)
        limiter = StepLimiterProxy(stepper, max_event_steps, max_steps)

        i = 0
        while limiter.valid():
            if observer.event_detected():
                times[i] = observer.extract_event(points[i])
                i += 1
                if i < n_points:
                    limiter.reset_constraint()
                else:
                    break
            limiter.advance()
        else:
            raise exceptions.MaxStepsExceeded

        return points, times

    return solve_points
