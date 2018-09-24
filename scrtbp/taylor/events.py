import numpy as np
import numba as nb

from scrtbp.taylor.expansion import TaylorExpansion, generate_func_adapter
from scrtbp.util import root


class MaxStepsExceededException(Exception):
    pass


class NoEventException(Exception):
    pass


def generate_event_solver(taylor_coeff_func,
                          poincare_char_func,
                          state_dim,
                          extra_dim,
                          step=0.01,
                          order=30,
                          max_event_steps=1000000,
                          max_steps=1000000000,
                          one_way_mode=True):
    FuncAdapter = generate_func_adapter(poincare_char_func)

    if one_way_mode:

        def py_root_condition(fa, fb):
            return fa < 0.0 and 0.0 < fb
    else:

        def py_root_condition(fa, fb):
            return fa * fb < 0.0

    root_condition = nb.njit(py_root_condition)

    taylor_adapter_spec = dict(
        state=nb.float64[:],
        extra_coeffs=nb.float64[:, :],
        coeffs=nb.float64[:, :],
        series=TaylorExpansion.class_type.instance_type)

    @nb.jitclass(taylor_adapter_spec)
    class TaylorAdapter:
        def __init__(self, state):
            self.state = state
            self.extra_coeffs = np.empty((extra_dim, order))
            self.coeffs = np.empty((state_dim, order + 1))
            self.series = TaylorExpansion(self.coeffs)

        def set_state(self, state):
            self.state = state
            self.compute()

        def advance(self, step):
            self.eval(step, self.state)
            self.compute()

        def compute(self):
            taylor_coeff_func(self.state, self.coeffs, self.extra_coeffs)

        def eval(self, delta_t, output):
            self.series.eval(delta_t, output)

        def tangent(self, delta_t, output):
            self.series.tangent(delta_t, output)

    fixed_stepper_spec = dict(
        taylor=TaylorAdapter.class_type.instance_type,
        step=nb.float64,
        step_num=nb.int64,
        t0=nb.float64,
        t=nb.float64)

    @nb.jitclass(fixed_stepper_spec)
    class FixedStepper:
        def __init__(self, state_init, t_init, step):
            self.taylor = TaylorAdapter(state_init)

            self.step = step
            self.step_num = 0

            self.t0 = t_init
            self.t = t_init

        def advance(self):
            self.taylor.advance(self.step)
            self.step_num += 1

        def valid(self):
            return True

        @property
        def next_t(self):
            return self.t0 + (self.step_num + 1) * self.step

    event_observer_spec = dict(
        stepper=FixedStepper.class_type.instance_type,
        func=FuncAdapter.class_type.instance_type)

    @nb.jitclass(event_observer_spec)
    class EventObserver:
        def __init__(self, stepper):
            self.stepper = stepper
            self.func = FuncAdapter(stepper.taylor.series)

            self.update()

        def update(self):
            self.t = self.stepper.t
            self.f = self.func.eval_from_state(self.stepper.state)
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
                output = self.stepper.taylor.state
                return self.stepper.t
            else:
                root_step = self.resolve_event()
                self.stepper.taylor.eval(root_step, output)
                return self.stepper.t + root_step

    limiter_proxy_spec = dict(
        stepper=FixedStepper.class_type.instance_type,
        counter=nb.int64,
        event_limit=nb.int64,
        step_limit=nb.int64)

    @nb.jitclass(limiter_proxy_spec)
    class StepLimiterProxy:
        def __init__(self, stepper, event_limit, step_limit):
            self.stepper = stepper
            self.counter = 0
            self.event_limit = event_limit
            self.step_limit = step_limit

        def reset_constraint(self):
            self.counter = 0

        def advance(self):
            self.stepper.advance()
            self.counter += 1

        def valid(self):
            event_steps_valid = self.counter < self.event_limit
            max_steps_valid = self.stepper.step_num < self.step_limit

            return event_steps_valid and max_steps_valid

    @nb.njit
    def solve_points(input_state, n_points, t0=0.0):
        points = np.empty((n_points, state_dim))
        times = np.empty(n_points)

        stepper = FixedStepper(input_state, t0, step)
        observer = EventObserver(stepper)
        limiter = StepLimiterProxy(stepper, max_event_steps, max_steps)

        i = 0
        while limiter.valid():
            if observer.event_detected():
                i += 1
                times[i] = observer.extract_event(points[i])
                if (i + 1) < n_points:
                    limiter.reset_constraint()
                else:
                    break
            limiter.advance()
        else:
            raise MaxStepsExceededException

        return points, times

    return solve_points
