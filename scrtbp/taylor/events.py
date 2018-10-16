import numpy as np
import numba as nb

from scrtbp.taylor import expansion
from scrtbp.taylor import steppers
from scrtbp.util import root


class NoEventException(Exception):
    pass


def generate_event_observer(StepperClass, FuncAdapterClass, one_way_mode=True):
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
        func=FuncAdapterClass.class_type.instance_type,
    )

    @nb.jitclass(event_observer_spec)
    class EventObserver:
        def __init__(self, stepper):
            self.stepper = stepper
            self.func = FuncAdapterClass(stepper.taylor.series)

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

    return EventObserver


def generate_event_solver(
    taylor_coeff_func,
    poincare_char_func,
    state_dim,
    extra_dim,
    step=0.01,
    order=30,
    max_event_steps=1000000,
    max_steps=1000000000,
    one_way_mode=True,
):
    TaylorExpansion = expansion.generate_taylor_expansion(
        taylor_coeff_func, state_dim, extra_dim
    )
    FuncAdapter = expansion.generate_func_adapter(TaylorExpansion, poincare_char_func)
    Stepper = steppers.generate_fixed_stepper(TaylorExpansion)
    StepLimiterProxy = steppers.generate_step_limter_proxy(Stepper)
    EventObserver = generate_event_observer(Stepper, FuncAdapter, one_way_mode)

    @nb.njit
    def solve_points(input_state, n_points, t0=0.0):
        points = np.empty((n_points, state_dim))
        times = np.empty(n_points)

        stepper = Stepper(input_state, t0, step)
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
            raise steppers.MaxStepsExceededException

        return points, times

    return solve_points
