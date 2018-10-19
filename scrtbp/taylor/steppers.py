import numba as nb


class MaxStepsExceededException(Exception):
    pass


def generate_fixed_stepper(TaylorExpansionClass):
    fixed_stepper_spec = dict(
        expansion=TaylorExpansionClass.class_type.instance_type,
        step=nb.float64,
        step_num=nb.int64,
        t0=nb.float64,
        t=nb.float64,
        order=nb.int64,
    )

    @nb.jitclass(fixed_stepper_spec)
    class FixedStepper:
        def __init__(self, init_cond, t_init, step, order):
            self.expansion = TaylorExpansionClass(init_cond, order)

            self.step = step
            self.step_num = 0

            self.t0 = t_init
            self.t = t_init

        def advance(self):
            self.expansion.advance(self.step)
            self.t = self.next_t
            self.step_num += 1

        def valid(self):
            return True

        @property
        def next_t(self):
            return self.t0 + (self.step_num + 1) * self.step

    return FixedStepper


def generate_step_limter_proxy(StepperClass):
    limiter_proxy_spec = dict(
        stepper=StepperClass.class_type.instance_type,
        counter=nb.int64,
        event_limit=nb.int64,
        step_limit=nb.int64,
    )

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

    return StepLimiterProxy
