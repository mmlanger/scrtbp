import numpy as np
import numba as nb

import scrtbp.exceptions as exceptions


def generate_fixed_stepper(TaylorExpansionClass):
    fixed_stepper_spec = dict(
        expansion=TaylorExpansionClass.class_type.instance_type,
        step=nb.float64,
        _step_num=nb.int64,
        t0=nb.float64,
        t=nb.float64,
    )

    @nb.jitclass(fixed_stepper_spec)
    class FixedStepper:
        def __init__(self, init_cond, t_init, step, order):
            self.expansion = TaylorExpansionClass(init_cond, order)

            self.step = step
            self._step_num = 0

            self.t0 = t_init
            self.t = t_init

        def advance(self):
            self.expansion.advance(self.step)
            self.advance_time()

        def advance_time(self):
            self.t = self.next_t
            self._step_num += 1

        def valid(self):
            return True

        @property
        def next_t(self):
            return self.t0 + (self._step_num + 1) * self.step

    return FixedStepper


def generate_adaptive_stepper(TaylorExpansionClass):
    fixed_stepper_spec = dict(
        expansion=TaylorExpansionClass.class_type.instance_type,
        rhs_cache=TaylorExpansionClass.class_type.instance_type,
        __step=nb.float64,
        t0=nb.float64,
        __t=nb.float64,
        tol_abs=nb.float64,
        tol_rel=nb.float64,
        safety_fac=nb.float64,
        defect_iters=nb.float64,
        defect_tolfac=nb.float64,
        defect_redfac=nb.float64,
        defect_rhs_diff=nb.float64[:],
    )

    @nb.jitclass(fixed_stepper_spec)
    class AdaptiveStepper:
        def __init__(self, init_cond, t_init, order, tol_abs, tol_rel):
            self.expansion = TaylorExpansionClass(init_cond, order)
            self.rhs_cache = TaylorExpansionClass(init_cond, 2)

            self.__step = 0.0

            self.t0 = t_init
            self.__t = t_init

            self.tol_abs = tol_abs
            self.tol_rel = tol_rel
            self.safety_fac = 0.9

            self.defect_iters = 5
            self.defect_tolfac = 10.0
            self.defect_redfac = 0.8
            self.defect_rhs_diff = np.zeros(self.expansion.state_dim)

        def estimate_step(self):
            n = self.expansion.order
            tol = self.tol

            # step estimation
            sup_norm1 = np.linalg.norm(self.expansion.coeffs[:, n - 1], np.inf)
            estimate1 = (tol / sup_norm1) ** (1.0 / (n - 1))

            sup_norm2 = np.linalg.norm(self.expansion.coeffs[:, n], np.inf)
            estimate2 = (tol / sup_norm2) ** (1.0 / n)

            self.__step = self.safety_fac * min(estimate1, estimate2)

            # defect control
            for _ in range(self.defect_iters):
                self.expansion.tangent(self.__step, self.defect_rhs_diff)

                self.expansion.eval(self.__step, self.rhs_cache.state)
                self.rhs_cache.compute()

                self.defect_rhs_diff -= self.rhs_cache.coeffs[:, 1]
                norm_diff = np.linalg.norm(self.defect_rhs_diff, np.inf)

                if norm_diff > self.defect_tolfac * tol:
                    self.__step = self.defect_redfac * self.__step
                else:
                    return

            raise exceptions.DefectControlFailure

        def advance(self):
            self.expansion.advance(self.step)
            self.advance_time()

        def advance_time(self):
            self.t = self.next_t

        def valid(self):
            return True

        @property
        def t(self):
            return self.__t

        @t.setter
        def t(self, value):
            self.__step = 0.0
            self.__t = value

        @property
        def tol(self):
            state_sup_norm = np.linalg.norm(self.expansion.state, np.inf)
            return self.tol_abs + self.tol_rel * state_sup_norm

        @property
        def step(self):
            if self.__step == 0.0:
                self.estimate_step()
            return self.__step

        @property
        def next_t(self):
            return self.t + self.step

    return AdaptiveStepper


def generate_step_limiter_proxy(StepperClass):
    limiter_proxy_spec = dict(
        stepper=StepperClass.class_type.instance_type,
        step_counter=nb.int64,
        constraint_counter=nb.int64,
        constraint_limit=nb.int64,
        step_limit=nb.int64,
    )

    @nb.jitclass(limiter_proxy_spec)
    class StepLimiterProxy:
        def __init__(self, stepper, constraint_limit, step_limit):
            self.stepper = stepper
            self.step_counter = 0
            self.constraint_counter = 0
            self.constraint_limit = constraint_limit
            self.step_limit = step_limit

        def reset_constraint(self):
            self.constraint_counter = 0

        def advance(self):
            self.stepper.advance()
            self.step_counter += 1
            self.constraint_counter += 1

        def constraint_valid(self):
            return self.constraint_counter < self.constraint_limit

        def steps_valid(self):
            return self.step_counter < self.step_limit

        def valid(self):
            return self.constraint_valid() and self.steps_valid()

    return StepLimiterProxy
