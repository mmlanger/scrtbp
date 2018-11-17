import numpy as np
import numba as nb

from . import steppers


def generate_ofli_proxy(StepperClass):
    ofli_proxy_spec = dict(
        stepper=StepperClass.class_type.instance_type,
        ofli=nb.float64,
        log_magn_sum=nb.float64,
        state_dim=nb.int64,
    )

    # @nb.jitclass(ofli_proxy_spec)
    class OfliProxy:
        def __init__(self, variational_stepper):
            self.stepper = variational_stepper
            self.ofli = 0.0
            self.log_magn_sum = 0.0
            self.state_dim = self.stepper.expansion.state_dim // 2

        def compute_ofli(self):
            expansion = self.stepper.expansion
            variation = expansion.state[self.state_dim :]

            # velocity from RHS of ODE system
            f = expansion.series.coeffs[: self.state_dim, 1]
            f_norm = np.linalg.norm(f)

            # parallel part of the variation
            var_par = (np.dot(variation, f) / (f_norm * f_norm)) * f

            # norm of the orthogonal part of the variation
            var_ort_norm = np.linalg.norm(variation - var_par)

            # take account of original variation magnitude for ofli computation
            self.ofli = max(self.ofli, self.log_magn_sum + np.log(var_ort_norm))

        def advance(self):
            # evaluate next step without computation of taylor coefficients
            expansion = self.stepper.expansion
            step = self.stepper.step
            expansion.eval(step, expansion.state)

            # renormalization of the variation to prevent overflow
            variation = expansion.state[expansion.state_dim :]
            norm = np.linalg.norm(variation)
            variation /= norm

            # but keep track of the log of the original magnitude
            self.log_magn_sum += np.log(norm)

            # compute the coefficients
            expansion.compute()
            self.stepper.advance_time()

        @property
        def t(self):
            return self.stepper.t

        def valid(self):
            return self.stepper.valid()

    return OfliProxy


def generate_dense_integrator(
    taylor_coeff_func,
    state_dim,
    extra_dim,
    max_time,
    order=20,
    max_ofli=None,
    max_event_steps=1000000,
    max_steps=1000000000,
):
    TaylorExpansion = expansion.generate_taylor_expansion(
        taylor_coeff_func, state_dim, extra_dim
    )
    Stepper = steppers.generate_fixed_stepper(TaylorExpansion)
    OfliProxy = generate_ofli_proxy(Stepper)
    StepLimiterProxy = steppers.generate_step_limter_proxy(OfliProxy)

    if max_ofli:

        def stop_constraint(ofli_proxy):
            return ofli_proxy.ofli < max_ofli and ofli_proxy.t < max_time

    else:

        def stop_constraint(ofli_proxy):
            return ofli_proxy.t < max_time

    # @nb.njit
    def ofli_integration(init_cond, init_t0=0.0):
        stepper = Stepper(init_cond, init_t0, step, order)
        ofli_stepper = OfliProxy(stepper)
        # limiter = StepLimiterProxy(ofli_stepper, max_event_steps, max_steps)

        while stop_constraint(ofli_stepper) and ofli_stepper.valid():
            ofli_stepper.advance()

        return ofli_stepper.ofli, ofli_stepper.t

    return ofli_integration
