import numpy as np
import numba as nb

from . import steppers
from . import expansion


def generate_ofli_proxy(StepperClass):
    ofli_proxy_spec = dict(
        stepper=StepperClass.class_type.instance_type,
        ofli=nb.float64,
        log_magn_sum=nb.float64,
        state_dim=nb.int64,
    )

    @nb.jitclass(ofli_proxy_spec)
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
            var_par = (np.sum(variation * f) / (f_norm * f_norm)) * f

            # norm of the orthogonal part of the variation
            var_ort_norm = np.linalg.norm(variation - var_par)

            # take account of original variation magnitude for ofli computation
            self.ofli = max(self.ofli, self.log_magn_sum + np.log(var_ort_norm))

        def advance(self):
            # evaluate next step without computation of taylor coefficients
            expansion = self.stepper.expansion
            expansion.eval(self.stepper.step, expansion.state)

            # renormalization of the variation to prevent overflow
            variation = expansion.state[self.state_dim :]
            norm = np.linalg.norm(variation)
            variation /= norm

            # but keep track of the log of the original magnitude
            self.log_magn_sum += np.log(norm)

            # update ofli
            self.compute_ofli()

            # update stepper for correct next step
            expansion.compute()
            self.stepper.t = self.stepper.next_t

        def force_step(self, step):
            # evaluate next step without computation of taylor coefficients
            expansion = self.stepper.expansion
            expansion.eval(step, expansion.state)

            self.compute_ofli()

            expansion.compute()
            self.stepper.t += step

        @property
        def t(self):
            return self.stepper.t

        @property
        def next_t(self):
            return self.stepper.next_t

        def valid(self):
            return self.stepper.valid()

    return OfliProxy


def generate_ofli_integrator(
    taylor_coeff_func,
    state_dim,
    extra_dim,
    max_time,
    order=20,
    tol_abs=1e-16,
    tol_rel=0.0,
    max_ofli=None,
    max_event_steps=1000000,
    max_steps=1000000000,
):
    TaylorExpansion = expansion.generate_taylor_expansion(
        taylor_coeff_func, state_dim, extra_dim
    )
    Stepper = steppers.generate_adaptive_stepper(TaylorExpansion)
    OfliProxy = generate_ofli_proxy(Stepper)

    if max_ofli:

        @nb.njit
        def stop_constraint(ofli_proxy):
            return ofli_proxy.ofli >= max_ofli

    else:

        @nb.njit
        def stop_constraint(ofli_proxy):
            return False

    @nb.njit
    def ofli_integration(init_cond, init_t0=0.0):
        stepper = Stepper(init_cond, init_t0, order, tol_abs, tol_rel)
        ofli_stepper = OfliProxy(stepper)

        while not stop_constraint(ofli_stepper) and ofli_stepper.valid():
            if ofli_stepper.next_t > max_time:
                ofli_stepper.force_step(max_time - ofli_stepper.t)
                break
            else:
                ofli_stepper.advance()

        return ofli_stepper.ofli, ofli_stepper.t

    return ofli_integration
