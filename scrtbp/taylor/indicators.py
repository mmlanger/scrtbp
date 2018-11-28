import numpy as np
import numba as nb

from scrtbp.util import root

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

        def compute_ofli(self, step):
            expansion = self.stepper.expansion

            if step == 0.0:
                state = expansion.state

                # velocity from RHS of ODE system
                f = expansion.series.coeffs[: self.state_dim, 1]
                f_norm = np.linalg.norm(f)
            else:
                # variation approximation from taylor series
                state = np.empty_like(expansion.state)
                expansion.eval(step, state)

                # velocity approximation from taylor series
                rhs = np.empty_like(expansion.state)
                expansion.tangent(step, rhs)
                f = rhs[: self.state_dim]

            variation = state[self.state_dim :]
            f_norm = np.linalg.norm(f)

            # parallel part of the variation
            var_par = (np.sum(variation * f) / (f_norm * f_norm)) * f

            # norm of the orthogonal part of the variation
            var_ort_norm = np.linalg.norm(variation - var_par)

            # take account of original variation magnitude for ofli computation
            return max(self.ofli, self.log_magn_sum + np.log(var_ort_norm))

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
            self.ofli = self.compute_ofli(0.0)

            # update stepper for correct next step
            expansion.compute()
            self.stepper.advance_time()

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
    taylor_params,
    max_time,
    order=20,
    tol_abs=1e-16,
    tol_rel=0.0,
    max_ofli=None,
    max_event_steps=1000000,
    max_steps=1000000000,
):
    TaylorExpansion = expansion.generate_taylor_expansion(*taylor_params)
    Stepper = steppers.generate_adaptive_stepper(TaylorExpansion)
    OfliProxy = generate_ofli_proxy(Stepper)

    adapter_spec = dict(ofli_stepper=OfliProxy.class_type.instance_type)

    @nb.jitclass(adapter_spec)
    class OfliFuncAdapter:
        def __init__(self, ofli_stepper):
            self.ofli_stepper = ofli_stepper

        def eval(self, delta_t):
            return self.ofli_stepper.compute_ofli(delta_t) - max_ofli

    if max_ofli:

        @nb.njit
        def ofli_integration(init_cond, init_t0=0.0):
            stepper = Stepper(init_cond, init_t0, order, tol_abs, tol_rel)
            ofli_stepper = OfliProxy(stepper)
            ofli_val = 0.0
            ofli_time = 0.0

            while ofli_stepper.valid():
                next_ofli = ofli_stepper.compute_ofli(ofli_stepper.stepper.step)

                if next_ofli > max_ofli:
                    func_adapter = OfliFuncAdapter(ofli_stepper)
                    target_step = root.solve_root(
                        func_adapter, 0.0, ofli_stepper.next_t - ofli_stepper.t
                    )
                    ofli_val = ofli_stepper.compute_ofli(target_step)
                    ofli_time = ofli_stepper.t + target_step
                    break
                elif ofli_stepper.next_t > max_time:
                    target_step = max_time - ofli_stepper.t
                    ofli_val = ofli_stepper.compute_ofli(target_step)
                    ofli_time = ofli_stepper.t + target_step
                    break
                else:
                    ofli_stepper.advance()

            return ofli_val, ofli_time

    else:

        @nb.njit
        def ofli_integration(init_cond, init_t0=0.0):
            stepper = Stepper(init_cond, init_t0, order, tol_abs, tol_rel)
            ofli_stepper = OfliProxy(stepper)
            ofli_val = 0.0
            ofli_time = 0.0

            while ofli_stepper.valid():
                if ofli_stepper.next_t > max_time:
                    target_step = max_time - ofli_stepper.t
                    ofli_time = ofli_stepper.t + target_step
                    ofli_val = ofli_stepper.compute_ofli(target_step)
                    break
                else:
                    ofli_stepper.advance()

            return ofli_val, ofli_time

    return ofli_integration
