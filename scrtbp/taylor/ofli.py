import numpy as np
import numba as nb

from .steppers import generate_adaptive_stepper


def generate_ofli_proxy(StepperClass):
    limiter_proxy_spec = dict(
        stepper=StepperClass.class_type.instance_type,
        counter=nb.int64,
        event_limit=nb.int64,
        step_limit=nb.int64,
    )

    @nb.jitclass(limiter_proxy_spec)
    class OfliProxy:
        def __init__(self, variational_stepper):
            self.stepper = variational_stepper
            self.ofli = 0.0
            self.log_magn_sum = 0.0

            self.variation = None

        def compute_ofli(self):
            # velocity from RHS of ODE system
            expansion = self.stepper.expansion
            variation = expansion.state[expansion.state_dim :]
            state_dim = expansion.state_dim // 2
            f = expansion.series.coeffs[:state_dim, 1]
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

        def valid(self):
            return self.stepper.valid()

    return OfliProxy


def H_grad(mu, state):
    """
    calculates normalized gradient of hamiltonian for scrtbp;
    used for initial var_state in ofli
    """
    x = state[0]
    y = state[1]
    z = state[2]
    px = state[3]
    py = state[4]
    pz = state[5]

    grad = np.empty(6)

    r1 = np.sqrt((x + mu) * (x + mu) + y * y + z * z)
    r2 = np.sqrt((x + mu - 1) * (x + mu - 1) + y * y + z * z)

    grad[0] = (
        -py + (1 - mu) * (x + mu) / (r1 * r1 * r1) + mu * (x + mu - 1) / (r2 * r2 * r2)
    )
    grad[1] = px + y * ((1 - mu) / (r1 * r1 * r1) + mu / (r2 * r2 * r2))
    grad[2] = z * ((1 - mu) / (r1 * r1 * r1) + mu / (r2 * r2 * r2))
    grad[3] = px + y
    grad[4] = py - x
    grad[5] = pz

    norm = np.linalg.norm(grad)
    grad = grad / norm

    return grad
