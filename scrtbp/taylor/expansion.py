import numpy as np
import numba as nb


@nb.njit
def TwoSum(a, b):
    x = a + b
    z = x - a
    y = (a - (x - z)) + (b - z)
    return x, y


@nb.njit
def Split(a):
    z = a * 134217729  # a * (2**r + 1) with r=27
    x = z - (z - a)
    y = a - x
    return x, y


@nb.njit
def TwoProd(a, b):
    x = a * b
    ah, al = Split(a)
    bh, bl = Split(b)
    y = al * bl - (((x - ah * bh) - al * bh) - ah * bl)
    return x, y


@nb.njit
def Horner(poly_coeffs, x):
    n = poly_coeffs.shape[0] - 1
    s = poly_coeffs[n]

    for i in range(n - 1, -1, -1):
        s = s * x + poly_coeffs[i]

    return s


@nb.njit
def CompHorner(poly_coeffs, x):
    n = poly_coeffs.shape[0] - 1
    s = poly_coeffs[n]
    c = 0.0

    for i in range(n - 1, -1, -1):
        p, pi = TwoProd(s, x)
        s, sigma = TwoSum(p, poly_coeffs[i])
        c = c * x + (pi + sigma)

    return s + c


@nb.njit
def CompTaylorTangent(poly_coeffs, x):
    n = poly_coeffs.shape[0] - 1
    s = poly_coeffs[n]
    c = 0.0

    for k in range(n - 2, -1, -1):
        p, pi = TwoProd(s, x)
        s, sigma = TwoSum(p, (k + 1) * poly_coeffs[k + 1])
        c = c * x + (pi + sigma)

    return s + c


@nb.jitclass([('coeffs', nb.float64[:, :])])
class TaylorExpansion:
    def __init__(self, coeffs):
        self.coeffs = coeffs

    @property
    def state_dim(self):
        return self.coeffs.shape[0]

    @property
    def n_coeffs(self):
        return self.coeffs.shape[1]

    @property
    def order(self):
        return self.n_coeffs - 1

    def eval(self, delta_t, output):
        state_dim = output.shape[0]

        for k in range(state_dim):
            output[k] = CompHorner(self.coeffs[k], delta_t)

    def tangent(self, delta_t, output):
        state_dim = output.shape[0]

        for k in range(state_dim):
            output[k] = CompTaylorTangent(self.coeffs[k], delta_t)


def generate_func_adapter(py_func):
    func = nb.njit(py_func)

    @nb.jitclass([('state_cache', nb.float64[:]),
                  ('expansion', TaylorExpansion.class_type.instance_type)])
    class FuncAdapter:
        def __init__(self, expansion):
            self.expansion = expansion
            self.state_cache = np.empty(expansion.state_dim)

        def eval(self, delta_t):
            self.expansion.eval(delta_t, self.state_cache)
            return func(self.state_cache)

        def eval_state(self, state):
            return func(state)

    return FuncAdapter
