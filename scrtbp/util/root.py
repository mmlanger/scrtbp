import numpy as np
import numba as nb

from scrtbp.exceptions import InvalidBrackets


bracket_spec = dict(
    left_x=nb.float64, left_fx=nb.float64, right_x=nb.float64, right_fx=nb.float64
)


@nb.jitclass(bracket_spec)
class Brackets:
    def __init__(self, x1, f1, x2, f2):
        if f1 == 0.0:
            self.left_x = x1
            self.left_fx = f1
            self.right_x = x1
            self.right_fx = f1
        elif f2 == 0.0:
            self.left_x = x2
            self.left_fx = f2
            self.right_x = x2
            self.right_fx = f2
        elif f1 * f2 > 0.0:
            msg = (
                "Interval doesn't contain a root!"
                "I.e. f(a) < 0 < f(b) or f(a) > 0 > f(b)"
            )
            raise InvalidBrackets(msg)
        else:
            if x1 <= x2:
                self.left_x = x1
                self.left_fx = f1
                self.right_x = x2
                self.right_fx = f2
            else:
                self.left_x = x2
                self.left_fx = f2
                self.right_x = x1
                self.right_fx = f1

    @property
    def width(self):
        return abs(self.right_x - self.left_x)

    @property
    def contains_exact_root(self):
        return self.left_fx == 0.0 or self.right_fx == 0.0

    @property
    def best_approx_root(self):
        if abs(self.left_fx) < abs(self.right_fx):
            return self.left_x
        else:
            return self.right_x

    def is_inside(self, x):
        return self.left_x < x and x < self.right_x

    def update(self, x, fx):
        if self.is_inside(x):
            if self.left_fx * fx < 0.0:
                self.right_x = x
                self.right_fx = fx
            elif fx * self.right_fx < 0.0:
                self.left_x = x
                self.left_fx = fx
            elif fx == 0.0:
                self.left_x = x
                self.left_fx = fx
                self.right_x = x
                self.right_fx = fx
            else:
                msg = "Inconsistent bracket state!"
                raise InvalidBrackets(msg)


@nb.njit
def bracket_iteration(func_adapter, brackets):
    x1 = brackets.left_x
    f1 = brackets.left_fx
    x3 = brackets.right_x
    f3 = brackets.right_fx

    # bisection for additional point
    x2 = (x1 + x3) / 2.0
    f2 = func_adapter.eval(x2)

    # update brackets and check for root
    brackets.update(x2, f2)
    if brackets.contains_exact_root:
        return brackets.best_approx_root

    # try inverse quadratic interpolation (function has to be monotonic)
    if (f1 < f2 and f2 < f3) or (f1 > f2 and f2 > f3):
        diff21 = f2 - f1
        diff23 = f2 - f3
        diff31 = f3 - f1

        num1 = f2 * f3
        num2 = f1 * f3
        num3 = f2 * f1

        denom1 = diff21 * diff31
        denom2 = diff21 * diff23
        denom3 = -diff23 * diff31

        eta = (x2 - x1) / (x3 - x1)
        phi = diff21 / diff31
        phi2 = 1.0 - phi

        if phi * phi < eta and 1.0 - eta > phi2 * phi2:
            x = x1 * (num1 / denom1) + x2 * (num2 / denom2) + x3 * (num3 / denom3)

            if brackets.is_inside(x):
                brackets.update(x, func_adapter.eval(x))
                return brackets.best_approx_root

    # inverse quadratic interpolation failed, try ridders method
    radicand = f2 * f2 - f1 * f3
    if 0.0 < radicand:
        x = x2 + (x2 - x1) * (np.sign(f1) * f2 / np.sqrt(radicand))

        if brackets.is_inside(x):
            brackets.update(x, func_adapter.eval(x))
            return brackets.best_approx_root

    # ridder failed, try regula falsi
    denom = brackets.right_fx - brackets.left_fx
    if denom != 0.0:
        num = brackets.left_x * brackets.right_fx - brackets.right_x * brackets.left_fx
        x = num / denom
        if brackets.is_inside(x):
            brackets.update(x, func_adapter.eval(x))
            return brackets.best_approx_root

    # all higher order methods failed, just did a bisection step
    return brackets.best_approx_root


def generate_pyfunc_adapter(py_func):
    func = nb.njit(py_func)

    @nb.jitclass([])
    class PythonFuncAdapter:
        def __init__(self):
            pass

        def eval(self, x):
            return func(x)

    return PythonFuncAdapter()


@nb.njit
def solve(func_adapter, brackets):
    prev_diff = brackets.width * 1.1
    while prev_diff != brackets.width:
        prev_diff = brackets.width
        bracket_iteration(func_adapter, brackets)

    return brackets.best_approx_root


@nb.njit
def solve_root(func_adapter, a, b):
    brackets = Brackets(a, func_adapter.eval(a), b, func_adapter.eval(b))
    return solve(func_adapter, brackets)


def solve_py_root(py_func, x1, x2):
    adapter = generate_pyfunc_adapter(py_func)
    brackets = Brackets(x1, adapter.eval(x1), x2, adapter.eval(x2))

    return solve(adapter, brackets)
