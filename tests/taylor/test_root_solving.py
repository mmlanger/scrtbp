import math

import numba as nb
import numpy as np

from scrtbp.util import root


def test_simple_root():
    def test_func(x):
        return np.sin(x) - x / 2

    adapter = root.generate_pyfunc_adapter(test_func)
    x1 = np.pi / 2
    x2 = np.pi
    brackets = root.Brackets(x1, adapter.eval(x1), x2, adapter.eval(x2))

    prev_diff = brackets.width * 1.1
    while prev_diff != brackets.width:
        prev_diff = brackets.width
        root.bracket_iteration(adapter, brackets)

    result = brackets.best_approx_root
    assert math.isclose(result, 1.895494267033981, rel_tol=0.0, abs_tol=1e-15)


def test_func_collection():
    result = root.solve_py_root(lambda x: np.sin(x) - 0.5, 0.0, 1.5)
    assert math.isclose(result, 0.5235987755982989, rel_tol=0.0, abs_tol=1e-15)

    result = root.solve_py_root(lambda x: x, -1.0, 1.0)
    assert math.isclose(result, 0.0, rel_tol=0.0, abs_tol=0.0)

    result = root.solve_py_root(lambda x: x, -np.pi, 1.0)
    assert math.isclose(result, 0.0, rel_tol=0.0, abs_tol=0.0)

    result = root.solve_py_root(lambda x: (2 * x - 1) / x, 0.00001, 1.0)
    assert math.isclose(result, 0.5, rel_tol=0.0, abs_tol=4e-16)
