import math

import numpy as np

from scrtbp.system import coeffs, sections
from scrtbp.taylor import escape


def test_fixed_step_escape_orbit():
    mu = 0.01215

    taylor_params = coeffs.generate_taylor_coeffs(mu)
    poincare_func = sections.generate_poincare_tools(mu)

    def escape_condition(point):
        return np.linalg.norm(point) > 2.0

    order = 20
    escape_solve = escape.generate_poincare_escape_solver(
        taylor_params, poincare_func, escape_condition, order
    )

    init_cond = np.array(
        [0.42, 0.783403421942971, 0.1, -0.905419824338076, 0.540413382924902, 0.0]
    )

    point, t = escape_solve(init_cond, 10)

    escaped_point = np.array(
        [
            0.11057721919072513,
            0.21256977910997807,
            -0.072191339909561877,
            -2.3803412262232362,
            0.15226439712815315,
            0.35365331439714465,
        ]
    )
    escaped_time = 801.6231569667372

    rel_tol = 1e-14
    abs_tol = 1e-14

    assert np.allclose(point, escaped_point, rel_tol, abs_tol)
    assert math.isclose(t, escaped_time, rel_tol=rel_tol, abs_tol=abs_tol)


def test_adaptive_step_escape_orbit():
    mu = 0.01215

    taylor_params = coeffs.generate_taylor_coeffs(mu)

    def escape_char_func(point):
        return point[1] - 0.2

    order = 20
    t_max = 10000.0
    escape_solve = escape.generate_adaptive_escape_solver(
        taylor_params, escape_char_func, t_max, order
    )

    init_cond = np.array(
        [0.42, 0.783403421942971, 0.1, -0.905419824338076, 0.540413382924902, 0.0]
    )

    point, t = escape_solve(init_cond, 10)

    escaped_point = np.array(
        [
            1.0074683467072398,
            0.2,
            0.09772610173373972,
            -0.3800603026613051,
            0.8027001698297486,
            -0.04675504994802894,
        ]
    )
    escaped_time = 784.0811561375356

    rel_tol = 1e-14
    abs_tol = 1e-14

    assert np.allclose(point, escaped_point, rel_tol, abs_tol)
    assert math.isclose(t, escaped_time, rel_tol=rel_tol, abs_tol=abs_tol)
