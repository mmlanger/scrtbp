import math

import numpy as np

from scrtbp.system import coeffs, sections
from scrtbp.taylor import events


def vector_isclose(x1, x2, rel_tol, abs_tol):
    return math.isclose(np.linalg.norm(x2 - x1), 0.0, rel_tol=rel_tol, abs_tol=abs_tol)


def test_poincare_periodic_orbit():
    mu = 0.01215
    jacobi = 2.992
    period = 21.1810525829419

    coeff_func, state_dim, extra_dim = coeffs.generate_taylor_coeffs(mu)
    poincare_func, _, _ = sections.generate_poincare_tools(mu, jacobi)

    step = 0.1
    order = 20
    poincare_solve = events.generate_event_solver(
        coeff_func, state_dim, extra_dim, poincare_func, step, order
    )

    init_cond = np.array(
        [
            0.440148176542848,
            0.783403421942971,
            0.0,
            -0.905419824338076,
            0.540413382924902,
            0.0,
        ]
    )

    points, t = poincare_solve(init_cond, 3)

    rel_tol = 1e-14
    abs_tol = 1e-14

    assert vector_isclose(points[0], points[1], rel_tol, abs_tol)
    assert vector_isclose(points[1], points[2], rel_tol, abs_tol)
    assert vector_isclose(points[0], points[2], rel_tol, abs_tol)

    assert math.isclose(period, t[2] - t[1], rel_tol=rel_tol, abs_tol=abs_tol)


def test_adaptive_event_solver():
    mu = 0.01215
    jacobi = 2.992
    period = 21.1810525829419

    coeff_func, state_dim, extra_dim = coeffs.generate_taylor_coeffs(mu)
    poincare_func, _, _ = sections.generate_poincare_tools(mu, jacobi)

    order = 20
    eps_abs = 1e-16
    eps_tol = 1e-16
    poincare_solve = events.generate_adaptive_event_solver(
        coeff_func, state_dim, extra_dim, poincare_func, order, eps_abs, eps_tol
    )

    init_cond = np.array(
        [
            0.440148176542848,
            0.783403421942971,
            0.0,
            -0.905419824338076,
            0.540413382924902,
            0.0,
        ]
    )

    points, t = poincare_solve(init_cond, 3)

    rel_tol = 1e-14
    abs_tol = 1e-14

    assert vector_isclose(points[0], points[1], rel_tol, abs_tol)
    assert vector_isclose(points[1], points[2], rel_tol, abs_tol)
    assert vector_isclose(points[0], points[2], rel_tol, abs_tol)

    assert math.isclose(period, t[2] - t[1], rel_tol=rel_tol, abs_tol=abs_tol)
