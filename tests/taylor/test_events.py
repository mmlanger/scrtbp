import math

import numpy as np

from scrtbp.system import coeffs, sections
from scrtbp.taylor import events


def test_poincare_periodic_orbit():
    mu = 0.01215
    period = 21.1810525829419

    taylor_params = coeffs.generate_taylor_coeffs(mu)
    poincare_func = sections.generate_poincare_tools(mu)

    step = 0.1
    order = 20
    poincare_solve = events.generate_event_solver(
        taylor_params, poincare_func, step, order
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

    assert np.allclose(points[0], points[1], rel_tol, abs_tol)
    assert np.allclose(points[1], points[2], rel_tol, abs_tol)
    assert np.allclose(points[0], points[2], rel_tol, abs_tol)

    assert math.isclose(period, t[2] - t[1], rel_tol=rel_tol, abs_tol=abs_tol)


def test_adaptive_event_solver():
    mu = 0.01215
    period = 21.1810525829419

    taylor_params = coeffs.generate_taylor_coeffs(mu)
    poincare_func = sections.generate_poincare_tools(mu)

    order = 20
    eps_abs = 1e-16
    eps_rel = 1e-16
    poincare_solve = events.generate_adaptive_event_solver(
        taylor_params, poincare_func, order, eps_abs, eps_rel
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

    assert np.allclose(points[0], points[1], rel_tol, abs_tol)
    assert np.allclose(points[1], points[2], rel_tol, abs_tol)
    assert np.allclose(points[0], points[2], rel_tol, abs_tol)

    assert math.isclose(period, t[2] - t[1], rel_tol=rel_tol, abs_tol=abs_tol)


def test_adaptive_event_solver2():
    mu = 0.01215
    period = 21.070381823628498
    taylor_params = coeffs.generate_taylor_coeffs(mu)
    poincare_func = sections.generate_poincare_tools(mu)

    order = 50
    eps_abs = 1e-16
    eps_rel = 0.0
    poincare_solve = events.generate_adaptive_event_solver(
        taylor_params, poincare_func, order, eps_abs, eps_rel
    )

    init_cond = np.array(
        [
            0.48746299023725853,
            0.86535508321234522,
            0.0,
            -0.86638425004587294,
            0.48805050039736497,
            0.0,
        ]
    )

    points, t = poincare_solve(init_cond, 3)

    rel_tol = 1e-14
    abs_tol = 1e-14

    assert np.allclose(points[0], points[1], rel_tol, abs_tol)
    assert np.allclose(points[1], points[2], rel_tol, abs_tol)
    assert np.allclose(points[0], points[2], rel_tol, abs_tol)

    assert math.isclose(period, t[2] - t[1], rel_tol=0.0, abs_tol=1e-10)
