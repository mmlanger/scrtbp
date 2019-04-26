import math
from itertools import islice

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

    assert math.isclose(period, t[2] - t[1], rel_tol=0.0, abs_tol=1e-8)


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

    assert math.isclose(period, t[2] - t[1], rel_tol=0.0, abs_tol=1e-8)


def test_adaptive_event_generator():
    mu = 0.01215

    taylor_params = coeffs.generate_taylor_coeffs(mu)
    poincare_func = sections.generate_poincare_tools(mu)
    order = 20

    point_generator = events.generate_adaptive_event_generator(
        taylor_params, poincare_func, order, tol_abs=1e-16, tol_rel=1e-16
    )

    init_cond = np.array([0.39785, 0.71014, 0.001, -0.98602, 0.571588, 0.00025])
    points = np.array([s for s, t in islice(point_generator(init_cond), None, 3)])

    precomputed_points = np.array(
        [
            [
                0.3978496481075093,
                0.710140221607567,
                0.001000000318880387,
                -0.9860201962698747,
                0.5715876208422852,
                0.0002499976950271774,
            ],
            [
                0.4263632104330913,
                0.7595271602602567,
                -0.000965633500996545,
                -0.9173784469340452,
                0.5684961830311902,
                -0.0005994497329097056,
            ],
            [
                0.4582748256453528,
                0.8147996991594816,
                0.001132265237324626,
                -0.8833310514195326,
                0.5044621880970391,
                -8.366385185794535e-05,
            ],
        ]
    )
    assert np.allclose(precomputed_points, points, 0.0, 1e-15)
