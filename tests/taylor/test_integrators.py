from itertools import islice

import numpy as np

from scrtbp.system import coeffs
from scrtbp.taylor import integrators


def test_fixed_stepper():
    mu = 0.01215

    taylor_params = coeffs.generate_taylor_coeffs(mu)

    period = 21.1810525829419
    n_points = 70
    step = period / (n_points - 1)
    order = 20
    solve = integrators.generate_fixed_step_integrator(taylor_params, step, order)

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

    points = solve(init_cond, n_points)
    assert np.allclose(points[0], points[-1], 0.0, 1e-14)


def test_adaptive_dense_integration():
    mu = 0.01215

    taylor_params = coeffs.generate_taylor_coeffs(mu)
    order = 20
    solve = integrators.generate_adaptive_dense_integrator(
        taylor_params, order, tol_abs=1e-16, tol_rel=1e-16
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
    period = 21.1810525829419

    t = np.linspace(0.0, period, 1000, endpoint=True)
    points = solve(init_cond, t)
    assert np.allclose(points[0], points[-1], 0.0, 1e-14)


def test_adaptive_fixed_integration():
    mu = 0.01215

    taylor_params = coeffs.generate_taylor_coeffs(mu)
    order = 20

    def exit_condition(state):
        y = state[1]
        return y < 0.2

    solve = integrators.generate_fixed_step_adaptive_integrator(
        taylor_params,
        order,
        tol_abs=1e-16,
        tol_rel=1e-16,
        py_exit_condition=exit_condition,
    )

    init_cond = np.array(
        [0.39785, 0.7101408311032396, 0.0, -0.9860206223973105, 0.5715886728443681, 0.0]
    )

    points = solve(init_cond, 20, 5.0)

    assert points.shape[0] == 20

    last_state = np.array(
        [
            -0.13265194931562035,
            1.0531334795398335,
            0.0,
            -0.9420753032804786,
            -0.08777226445063985,
            0.0,
        ]
    )
    assert np.allclose(last_state, points[-1], 0.0, 1e-14)


def test_adaptive_fixed_integration_with_exit():
    mu = 0.01215

    taylor_params = coeffs.generate_taylor_coeffs(mu)
    order = 20

    def exit_condition(state):
        y = state[1]
        return y < 0.2

    solve = integrators.generate_fixed_step_adaptive_integrator(
        taylor_params,
        order,
        tol_abs=1e-16,
        tol_rel=1e-16,
        py_exit_condition=exit_condition,
    )

    init_cond = np.array(
        [0.39785, 0.7101408311032396, 0.0, -0.9860206223973105, 0.5715886728443681, 0.0]
    )

    points = solve(init_cond, 1000, step=5.0)
    assert points.shape[0] == 50

    state_before_exit = np.array(
        [
            -0.6104160649803588,
            0.3908450895714359,
            0.0,
            -0.6582608984046822,
            -1.1016494227475653,
            0.0,
        ]
    )
    assert np.allclose(state_before_exit, points[-1], 0.0, 1e-14)


def test_adaptive_integration_generator():
    mu = 0.01215

    taylor_params = coeffs.generate_taylor_coeffs(mu)
    order = 20

    point_generator = integrators.generate_fixed_step_adaptive_generator(
        taylor_params, order, tol_abs=1e-16, tol_rel=1e-16
    )

    init_cond = np.array(
        [0.39785, 0.7101408311032396, 0.0, -0.9860206223973105, 0.5715886728443681, 0.0]
    )

    step = 0.5
    
    points = np.array(list(islice(point_generator(init_cond, step), None, 3)))

    precomputed_points = np.array(
        [
            [
                0.2666141303190991,
                0.792529045113351,
                0.0,
                -1.0335204496266346,
                0.41976871859252407,
                0.0,
            ],
            [
                0.16324734999014953,
                0.8621106824396213,
                0.0,
                -1.030510584313352,
                0.2881080551484904,
                0.0,
            ],
            [
                0.09972978852637138,
                0.9171405586713944,
                0.0,
                -1.0036799157801077,
                0.19450015404185825,
                0.0,
            ],
        ]
    )
    assert np.allclose(precomputed_points, points, 0.0, 1e-15)
