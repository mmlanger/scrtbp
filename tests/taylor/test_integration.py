import numpy as np

from scrtbp.system import coeffs
from scrtbp.taylor import integrators


def test_fixed_stepper():
    mu = 0.01215

    coeff_func, state_dim, extra_dim = coeffs.generate_taylor_coeffs(mu)

    period = 21.1810525829419
    n_points = 70
    step = period / (n_points - 1)
    order = 20
    solve = integrators.generate_fixed_step_integrator(
        coeff_func, state_dim, extra_dim, step, order
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

    points = solve(init_cond, n_points)
    assert np.allclose(points[0], points[-1], 0.0, 1e-14)


def test_adaptive_dense_integration():
    mu = 0.01215

    coeff_func, state_dim, extra_dim = coeffs.generate_taylor_coeffs(mu)

    order = 20
    solve = integrators.generate_adaptive_dense_integrator(
        coeff_func, state_dim, extra_dim, order, tol_abs=1e-16, tol_rel=1e-16
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
