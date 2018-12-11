import math

import numpy as np

from scrtbp.system import coeffs
from scrtbp.taylor import stability, integrators


def test_ofli_basic():
    mu = 0.01215

    taylor_params = coeffs.generate_variational_taylor_coeffs(mu)
    solve = stability.generate_ofli_integrator(taylor_params, 100.0)

    init_cond = np.array(
        [
            0.440148176542848,
            0.783403421942971,
            0.0,
            -0.905419824338076,
            0.540413382924902,
            0.0,
            1.0,
            0.5,
            0.2,
            0.5,
            0.5,
            0.5,
        ]
    )

    ofli, _ = solve(init_cond)
    assert math.isclose(4.163497918, ofli, rel_tol=1e-6, abs_tol=1e-6)


def test_max_ofli_limit():
    mu = 0.01215

    max_ofli = 4.0
    max_time = 100.0

    taylor_params = coeffs.generate_variational_taylor_coeffs(mu)
    solve = stability.generate_ofli_integrator(
        taylor_params, max_time=max_time, max_ofli=max_ofli
    )

    init_cond = np.array(
        [
            0.440148176542848,
            0.783403421942971,
            0.0,
            -0.905419824338076,
            0.540413382924902,
            0.0,
            1.0,
            0.5,
            0.2,
            0.5,
            0.5,
            0.5,
        ]
    )

    ofli, time = solve(init_cond)
    assert math.isclose(max_ofli, ofli, rel_tol=1e-15, abs_tol=1e-15)
    assert math.isclose(91.1161972, time, rel_tol=1e-6, abs_tol=1e-6)


def test_floquet_multiplier():
    mu = 0.01215

    taylor_params = coeffs.generate_variational_taylor_coeffs(mu)
    order = 20
    solve = integrators.generate_adaptive_dense_integrator(
        taylor_params, order, tol_abs=1e-16, tol_rel=1e-16
    )

    init_cond = np.array(
        [
            0.44014817654284644,
            0.78340342194296775,
            0.0,
            -0.90541982433807799,
            0.54041338292490493,
            0.0,
        ]
    )
    period = 21.181052582941817

    multiplier, _ = stability.compute_floquet_multiplier(solve, init_cond, period)
    assert np.allclose(abs(multiplier), 1.0, rtol=0.0, atol=1e-10)

    #n = 3  # number of 2*pi shifts in the phase
    #multiplier = multiplier[multiplier.imag >= 0.0]
    #freqs = (np.angle(multiplier) + n * 2.0 * np.pi) / period

    #assert np.allclose(abs(multiplier), 1.0, rtol=0.0, atol=1e-10)
