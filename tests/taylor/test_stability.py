import math

import numpy as np

from scrtbp import exceptions
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


def test_ofli_overflow():
    mu = 0.01215

    taylor_params = coeffs.generate_variational_taylor_coeffs(mu)
    solve = stability.generate_ofli_integrator(
        taylor_params, 100.0, order=50, tol_abs=1e-5, tol_rel=1e-5
    )

    init_cond = np.array(
        [
            0.39785,
            0.7101408311032396,
            0.0,
            -0.998522098295406,
            0.5499388898599199,
            0.0,
            0.3757658020984179,
            0.6096142141175671,
            0.0,
            -0.6173786243534443,
            0.3255982280520004,
            0.0,
        ]
    )

    try:
        solve(init_cond)
    except exceptions.StepControlFailure:
        assert True
    except:
        assert False
    else:
        assert False


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


def test_ofli_parallel():
    mu = 0.01215

    taylor_params = coeffs.generate_variational_taylor_coeffs(mu)
    parallel_solve = stability.generate_ofli_integrator(
        taylor_params, 100.0, parallel=True
    )

    init_cond_template = np.array(
        [
            0.440148176542848,
            0.783403421942971,
            0.0,
            -0.905419824338076,
            0.540413382924902,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    init_conds = []
    for i in range(6):
        init_cond = init_cond_template.copy()
        init_cond[6 + i] += 1.0
        init_conds.append(init_cond)
    init_conds = np.array(init_conds)

    ofli, _ = parallel_solve(init_conds)

    ofli_ref = np.array(
        [
            3.3844254390624933,
            4.3703843414232075,
            0.16896605328460246,
            4.248470717714327,
            3.996114068586288,
            0.010722916367483078,
        ]
    )
    assert np.allclose(ofli, ofli_ref, rtol=1e-10, atol=1e-10)


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

    multiplier = sorted(multiplier, key=lambda x: x.real)

    def complex_arg(x):
        if x.imag >= 0.0:
            return np.angle(x)
        else:
            return np.angle(np.conjugate(x))

    freqs = sorted(complex_arg(x) / period for x in multiplier[::2])

    om1 = (2.0 * np.pi) / period
    om2 = freqs[1] + 3 * om1
    om3 = freqs[2] + 3 * om1

    abs_tol = 1e-5
    assert math.isclose(om1, 0.2966418, rel_tol=0.0, abs_tol=abs_tol)
    assert math.isclose(om2, 0.9282165, rel_tol=0.0, abs_tol=abs_tol)
    assert math.isclose(om3, 1.0021599, rel_tol=0.0, abs_tol=abs_tol)
