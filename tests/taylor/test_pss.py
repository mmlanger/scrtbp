import math

import numpy as np

from scrtbp.system import coeffs, sections
from scrtbp.taylor import pss


def test_pss_orbit():
    mu = 0.01215
    jacobi = 2.992

    taylor_params = coeffs.generate_taylor_coeffs(mu)
    poincare_func, _, _ = sections.generate_poincare_tools(mu, jacobi)

    def slice_condition(point):
        pz = point[5]
        return abs(pz) < 1e-3

    order = 20
    pss_solve = pss.generate_adaptive_pss_solver(
        taylor_params, poincare_func, slice_condition, order
    )

    init_cond = np.array(
        [
            0.440148176542848,
            0.783403421942971,
            0.02,
            -0.905419824338076,
            0.540413382924902,
            0.0,
        ]
    )

    points, t = pss_solve(init_cond, 10)

    last_pss_point = np.array(
        [
            0.44178915227058674,
            0.7862456752774013,
            0.020068511232556891,
            -0.9034821179184479,
            0.53757237305311767,
            0.00081354636533123928,
        ]
    )
    last_pss_time = 10928.635911486037

    rel_tol = 1e-14
    abs_tol = 1e-14

    assert np.allclose(points[-1], last_pss_point, rel_tol, abs_tol)
    assert math.isclose(t[-1], last_pss_time, rel_tol=rel_tol, abs_tol=abs_tol)

