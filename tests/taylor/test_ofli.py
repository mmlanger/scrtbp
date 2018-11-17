import math

import numpy as np

from scrtbp.system import coeffs, sections
from scrtbp.taylor import integrators


def test_ofli_basic():
    mu = 0.01215

    coeff_func, state_dim, extra_dim = coeffs.generate_taylor_coeffs(mu)

    step = 0.05
    order = 20
    solve = integrators.generate_adaptive_dense_integrator(
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

    t = np.linspace(0.0, 100.0, 1000)
    points = solve(init_cond, t)
    print(points)
