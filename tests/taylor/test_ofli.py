import math

import numpy as np

from scrtbp.system import coeffs, sections
from scrtbp.taylor import indicators


def test_ofli_basic():
    mu = 0.01215

    coeff_func, state_dim, extra_dim = coeffs.generate_variational_taylor_coeffs(mu)
    solve = indicators.generate_ofli_integrator(coeff_func, state_dim, extra_dim, 500.0)

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
    print(ofli, "  ", time)
