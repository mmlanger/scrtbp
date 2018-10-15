import numpy as np

from scrtbp.system import coeffs



def test_basic_poincare_orbit():
    mu = 0.01

    #coeff_func, state_dim, extra_dim = coeffs.generate_taylor_coeffs(mu)