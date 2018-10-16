import numpy as np

from scrtbp.system import coeffs
from scrtbp.taylor import events


def test_basic_poincare_orbit():
    mu = 0.01

    coeff_func, state_dim, extra_dim = coeffs.generate_taylor_coeffs(mu)

    step = 0.01
    order = 30
    solver = events.generate_event_solver(coeff_func, state_dim, extra_dim, step, order)

    init_cond = np.array([0.01, 0.01, 0.1, 0.01, 0.01, 0.01])
    points = solver(init_cond, 100)
    print(points)

test_basic_poincare_orbit()