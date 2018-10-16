import numpy as np

from scrtbp.system import coeffs, sections
from scrtbp.taylor import events


def test_basic_poincare_orbit():
    mu = 0.01
    jacobi = 2.992

    coeff_func, state_dim, extra_dim = coeffs.generate_taylor_coeffs(mu)
    poincare_func, _, _ = sections.generate_poincare_tools(mu, jacobi)

    step = 0.01
    order = 30
    poincare_solve = events.generate_event_solver(
        coeff_func, state_dim, extra_dim, poincare_func, step, order
    )

    init_cond = np.array([0.01, 0.01, 0.1, 0.01, 0.01, 0.01])
    points = poincare_solve(init_cond, 10)
    print(points)


test_basic_poincare_orbit()
