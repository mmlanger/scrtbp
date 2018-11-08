import math

import numpy as np

from scrtbp.system import coeffs, sections
from scrtbp.taylor import events, po_solver


def vector_isclose(x1, x2, rel_tol, abs_tol):
    return math.isclose(np.linalg.norm(x2 - x1), 0.0, rel_tol=rel_tol, abs_tol=abs_tol)


def test_adaptive_event_solver():
    mu = 0.01215
    jacobi = 2.992
    period = 21.1810525829419

    coeff_func, state_dim, extra_dim = coeffs.generate_taylor_coeffs(mu)
    poincare_func, trans_red, trans_full = sections.generate_poincare_tools(mu, jacobi)

    order = 20
    eps_abs = 1e-16
    eps_tol = 1e-16
    solve_events = events.generate_adaptive_event_solver(
        coeff_func, state_dim, extra_dim, poincare_func, order, eps_abs, eps_tol
    )
    poincare_map = po_solver.generate_poinare_map(solve_events, trans_red, trans_full)

    init_cond = np.array([0.44, 0.78, 0.0, -0.90, 0.54, 0.0])
    init_cond = trans_red(init_cond)
    dirm_solver = po_solver.DirmIterator(poincare_map, init_cond)

    for i in range(1000):
        dirm_solver.iterate(1e-1)
        dist = dirm_solver.distance

    # rel_tol = 1e-14
    # abs_tol = 1e-14
    # assert vector_isclose(points[0], points[2], rel_tol, abs_tol)
    # assert math.isclose(period, t[2] - t[1], rel_tol=rel_tol, abs_tol=abs_tol)
