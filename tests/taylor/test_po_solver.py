import math

import numpy as np

from scrtbp.system import coeffs, sections
from scrtbp.taylor import events, po_solver


def vector_isclose(x1, x2, rel_tol, abs_tol):
    return math.isclose(np.linalg.norm(x2 - x1), 0.0, rel_tol=rel_tol, abs_tol=abs_tol)


def test_adaptive_dirm_solver():
    mu = 0.01215
    jacobi = 2.992

    coeff_func, state_dim, extra_dim = coeffs.generate_taylor_coeffs(mu)
    poincare_func, trans_red, trans_full = sections.generate_poincare_tools(mu, jacobi)

    order = 20
    eps_abs = 1e-16
    eps_tol = 1e-16
    solve_events = events.generate_adaptive_event_solver(
        coeff_func, state_dim, extra_dim, poincare_func, order, eps_abs, eps_tol
    )
    poincare_map = po_solver.generate_poinare_map(solve_events, trans_red, trans_full)

    init_cond = np.array([0.9043, 0.0176, 0.0, 0.0])
    dirm_solver = po_solver.DirmIterator(poincare_map, init_cond)

    for _ in range(1000):
        dirm_solver.iterate(0.1)

    for _ in range(10):
        dirm_solver.iterate(0.01)

    period = dirm_solver.return_time
    distance = dirm_solver.distance

    assert math.isclose(period, 21.1810525829419, rel_tol=5e-15, abs_tol=0.0)
    assert math.isclose(distance, 0.0, rel_tol=0.0, abs_tol=8e-16)


def test_po_solver_func():
    mu = 0.01215
    jacobi = 2.992

    coeff_func, state_dim, extra_dim = coeffs.generate_taylor_coeffs(mu)
    poincare_func, trans_red, trans_full = sections.generate_poincare_tools(mu, jacobi)

    order = 20
    eps_abs = 1e-16
    eps_tol = 1e-16
    solve_events = events.generate_adaptive_event_solver(
        coeff_func, state_dim, extra_dim, poincare_func, order, eps_abs, eps_tol
    )
    poincare_map = po_solver.generate_poinare_map(solve_events, trans_red, trans_full)

    init_cond = np.array([0.9043, 0.0176, 0.0, 0.0])
    point, period = po_solver.solve_periodic_orbit(poincare_map, init_cond)
    state = trans_full(point)

    po_period = 21.1810525829419
    po_state = np.array(
        [
            0.440148176542848,
            0.783403421942971,
            0.0,
            -0.905419824338076,
            0.540413382924902,
            0.0,
        ]
    )

    assert math.isclose(period, po_period, rel_tol=5e-15, abs_tol=0.0)
    assert vector_isclose(state, po_state, rel_tol=0.0, abs_tol=1e-14)
