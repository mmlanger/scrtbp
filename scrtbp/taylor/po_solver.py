import numpy as np
import numba as nb


def generate_poinare_map(solve_events, transform_reduced, transform_full):
    transform_full = nb.jit(transform_full, forceobj=True)
    transform_reduced = nb.jit(transform_reduced, forceobj=True)

    @nb.jit(forceobj=True)
    def poincare_map(point, n_iter):
        state = transform_full(point)
        states, times = solve_events(state, n_iter + 1)
        return transform_reduced(states[n_iter]), times[n_iter]

    return poincare_map


class DirmIterator:
    def __init__(self, poincare_map, init_guess, order=1, refl_perm_matrix=None):
        self.poincare_map = poincare_map
        self.order = order

        self.guess = init_guess
        self.prev_guess = init_guess
        self.return_time = None
        self.distance = 0.0

        self.refl_perm_matrix = refl_perm_matrix

    @property
    def state(self):
        return self.guess

    @state.setter
    def state(self, guess):
        self.guess = guess
        self.prev_guess = guess
        self.distance = 0.0

    def iterate(self, tau=0.1):
        self.prev_guess = self.guess.copy()
        iterated_guess, return_time = self.poincare_map(self.guess, self.order)

        if self.refl_perm_matrix:
            self.guess += tau * self.refl_perm_matrix @ (iterated_guess - self.guess)
        else:
            self.guess += tau * (iterated_guess - self.guess)

        self.return_time = return_time
        self.distance = np.linalg.norm(self.guess - self.prev_guess)


def solve_periodic_orbit(
    poincare_map,
    init_guess,
    tau=0.1,
    dirm_iter=1000,
    refine_iter=50,
    po_order=1,
    refl_perm_matrix=None,
    refine_fac=0.1,
    verbose=False,
):
    dirm_solver = DirmIterator(poincare_map, init_guess, order=po_order)
    msg = "iteration {}: dist {:.15e} and period {}"

    for i in range(dirm_iter):
        dirm_solver.iterate(tau)
        if verbose:
            if i % 10 == 0:
                print(msg.format(i, dirm_solver.distance, dirm_solver.return_time))

    if verbose:
        print("refinement loop:")

    refined_tau = refine_fac * tau
    for i in range(refine_iter):
        dirm_solver.iterate(refined_tau)
        if verbose:
            if i % 10 == 0:
                print(msg.format(i, dirm_solver.distance, dirm_solver.return_time))

    return dirm_solver.state, dirm_solver.return_time
