import numpy as np

from . import tools


def generate_poincare_tools(mu, Cj, tolerance=1e-15):
    _, _, jacobi = tools.generate_tools(mu)

    sqrt_3 = np.sqrt(3.0)

    def char_func(state):
        x = state[0]
        y = state[1]
        return y - sqrt_3 * (x + mu)

    def to_phase_space(phase_space_coord, poincare_coord):
        rho = poincare_coord[0]
        prho = poincare_coord[1]
        z = poincare_coord[2]
        pz = poincare_coord[3]

        x = 0.5 * rho - mu
        y = np.sqrt(3) / 2 * rho

        V = -mu / np.sqrt((mu + x - 1) ** 2 + y ** 2 + z ** 2) - (1 - mu) / np.sqrt(
            (mu + x) ** 2 + y ** 2 + z ** 2
        )
        A = (
            -Cj
            - 2 * V
            - prho ** 2
            + np.sqrt(3) * prho * x
            - prho * y
            - pz ** 2
            + 0.25 * x ** 2
            + np.sqrt(3) / 2 * x * y
            + 3 / 4 * y ** 2
        )

        if A < 0:
            print(
                "A = {0} smaller 0; initial state {1} invalid for Cj = {2}".format(
                    A, poincare_coord, Cj
                )
            )
            return

        # px_plus = prho / 2 - np.sqrt(3) / 4 * x - 3 / 4 * y + np.sqrt(
        #    3) / 2 * np.sqrt(A)
        px_minus = (
            prho / 2 - np.sqrt(3) / 4 * x - 3 / 4 * y - np.sqrt(3) / 2 * np.sqrt(A)
        )

        # py_plus = 1 / np.sqrt(3) * (2 * prho - px_plus)
        py_minus = 1 / np.sqrt(3) * (2 * prho - px_minus)

        # Cj_plus = jacobi(([x, y, z, px_plus, py_plus, pz]))
        Cj_minus = jacobi(([x, y, z, px_minus, py_minus, pz]))

        if abs(Cj_minus - Cj) <= tolerance:
            # print("Chose minus")
            px = px_minus
            py = py_minus
        # elif abs(Cj_plus - Cj ) <= tolerance:
        #    print("Chose plus")
        #    px = px_plus
        #    py = py_plus
        else:
            msg = "jacobi does not match Cj; " "initial state {0} invalid for Cj = {1}"
            print(msg.format(poincare_coord, Cj))
            return

        phase_space_coord[0] = x
        phase_space_coord[1] = y
        phase_space_coord[2] = z
        phase_space_coord[3] = px
        phase_space_coord[4] = py
        phase_space_coord[5] = pz

        return phase_space_coord

    def to_poincare(phase_space_coord, poincare_coord):

        x = phase_space_coord[0]
        y = phase_space_coord[1]
        z = phase_space_coord[2]
        px = phase_space_coord[3]
        py = phase_space_coord[4]
        pz = phase_space_coord[5]

        sqr_3 = np.sqrt(3)

        if not abs(y - sqr_3 * (x + mu)) <= tolerance:
            print(
                "CAUTION: Coordinates ",
                phase_space_coord,
                " not in Poincare section. S = ",
                y - sqr_3 * (x + mu),
            )

        rho = 2 * (x + mu)
        prho = 0.5 * (px + sqr_3 * py)

        poincare_coord[0] = rho
        poincare_coord[1] = prho
        poincare_coord[2] = z
        poincare_coord[3] = pz

        return poincare_coord

    return char_func, to_phase_space, to_poincare
