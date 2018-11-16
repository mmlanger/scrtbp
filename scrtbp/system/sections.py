import numpy as np

from . import tools
from scrtbp.exceptions import TransformationNotDefined


def generate_poincare_tools(mu, Cj):

    sqrt_3 = np.sqrt(3.0)

    def char_func(state):
        """
        characteristic function of the Poincare section
        """
        x = state[0]
        y = state[1]
        return y - sqrt_3 * (x + mu)

    def to_poincare(phase_space_coord):
        """
        transformation from full to reduced phase space
        """
        x = phase_space_coord[0]
        z = phase_space_coord[2]
        px = phase_space_coord[3]
        py = phase_space_coord[4]
        pz = phase_space_coord[5]

        rho = 2 * (x + mu)
        prho = 0.5 * (px + sqrt_3 * py)

        return np.array([rho, prho, z, pz])

    def to_phase_space(poincare_coord):
        """
        transformation from reduced to full phase space
        """
        rho = poincare_coord[0]
        prho = poincare_coord[1]
        z = poincare_coord[2]
        pz = poincare_coord[3]

        x = 0.5 * rho - mu
        y = sqrt_3 / 2 * rho

        radicand_1 = (mu + x - 1) ** 2 + y ** 2 + z ** 2
        radicand_2 = (mu + x) ** 2 + y ** 2 + z ** 2

        potential = -mu / np.sqrt(radicand_1) - (1 - mu) / np.sqrt(radicand_2)
        A = (
            -Cj
            - 2 * potential
            - prho ** 2
            + sqrt_3 * prho * x
            - prho * y
            - pz ** 2
            + 0.25 * x ** 2
            + sqrt_3 / 2.0 * x * y
            + 3.0 / 4.0 * y ** 2
        )
        if A < 0:
            msg = "A = {} smaller 0; state {} invalid for Cj = {}"
            raise TransformationNotDefined(msg.format(A, poincare_coord, Cj))

        px = prho / 2.0 - sqrt_3 / 4.0 * x - 3.0 / 4.0 * y - sqrt_3 / 2.0 * np.sqrt(A)
        py = 1.0 / sqrt_3 * (2.0 * prho - px)

        return np.array([x, y, z, px, py, pz])

    return char_func, to_poincare, to_phase_space
