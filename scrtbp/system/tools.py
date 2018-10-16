import numpy as np


def generate_tools(mu):
    def rhs(state, t):
        x = state[0]
        y = state[1]
        z = state[2]
        px = state[3]
        py = state[4]
        pz = state[5]

        radicand_1 = (x + mu) * (x + mu) + y * y + z * z
        radicand_2 = (x + mu - 1.0) * (x + mu - 1.0) + y * y + z * z

        r1_fac = (1.0 - mu) / (radicand_1 * np.sqrt(radicand_1))
        r2_fac = mu / (radicand_2 * np.sqrt(radicand_2))

        rhs = np.array(
            [
                px + y,
                py - x,
                pz,
                py - (x + mu) * r1_fac - (x + mu - 1.0) * r2_fac,
                -px - y * (r1_fac + r2_fac),
                -z * (r1_fac + r2_fac),
            ]
        )

        return rhs

    def energy(state):
        x = state[0]
        y = state[1]
        z = state[2]
        px = state[3]
        py = state[4]
        pz = state[5]

        r1 = np.sqrt((x + mu) * (x + mu) + y * y + z * z)
        r2 = np.sqrt((x + mu - 1.0) * (x + mu - 1.0) + y * y + z * z)

        kinetic = 0.5 * (px * px + py * py + pz * pz)
        mixed = px * y - py * x
        potential = -(1.0 - mu) / r1 - mu / r2

        return kinetic + mixed + potential

    def jacobi(state):
        return -2.0 * energy(state)

    return rhs, energy, jacobi
