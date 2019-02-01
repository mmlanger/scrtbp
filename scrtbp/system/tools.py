import numpy as np


def generate_rhs(mu):
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

def generate_jacobi(mu):
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

    return jacobi


def H_grad(mu, state):
    """
    calculates normalized gradient of hamiltonian for scrtbp;
    used for initial var_state in ofli
    """
    x = state[0]
    y = state[1]
    z = state[2]
    px = state[3]
    py = state[4]
    pz = state[5]

    grad = np.empty(6)

    r1 = np.sqrt((x + mu) * (x + mu) + y * y + z * z)
    r2 = np.sqrt((x + mu - 1) * (x + mu - 1) + y * y + z * z)

    r2_pow3 = r2 * r2 * r2

    grad[0] = -py + (1 - mu) * (x + mu) / (r1 * r1 * r1) + mu * (x + mu - 1) / r2_pow3
    grad[1] = px + y * ((1 - mu) / (r1 * r1 * r1) + mu / r2_pow3)
    grad[2] = z * ((1 - mu) / (r1 * r1 * r1) + mu / r2_pow3)
    grad[3] = px + y
    grad[4] = py - x
    grad[5] = pz

    return grad


def generate_ofli_init_func(mu):
    def generate_ofli_init_cond(state):
        init_cond = np.empty(12)

        grad = H_grad(mu, state)
        norm = np.linalg.norm(grad)

        init_cond[:6] = state
        init_cond[6:] = grad / norm

        return init_cond

    return generate_ofli_init_cond
