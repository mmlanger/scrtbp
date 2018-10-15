from numba import njit
from scrtbp.taylor import rules


def generate_taylor_coeffs(mu):
    state_dim = 6
    extra_dim = 17

    gamma1 = mu - 1
    gamma2 = -mu
    exponent = -3 / 2

    @njit
    def taylor_coeff_func(state, taylor_coeffs, ext_coeffs):
        # taylor coefficients shorthands
        x = taylor_coeffs[0]
        y = taylor_coeffs[1]
        z = taylor_coeffs[2]
        px = taylor_coeffs[3]
        py = taylor_coeffs[4]
        pz = taylor_coeffs[5]

        # auxiliary variable array shorthands
        u1 = ext_coeffs[0]
        u2 = ext_coeffs[1]
        v1 = ext_coeffs[2]
        v2 = ext_coeffs[3]
        v3 = ext_coeffs[4]
        v4 = ext_coeffs[5]
        w1 = ext_coeffs[6]
        w2 = ext_coeffs[7]
        w3 = ext_coeffs[8]
        a1 = ext_coeffs[9]
        a2 = ext_coeffs[10]
        a3 = ext_coeffs[11]
        a4 = ext_coeffs[12]
        b = ext_coeffs[13]
        c = ext_coeffs[14]
        d = ext_coeffs[15]
        f = ext_coeffs[16]

        # initial condition
        x[0] = state[0]
        y[0] = state[1]
        z[0] = state[2]
        px[0] = state[3]
        py[0] = state[4]
        pz[0] = state[5]

        # computation of taylor coefficients
        for n in range(taylor_coeffs.shape[1] - 1):
            # extended components
            u1[n] = x[n] + rules.constant(mu, n)
            u2[n] = x[n] + rules.constant(gamma1, n)
            v1[n] = rules.product(v1, y, y, n)
            v2[n] = rules.product(v2, z, z, n)
            v3[n] = rules.product(v3, u1, u1, n)
            v4[n] = rules.product(v4, u2, u2, n)
            w1[n] = v1[n] + v2[n]
            w2[n] = w1[n] + v3[n]
            w3[n] = w1[n] + v4[n]
            a1[n] = rules.power(a1, w2, exponent, n)
            a2[n] = rules.power(a2, w3, exponent, n)
            a3[n] = rules.product(a3, u1, a1, n)
            a4[n] = rules.product(a4, u2, a2, n)
            b[n] = gamma1 * a1[n] + gamma2 * a2[n]
            c[n] = rules.product(c, y, b, n)
            d[n] = gamma1 * a3[n] + gamma2 * a4[n]
            f[n] = rules.product(f, z, b, n)

            # next derivate order
            next_n = n + 1
            x[next_n] = (px[n] + y[n]) / next_n
            y[next_n] = (py[n] - x[n]) / next_n
            z[next_n] = pz[n] / next_n
            px[next_n] = (py[n] + d[n]) / next_n
            py[next_n] = (-px[n] + c[n]) / next_n
            pz[next_n] = f[n] / next_n

    return taylor_coeff_func, state_dim, extra_dim
