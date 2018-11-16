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


def generate_variational_taylor_coeffs(mu):
    state_dim = 6
    extra_dim = 49

    gamma1 = mu - 1
    gamma2 = -mu
    exponent1 = -3 / 2
    exponent2 = -5 / 2

    @njit
    def taylor_coeff_func(state, var_state, taylor_coeffs, var_coeffs, ext_coeffs):
        # taylor coefficients shorthands
        x = taylor_coeffs[0]
        y = taylor_coeffs[1]
        z = taylor_coeffs[2]
        px = taylor_coeffs[3]
        py = taylor_coeffs[4]
        pz = taylor_coeffs[5]

        # taylor variational coefficients shorthands
        dx = var_coeffs[0]
        dy = var_coeffs[1]
        dz = var_coeffs[2]
        dpx = var_coeffs[3]
        dpy = var_coeffs[4]
        dpz = var_coeffs[5]

        # auxiliary variable array shorthands
        # for normal DE
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

        # for linearized DE
        g1 = ext_coeffs[17]
        g2 = ext_coeffs[18]
        j1 = ext_coeffs[19]
        j2 = ext_coeffs[20]
        j3 = ext_coeffs[21]
        j4 = ext_coeffs[22]
        h1 = ext_coeffs[23]
        h2 = ext_coeffs[24]
        h3 = ext_coeffs[25]
        v5 = ext_coeffs[26]
        k1 = ext_coeffs[27]
        k2 = ext_coeffs[28]
        k3 = ext_coeffs[29]
        k4 = ext_coeffs[30]
        k5 = ext_coeffs[31]
        c1 = ext_coeffs[32]
        c2 = ext_coeffs[33]
        c3 = ext_coeffs[34]
        l1 = ext_coeffs[35]
        l2 = ext_coeffs[36]
        l3 = ext_coeffs[37]
        l4 = ext_coeffs[38]
        l5 = ext_coeffs[39]
        l6 = ext_coeffs[40]
        n1 = ext_coeffs[41]
        n2 = ext_coeffs[42]
        n3 = ext_coeffs[43]
        e1 = ext_coeffs[44]
        e2 = ext_coeffs[45]
        e3 = ext_coeffs[46]
        s1 = ext_coeffs[47]
        s2 = ext_coeffs[48]

        # initial condition for phase space
        x[0] = state[0]
        y[0] = state[1]
        z[0] = state[2]
        px[0] = state[3]
        py[0] = state[4]
        pz[0] = state[5]

        # initial condition for variational vectors
        dx[0] = var_state[0]
        dy[0] = var_state[1]
        dz[0] = var_state[2]
        dpx[0] = var_state[3]
        dpy[0] = var_state[4]
        dpz[0] = var_state[5]

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
            a1[n] = rules.power(a1, w2, exponent1, n)
            a2[n] = rules.power(a2, w3, exponent1, n)
            a3[n] = rules.product(a3, u1, a1, n)
            a4[n] = rules.product(a4, u2, a2, n)
            b[n] = gamma1 * a1[n] + gamma2 * a2[n]
            c[n] = rules.product(c, y, b, n)
            d[n] = gamma1 * a3[n] + gamma2 * a4[n]
            f[n] = rules.product(f, z, b, n)

            g1[n] = rules.power(g1, w2, exponent2, n)
            g2[n] = rules.power(g2, w3, exponent2, n)
            j1[n] = rules.product(j1, u1, g1, n)
            j2[n] = rules.product(j2, u2, g2, n)
            j3[n] = rules.product(j3, v3, g1, n)
            j4[n] = rules.product(j4, v4, g2, n)
            h1[n] = -gamma1 * g1[n] - gamma2 * g2[n]
            h2[n] = -gamma1 * j1[n] - gamma2 * j2[n]
            h3[n] = -gamma1 * j3[n] - gamma2 * j4[n]
            v5[n] = rules.product(v5, y, z, n)
            k1[n] = rules.product(k1, h1, v5, n)
            k2[n] = rules.product(k2, h1, v1, n)
            k3[n] = rules.product(k3, h1, v2, n)
            k4[n] = rules.product(k4, h2, y, n)
            k5[n] = rules.product(k5, h2, z, n)
            c1[n] = b[n] + 3 * h3[n]
            c2[n] = b[n] + 3 * k2[n]
            c3[n] = b[n] + 3 * k3[n]
            l1[n] = rules.product(l1, k4, dy, n)
            l2[n] = rules.product(l2, k4, dx, n)
            l3[n] = rules.product(l3, k5, dz, n)
            l4[n] = rules.product(l4, k5, dx, n)
            l5[n] = rules.product(l5, k1, dz, n)
            l6[n] = rules.product(l6, k1, dy, n)
            n1[n] = rules.product(n1, c1, dx, n)
            n2[n] = rules.product(n2, c2, dy, n)
            n3[n] = rules.product(n3, c3, dz, n)
            e1[n] = 3 * (l1[n] + l3[n])
            e2[n] = 3 * (l2[n] + l5[n])
            e3[n] = 3 * (l4[n] + l6[n])
            s1[n] = n1[n] + e1[n]
            s2[n] = n2[n] + e2[n]

            # next derivate order
            next_n = n + 1
            x[next_n] = (px[n] + y[n]) / next_n
            y[next_n] = (py[n] - x[n]) / next_n
            z[next_n] = pz[n] / next_n
            px[next_n] = (py[n] + d[n]) / next_n
            py[next_n] = (-px[n] + c[n]) / next_n
            pz[next_n] = f[n] / next_n

            # for k in range(n_var):
            dx[next_n] = (dpx[n] + dy[n]) / next_n
            dy[next_n] = (dpy[n] - dx[n]) / next_n
            dz[next_n] = dpz[n] / next_n
            dpx[next_n] = (dpy[n] + s1[n]) / next_n
            dpy[next_n] = (-dpx[n] + s2[n]) / next_n
            dpz[next_n] = (n3[n] + e3[n]) / next_n

    return taylor_coeff_func, state_dim, extra_dim
