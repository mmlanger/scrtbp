from numba import njit


@njit
def deriv(x, n):
    if n < 2:
        return x[n - 1]
    return x[n - 1] / n


@njit
def constant(constant, n):
    if n == 0:
        return constant
    return 0.0


@njit
def product(w, u, v, n):
    if n == 0:
        return u[0] * v[0]

    result = 0.0
    c = 0.0
    temp = 0.0
    for i in range(n + 1):
        element = u[i] * v[n - i]
        temp = result + element
        if abs(result) >= abs(element):
            c += (result - temp) + element
        else:
            c += (element - temp) + result
        result = temp

    return result + c


@njit
def quotient(r, u, v, n):
    if n == 0:
        return u[0] / v[0]

    result = 0.0
    c = 0.0
    temp = 0.0
    for i in range(1, n + 1):
        element = r[i] * v[n - i]
        temp = result + element
        if abs(result) >= abs(element):
            c += (result - temp) + element
        else:
            c += (element - temp) + result
        result = temp

    return (u[n] - (result + c)) / v[0]


@njit
def power(p, u, alpha, n):
    if n == 0:
        return u[0]**alpha

    result = 0.0
    c = 0.0
    temp = 0.0

    for i in range(n):
        int_ratio = i / n
        prefac = alpha * (1.0 - int_ratio) - int_ratio
        element = prefac * p[i] * u[n - i]
        temp = result + element
        if abs(result) >= abs(element):
            c += (result - temp) + element
        else:
            c += (element - temp) + result
        result = temp

    return (result + c) / u[0]
