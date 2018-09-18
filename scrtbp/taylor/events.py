import numpy as np
from numba import njit

from scrtbp.taylor.expansion import TaylorExpansion, generate_func_adapter


@njit
def bracket_iteration(func_adapter, bracket_cache):
    """root finding algorithm; every step is bisection + Brent or Ridder, 
       Regula Falsi as a fallback 1,2 (or none, if all of them fail);
        only works for bracketed roots (sign change)!
        takes in taylor coeff and  2x2 matrix of the form [[a,f(a)],[b,f(b)]]
    """
    # determines left/right side of interval or if interval is degenerate
    if bracket_cache[0, 0] < bracket_cache[1, 0]:
        left_x = bracket_cache[0, 0]
        left_fx = bracket_cache[0, 1]
        right_x = bracket_cache[1, 0]
        right_fx = bracket_cache[1, 1]
    elif bracket_cache[0, 0] > bracket_cache[1, 0]:
        right_x = bracket_cache[0, 0]
        right_fx = bracket_cache[0, 1]
        left_x = bracket_cache[1, 0]
        left_fx = bracket_cache[1, 1]
    else:
        # if interval is [a, a] then a is root
        return bracket_cache[0, 0]

    if left_fx * right_fx > 0.0:
        raise Exception(
            "Interval doesn't contain a root! f(a) < 0 < f(b) or f(a) > 0 > f(b)"
        )

    x1 = left_x
    f1 = left_fx
    x3 = right_x
    f3 = right_fx

    # bisection for additional point
    x2 = (x1 + x3) / 2.0
    f2 = func_adapter.eval(x2)

    # check and update brackets
    if left_fx * f2 < 0.0:
        right_x = x2
        right_fx = f2
    elif f2 * right_fx < 0.0:
        left_x = x2
        left_fx = f2
    # cases, where root is already found
    elif f2 == 0.0:
        bracket_cache[0, 0] = x2
        bracket_cache[0, 1] = f2
        bracket_cache[1, 0] = x2
        bracket_cache[1, 1] = f2
        return x2
    elif f1 == 0.0:
        bracket_cache[0, 0] = x1
        bracket_cache[0, 1] = f1
        bracket_cache[1, 0] = x1
        bracket_cache[1, 1] = f1
        return x1
    elif f3 == 0.0:
        bracket_cache[0, 0] = x3
        bracket_cache[0, 1] = f3
        bracket_cache[1, 0] = x3
        bracket_cache[1, 1] = f3
        return x3
    else:
        raise Exception("BIG UNHANDLED PROBLEM")

    first_fallback = False
    second_fallback = False
    x = left_x  # x is new estimation of root
    # here: chosen arbitrarily, such that not x_left < x (initial x outside bracket)

    # check if function is monotonic
    if (f1 < f2 and f2 < f3) or (f1 > f2 and f2 > f3):
        # inverse quadratic interpolation in Brent-Dekker method
        diff21 = f2 - f1
        diff23 = f2 - f3
        diff31 = f3 - f1

        num1 = f2 * f3
        num2 = f1 * f3
        num3 = f2 * f1

        denom1 = diff21 * diff31
        denom2 = diff21 * diff23
        denom3 = -diff23 * diff31

        eta = (x2 - x1) / (x3 - x1)
        phi = diff21 / diff31
        phi2 = 1.0 - phi

        if phi * phi < eta and 1.0 - eta > phi2 * phi2:
            x = x1 * (num1 / denom1) + x2 * (num2 / denom2) + x3 * (
                num3 / denom3)

            # check if result is in the current bracket
            if not (left_x < x and x < right_x):
                #print("INTERPOLATION FAILED (interpolation not landed) 1")
                first_fallback = True
        else:
            # print("INTERPOLATION FAILED (conditions not met)")
            first_fallback = True
    else:
        # print("SKIPPED INTERPOLATION (f(x) not monotonic)")
        first_fallback = True

    if first_fallback:
        # print("FIRST FALLBACK to RIDDER")
        radicand = f2 * f2 - f1 * f3

        if 0.0 < radicand:
            x = x2 + (x2 - x1) * (np.sign(f1) * f2 / np.sqrt(radicand))

            # check if result is in the current bracket
            if not (left_x < x and x < right_x):
                second_fallback = True
                #print("INTERPOLATION FAILED (interpolation not landed) 2")
        else:
            second_fallback = True

    if second_fallback:
        # print("SECOND FALLBACK to REGULA FALSI")
        # checks if secant has slope 0
        if (right_fx - left_fx) != 0.0:
            x = (left_x * right_fx - right_x * left_fx) / (right_fx - left_fx)

    # check if result is in the current bracket
    if left_x < x and x < right_x:
        # interpolation or fallback landed
        fx = func_adapter.eval(x)

        # rebracket root
        if left_fx * fx < 0.0:
            right_x = x
            right_fx = fx
        elif fx * right_fx < 0.0:
            left_x = x
            left_fx = fx
        elif fx == 0.0:
            left_x = x
            left_fx = fx
            right_x = x
            right_fx = fx
        else:
            print("BIG UNHANDLED PROBLEM 2")
    else:
        pass
        #print("FALLBACK FAILED (not inside interval)")
        # only bisection was conducted during this step

    # update bracket cache
    bracket_cache[0, 0] = left_x
    bracket_cache[0, 1] = left_fx
    bracket_cache[1, 0] = right_x
    bracket_cache[1, 1] = right_fx

    # interval endpoint closer to zero is new approximation of root
    if abs(left_fx) < abs(right_fx):
        return left_x
    else:
        return right_x


def generate_event_solver(taylor_coeff_func,
                          poincare_char_func,
                          state_dim,
                          extra_dim,
                          step=0.01,
                          order=30,
                          max_steps=1000000,
                          one_way_mode=True):
    FuncAdapter = generate_func_adapter(poincare_char_func)

    if one_way_mode:
        def py_root_condition(fa, fb):
            return fa < 0.0 and 0.0 < fb
    else:
        def py_root_condition(fa, fb):
            return fa * fb < 0.0

    root_condition = njit(py_root_condition)

    @njit
    def solve_points(input_state, n_points, t0=0.0):
        """
            solves trajectory step wise, looks if its trajectory 
            crosses the poincare section during a step via 
            root search, saves n_points many section points 
            and their time of section

            n_points = number of points inside poincare section to be calculated
        """
        points = np.empty((n_points, state_dim))
        t_points = np.empty(n_points)

        extra_coeffs = np.empty((extra_dim, order))
        series = TaylorExpansion(np.empty((state_dim, order + 1)))
        event_func = FuncAdapter(series)

        state_now = np.empty(state_dim)
        state_next = np.empty(state_dim)
        brackets = np.empty((2, 2))

        state_next = input_state
        t_next = t0
        f_next = event_func.eval_state(state_next)
        step_num = 0

        for i in range(n_points):
            # calculates trajectory steps until intersection is found
            # (or unfavourably max_steps is reached)
            for j in range(max_steps):
                state_now = state_next
                t_now = t_next
                f_now = f_next

                # compute taylor series around state_now
                taylor_coeff_func(state_now, series.coeffs, extra_coeffs)

                # previously j, which is reset after each new point !
                t_next = t0 + (step_num + 1) * step
                f_next = event_func.eval(step)
                state_next = event_func.state_cache

                step_num = step_num + 1

                # searches for root in current step interval t_now, t_next
                if root_condition(f_now, f_next):
                    # 0 instead of t_now, since taylor stutzstelle already is t_now
                    brackets[0, 0] = 0.0
                    brackets[0, 1] = f_now
                    brackets[1, 0] = step
                    brackets[1, 1] = f_next

                    # bracket size
                    diff_curr = abs(brackets[0, 0] - brackets[1, 0])
                    # hack: just need a worse diff (only used once)
                    diff_prev = 2.0 * diff_curr

                    # until bracket size doesn't change anymore, do ...
                    while diff_prev != diff_curr:
                        root_step = bracket_iteration(event_func, brackets)
                        diff_prev = diff_curr
                        diff_curr = abs(brackets[0, 0] - brackets[1, 0])

                    # uses taylor series around t_now to approximate point[i]
                    series.eval(root_step, points[i])
                    t_points[i] = t_now + root_step
                    break
                elif f_now == 0.0:
                    # immediatly found a root
                    points[i] = state_now
                    t_points[i] = t_now
                    break
                elif f_next == 0.0:
                    # ignore upcoming root, will be caught in next step
                    pass
                else:
                    # no root in interval, continue steps
                    pass
            # previously: if max_steps is exceeded, point[i] will be random and make no sense!
            # if max_steps is exceeded, print error, return points
            # ATTENTION: points from i onward will be random remnants from np.empty!
            if j == (max_steps - 1):
                #raise Exception('ERROR: max_steps exceeded')
                print('ERROR: max_steps ', max_steps, ' exceeded')
                return points, t_points

        return points, t_points

    return solve_points
