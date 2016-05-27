def convert_table(tab_short, tab_long):
    """
    Converts a table with few poles into an equivalent table with many poles.
    When tables produced by different methods fail to look the same, it is often
    because their polynomials are being multiplied by different positive
    prefactors. This adjusts the prefactors so that they are the same.

    Parameters
    ----------
    tab_short: A `ConformalBlockTable` where the blocks have a certain number of
               poles which is hopefully optimal.
    tab_long:  A `ConformalBlockTable` with all of the poles that `tab_short` has
               plus more.
    """
    for l in range(0, len(tab_short.table)):
        pole_prod = 1
        small_list = short_table[l].poles[:]

        for p in tab_long.table[l].poles:
            index = get_index_approx(small_list, p)

            if index == -1:
                pole_prod *= delta - p
                tab_short.table[l].poles.append(p)
            else:
                small_list.remove(small_list[index])

        for n in range(0, len(tab_short.table[l].vector)):
            tab_short.table[l].vector[n] = tab_short.table[l].vector[n] * pole_prod
            tab_short.table[l].vector[n] = tab_short.table[l].vector[n].expand()

def cancel_poles(polynomial_vector):
    """
    Checks which roots of a conformal block denominator are also roots of the
    numerator. Whenever one is found, a simple factoring is applied.

    Parameters
    ----------
    polynomial_vector: The `PolynomialVector` that will be modified in place if
                       it has superfluous poles.
    """
    poles = []
    zero_poles = []
    for p in polynomial_vector.poles:
        if abs(p) > tiny:
            poles.append(p)
        else:
            zero_poles.append(p)
        poles = zero_poles + poles

    for p in poles:
        # We should really make sure the pole is a root of all numerators
        # However, this is automatic if it is a root before differentiating
        if abs(polynomial_vector.vector[0].subs(delta, p)) < tiny:
            polynomial_vector.poles.remove(p)

            # A factoring algorithm which works if the zeros are first
            for n in range(0, len(polynomial_vector.vector)):
                coeffs = coefficients(polynomial_vector.vector[n])
                if abs(p) > tiny:
                    new_coeffs = [coeffs[0] / eval_mpfr(-p, prec)]
                    for i in range(1, len(coeffs) - 1):
                        new_coeffs.append((new_coeffs[i - 1] - coeffs[i]) / eval_mpfr(p, prec))
                else:
                    coeffs.remove(coeffs[0])
                    new_coeffs = coeffs

                prod = 1
                polynomial_vector.vector[n] = 0
                for i in range(0, len(new_coeffs)):
                    polynomial_vector.vector[n] += prod * new_coeffs[i]
                    prod *= delta

class ConformalBlockTableSeed2:
    """
    A class which calculates tables of conformal block derivatives from scratch
    using a power series solution of their fourth order differential equation.
    Usually, it will not be necessary for the user to call it. Instead,
    `ConformalBlockTable` calls it automatically for `m_max = 3`. Note that there
    is no `n_max` for this method.
    """
    def __init__(self, dim, k_max, l_max, m_max, delta_12 = 0, delta_34 = 0, odd_spins = False):
        self.dim = dim
        self.k_max = k_max
        self.l_max = l_max
        self.m_max = m_max
        self.delta_12 = delta_12
        self.delta_34 = delta_34
        self.odd_spins = odd_spins
        self.m_order = []
        self.n_order = []
        self.table = []

        if odd_spins:
            step = 1
        else:
            step = 2

        pole_set = []
        conformal_blocks = []
        nu = eval_mpfr((dim / Integer(2)) - 1, prec)
        delta_prod = delta_12 * delta_34 / (eval_mpfr(-2, prec))
        delta_sum = (delta_12 - delta_34) / (eval_mpfr(-2, prec))

        for l in range(0, l_max + 1, step):
            poles = []
            for k in range(1, k_max + 1):
                poles.append(eval_mpfr(1 - k - l, prec))
                poles.append((2 + 2 * nu - k) / eval_mpfr(2, prec))
                poles.append(1 - k + l + 2 * nu)
            pole_set.append(poles)

        for l in range(0, l_max + 1, step):
            frob_coeffs = [1]
            conformal_blocks.append([])
            self.table.append(PolynomialVector([], [l, 0], pole_set[l // step]))

            for k in range(1, k_max + 1):
                # In the identical scalar case, there would be only three
                # The min(k, 7) would also be min(k, 3)
                recursion_coeffs = [0, 0, 0, 0, 0, 0, 0]

                # These have been copied from JuliBoots
                # Explicitly including c_2 and c_4 could aid brevity
                recursion_coeffs[0] += (k ** 4) * (-1)
                recursion_coeffs[0] += (k ** 3) * 4 * (2 - delta + 4 * delta_sum + nu)
                recursion_coeffs[0] += (k ** 2) * (l * l + 2 * l * nu - 4 * nu * nu - 22 * nu - 23 - delta * (5 * delta - 10 * nu - 22) + 24 * delta_sum * (2 * delta - 2 * nu - 3) + 8 * delta_prod)
                recursion_coeffs[0] += (k ** 1) * 2 * ((delta - nu - 2) * (l * l - 7 - delta * (delta - 6) + 2 * nu * (delta + l - 3)) + 4 * delta_sum * (l * l - 4 * nu * nu - 13 - delta * (5 * delta - 16) + 2 * nu * (5 * delta + l - 8)) - 2 * delta_prod * (4 * delta - 6 * nu - 5))
                recursion_coeffs[0] += (k ** 0) * (3 + 2 * nu - 2 * delta) * (4 * delta_prod * (1 + 4 * nu - delta) - (4 * delta_sum + 1) * (delta + l - 2) * (delta - l - 2 * nu - 2))
                recursion_coeffs[1] += (k ** 4) * 3
                recursion_coeffs[1] += (k ** 3) * 4 * (3 * delta - nu - 7 + 4 * delta_sum)
                recursion_coeffs[1] += (k ** 2) * (3 * delta * (5 * delta - 2 * nu - 26) + 2 * nu * (11 - 3 * l - 2 * nu) - 3 * l * l + 107 - 8 * delta_prod - 8 * delta_sum * (10 * delta_sum + 15 - 6 * nu - 6 * delta))
                recursion_coeffs[1] += (k ** 1) * 2 * (3 * delta * delta * delta + delta * delta * (nu - 29) - 3 * delta * (2 * nu * (nu + l - 1) + l * l - 31) - nu * (23 - 14 * l - l * l - 14 * nu - 2 * l * nu) + 7 * l * l - 97 + 2 * delta_prod * (16 * delta_sum + 4 * delta - 10 * nu - 79) + 8 * delta_sum * delta_sum * (10 * delta - 8 * nu - 21) + 4 * delta_sum * (delta * (28 - 5 * delta) + 2 * nu * (5 * delta + l - 14) - 4 * nu * nu + l * l - 37))
                recursion_coeffs[1] += (k ** 0) * 4 * (delta * delta * delta * (nu - 2) - delta * delta * (2 * nu * nu + 3 * nu - 15) - delta * (l * (l + 2 * nu) * (nu - 2) - 8 * nu * nu + nu + 39) + 34 - 4 * l * l + nu * (2 * l * nu - 10 * nu + l * l - 8 * l + 9) + delta_prod * (4 * delta_sum * (7 + 6 * nu - 4 * delta) - 18 * delta * delta + delta * (34 * nu + 79) - nu * (12 * nu - 8 * l + 74) + 4 * l * l - 86) + delta_sum * (5 + 2 * nu - 2 * delta) * (l * l + 2 * nu * (delta + l - 4) - (delta - 2) * (delta - 6)) + 4 * (delta_sum * delta_sum - delta_prod) * (4 * delta * delta - delta * (19 + 6 * nu) - 2 * nu * (l - 8) + l * l - 22))
                recursion_coeffs[2] += (k ** 4) * 3
                recursion_coeffs[2] += (k ** 3) * 4 * (3 * delta - nu - 10 - 8 * delta_sum)
                recursion_coeffs[2] += (k ** 2) * (3 * delta * (5 * delta - 2 * nu - 38) - 2 * nu * (2 * nu + 3 * l - 17) - 3 * l * l + 209 - 16 * delta_prod - 16 * delta_sum * (5 * delta_sum + 6 * delta - 18))
                recursion_coeffs[2] += (k ** 1) * 2 * (3 * delta * delta * delta + delta * delta * (nu - 44) - 3 * delta * (2 * nu * (nu + l - 2) + l * l - 63) + nu * (2 * l * nu + 18 * nu + l * l + 20 * l - 51) + 10 * l * l - 252 + 32 * delta_sum * (delta_prod + 2 * delta_sum * delta_sum) + 8 * (delta_sum * delta_sum - delta_prod) * (31 + 8 * nu - 10 * delta) - 8 * delta_sum * (5 * delta * delta + 2 * delta * (nu - 17) - 2 * nu * (2 * nu + l) - l * l + 57))
                recursion_coeffs[2] += (k ** 0) * (2 * delta * delta * delta * (2 * nu - 7) + delta * delta * (133 - 14 * nu - 8 * nu) - 2 * delta * (2 * nu * nu * (2 * l - 11) + nu * (2 * l * l - 14 * l + 11) - 7 * l * l + 216) + nu * (4 * l * nu - 72 * nu + 2 * l * l - 66 * l + 108) - 33 * l * l + 468 + 64 * delta_sum * (2 * delta_sum * delta_sum - 3 * delta_prod) * (delta - 3) + 16 * (delta_sum * delta_sum - delta_prod) * (8 * delta_prod + delta * (29 + 6 * nu - 4 * delta) + 2 * nu * (l - 12) + l * l - 48) + 16 * delta_sum * (delta - 3) * (l * l - (delta - 3) * (delta - 7) - 2 * nu * (delta - l) + 4 * nu * nu) + 16 * delta_prod * delta_sum * (16 * delta - 10 * nu - 41) + 8 * delta_prod * (delta * (73 + 10 * nu - 10 * delta) + 2 * nu * (4 * nu + 2 * l - 27) + 2 * l * l - 123))
                recursion_coeffs[3] += (k ** 4) * (-3)
                recursion_coeffs[3] += (k ** 3) * 4 * (11 - nu - 3 * delta - 8 * delta_sum)
                recursion_coeffs[3] += (k ** 2) * (-15 * delta * delta - 18 * delta * (nu - 7) + nu * (4 * nu + 6 * l + 50) + 3 * l * l - 251 + delta_sum * (80 * delta_sum + 96 * (4 - delta)) + 16 * delta_prod)
                recursion_coeffs[3] += (k ** 1) * 2 * (-3 * delta * delta * delta + (49 - 11 * nu) * delta * delta + (3 * l * l + 2 * nu * (nu + 3 * l + 35) - 229) * delta + nu * (2 * l * nu - 10 * nu + l * l - 22 * l - 107) - 11 * l * l + 329 + 64 * delta_sum * delta_sum * delta_sum + 4 * delta_prod * (24 * delta + 14 * nu - 91 - 56 * delta_sum) + 8 * delta_sum * (delta * (46 - 5 * delta - 2 * nu) + 2 * nu * (2 * nu + l) + l * l - 99) + 8 * (delta_sum * delta_sum - delta_prod) * (10 * delta + 8 * nu - 39))
                recursion_coeffs[3] += (k ** 0) * (-4) * (2 * delta * delta * delta * (nu - 2) + delta * delta * (41 - 20 * nu) + delta * (2 * nu * (35 - l * l + 4 * l - 2 * l * nu) + 4 * l * l - 143) + 2 * (nu * (4 * l * nu - 2 * nu + 2 * l * l - 10 * l - 39) - 5 * l * l + 83) + 2 * delta_prod * (16 * delta_sum * delta_sum + delta * (87 - 18 * nu - 10 * delta) + 4 * nu * (2 * nu + l + 11) + 2 * l * l - 172) + 16 * (4 - delta) * delta_sum * (2 * delta_sum * delta_sum - 3 * delta_prod) + 4 * (delta_sum * delta_sum - delta_prod) * (delta * (37 - 10 * nu - 4 * delta) + 2 * nu * (l + 16) + l * l - 76) + 4 * delta_prod * delta_sum * (16 * delta + 10 * nu - 71) - 4 * delta_sum * (4 - delta) * (delta * (delta + 2 * nu - 14) - 2 * nu * (2 * nu + l) - l * l + 35))
                recursion_coeffs[4] += (k ** 4) * (-3)
                recursion_coeffs[4] += (k ** 3) * 4 * (14 - nu - 3 * delta + 4 * delta_sum)
                recursion_coeffs[4] += (k ** 2) * (-15 * delta * delta - 18 * delta * (nu - 9) + 2 * nu * (2 * nu + 3 * l + 31) + 3 * l * l - 401 + 80 * delta_sum * delta_sum + 24 * delta_sum * (2 * delta + 2 * nu - 9) + 8 * delta_prod)
                recursion_coeffs[4] += (k ** 1) * (-2) * (delta * delta * (3 * delta + 11 * nu - 64) - delta * (2 * nu * (nu + 3 * l + 44) + 3 * l * l - 373) + nu * (14 * nu - 2 * l * nu - l * l + 28 * l + 163) + 14 * l * l - 652 + 32 * delta_prod * delta_sum + 8 * (delta_sum * delta_sum - delta_prod) * (49 - 8 * nu - 10 * delta) + 4 * delta_sum * (delta * (52 - 14 * nu - 5 * delta) - 2 * nu * (2 * nu - 28 - l) + l * l - 121) + 2 * delta_prod * (221 - 42 * nu - 44 * delta))
                recursion_coeffs[4] += (k ** 0) * (2 * delta * delta * delta * (11 - 4 * nu) + delta * delta * (102 * nu - 277) + 2 * delta * (nu * (8 * l * nu - 2 * nu + 4 * l * l - 22 * l - 219) - 11 * l * l + 584) + 2 * nu * (20 * nu - 18 * l * nu - 9 * l * l + 65 * l + 290) + 65 * l * l - 1620 - 16 * (delta_sum * delta_sum - delta_prod) * (delta * (47 - 10 * nu - 4 * delta) + 2 * nu * (20 + l) + l * l - 120) + 4 * delta_sum * (2 * delta + 2 * nu - 9) * (delta * (delta + 6 * nu - 16) - 2 * nu * (l + 10) - l * l + 40) - 16 * delta_prod * delta_sum * (4 * delta + 6 * nu - 21) + 4 * delta_prod * (delta * (18 * delta + 50 * nu - 213) + 4 * nu * (3 * nu - 2 * l - 55) - 4 * l * l + 555))
                recursion_coeffs[5] += (k ** 4)
                recursion_coeffs[5] += (k ** 3) * 4 * (delta + nu - 5 + 4 * delta_sum)
                recursion_coeffs[5] += (k ** 2) * (5 * delta * delta + 2 * (7 * nu - 29) * delta + 2 * nu * (2 * nu - l - 31) - l * l + 149 - 24 * delta_sum * (11 - 2 * nu - 2 * delta) - 8 * delta_prod)
                recursion_coeffs[5] += (k ** 1) * (-2) * ((delta + nu - 5) * (delta * (18 - 6 * nu - delta) + 2 * nu * (l + 11) + l * l - 49) - 4 * delta_sum * (delta * (5 * delta + 14 * nu - 64) + 2 * nu * (2 * nu - l - 34) - l * l + 181) + 2 * delta_prod * (4 * delta + 6 * nu - 23))
                recursion_coeffs[5] += (k ** 0) * 4 * ((nu - 2) * (delta - l - 5) * (delta - 3) * (delta + 2 * nu + l - 5) + (2 * delta + 2 * nu - 11) * (delta_sum * (delta * (delta + 6 * nu - 20) - 2 * nu * (l + 12) - l * l + 60) - delta_prod * (delta + 2 * nu - 6)))
                recursion_coeffs[6] += (k ** 4)
                recursion_coeffs[6] += (k ** 3) * 4 * (delta + nu - 6)
                recursion_coeffs[6] += (k ** 2) * (delta * (5 * delta + 14 * nu - 70) + 2 * nu * (2 * nu - l - 37) - l * l + 215)
                recursion_coeffs[6] += (k ** 1) * (-2) * (delta + nu - 6) * (delta * (22 - 6 * nu - delta) + 2 * nu * (l + 13) + l * l - 71)
                recursion_coeffs[6] += (k ** 0) * (2 * nu - 5) * (delta - l - 6) * (2 * delta - 7) * (delta + 2 * nu + l - 6)

                pole_prod = 1
                frob_coeffs.append(0)
                for i in range(0, min(k, 7)):
                    frob_coeffs[k] += recursion_coeffs[i] * pole_prod * frob_coeffs[k - i - 1] / eval_mpfr(2 * k, prec)
                    frob_coeffs[k] = frob_coeffs[k].expand()
                    if i + 1 < min(k, 7):
                        pole_prod *= (delta - pole_set[l // step][3 * (k - i - 2)]) * (delta - pole_set[l // step][3 * (k - i - 2) + 1]) * (delta - pole_set[l // step][3 * (k - i - 2) + 2])

            # We have solved for the Frobenius coefficients times products of poles
            # Fix them so that they all carry the same product
            pole_prod = 1
            for k in range(k_max, -1, -1):
                frob_coeffs[k] *= pole_prod
                frob_coeffs[k] = frob_coeffs[k].expand()
                if k > 0:
                    pole_prod *= (delta - pole_set[l // step][3 * k - 1]) * (delta - pole_set[l // step][3 * k - 2]) * (delta - pole_set[l // step][3 * k - 3])

            conformal_blocks[l // step] = [0] * (m_max + 1)
            for k in range(0, k_max + 1):
                prod = 1
                for m in range(0, m_max + 1):
                    conformal_blocks[l // step][m] += prod * frob_coeffs[k] * (r_cross ** (k - m))
                    conformal_blocks[l // step][m] = conformal_blocks[l // step][m].expand()
                    prod *= (delta + k - m)

        (rules1, rules2, self.m_order, self.n_order) = rules(m_max, 0)
        chain_rule_single(self.m_order, rules1, self.table, conformal_blocks, lambda l, i: conformal_blocks[l][i])

        # Find the superfluous poles (including possible triple poles) to cancel
        for l in range(0, len(self.table)):
            cancel_poles(self.table[l])
