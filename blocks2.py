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
        small_list = tab_short.table[l].poles[:]

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
        if abs(float(p)) > tiny:
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
        c_2 = (ell * (ell + 2 * nu) + delta * (delta - 2 * nu - 2)) / 2
        c_4 = ell * (ell + 2 * nu) * (delta - 1) * (delta - 2 * nu - 1)
        delta_prod = delta_12 * delta_34 / (eval_mpfr(-2, prec))
        delta_sum = (delta_12 - delta_34) / (eval_mpfr(-2, prec))
        if delta_12 == 0 and delta_34 == 0:
            effective_power = 2
        else:
            effective_power = 1

        for l in range(0, l_max + 1, step):
            poles = []
            for k in range(effective_power, k_max + 1, effective_power):
                poles.append(eval_mpfr(1 - k - l, prec))
                poles.append((2 + 2 * nu - k) / eval_mpfr(2, prec))
                poles.append(1 - k + l + 2 * nu)
            pole_set.append(poles)

        l = 0
        while l <= l_max and effective_power == 1:
            frob_coeffs = [1]
            conformal_blocks.append([])
            self.table.append(PolynomialVector([], [l, 0], pole_set[l // step]))

            for k in range(1, k_max + 1):
                # A good check is to force this code to run for identical scalars too
                # This should produce the same blocks as the shorter recursion coming up
                recursion_coeffs = [0, 0, 0, 0, 0, 0, 0]
                recursion_coeffs[0] += 2 * c_2 * (2 * nu + 1) * (4 * delta_sum + 1) - c_4 + 8 * delta_prod * nu * (2 * nu + 1)
                recursion_coeffs[0] -= 2 * (delta + k - 1) * (c_2 * (2 * nu + 1) + 2 * delta_prod * (6 * nu - 1) + 8 * delta_sum * (c_2 + nu - 2 * nu * nu))
                recursion_coeffs[0] += 2 * (delta + k - 1) * (delta + k - 2) * (c_2 + nu - 2 * nu * nu + 4 * delta_prod + 12 * delta_sum * (1 - 2 * nu))
                recursion_coeffs[0] += 2 * (delta + k - 1) * (delta + k - 2) * (delta + k - 3) * (2 * nu - 1 + 8 * delta_sum)
                recursion_coeffs[0] -= 1 * (delta + k - 1) * (delta + k - 2) * (delta + k - 3) * (delta + k - 4)
                recursion_coeffs[1] += 3 * c_4 + 2 * c_2 * (4 * delta_sum * (4 * delta_sum + 2 * nu + 1) + 2 * nu - 3) - 8 * delta_prod * (2 * delta_sum * (1 - 6 * nu) + 6 * nu * nu - 5 * nu)
                recursion_coeffs[1] -= 2 * (delta + k - 2) * (2 * delta_prod * (16 * delta_sum - 10 * nu + 4) + 8 * delta_sum * (c_2 + nu - 2 * nu * nu) + (1 - 2 * nu) * (c_2 + 2 * nu - 2 + 32 * delta_sum * delta_sum))
                recursion_coeffs[1] -= 2 * (delta + k - 2) * (delta + k - 3) * (3 * c_2 + 7 * nu + 2 * nu * nu - 10 + 4 * delta_prod + 4 * delta_sum * (10 * delta_sum + 6 * nu - 3))
                recursion_coeffs[1] += 2 * (delta + k - 2) * (delta + k - 3) * (delta + k - 4) * (7 - 2 * nu + 8 * delta_sum)
                recursion_coeffs[1] += 3 * (delta + k - 2) * (delta + k - 3) * (delta + k - 4) * (delta + k - 5)
                recursion_coeffs[2] += 3 * c_4 + 2 * c_2 * (16 * delta_sum * delta_sum + 2 * nu - 3) + 16 * delta_prod * delta_sum * (8 * delta_sum + 2 * nu + 5)
                recursion_coeffs[2] -= 2 * (delta + k - 3) * ((1 - 2 * nu) * (c_2 + 2 * nu - 2 + 32 * delta_sum * delta_sum - 4 * delta_prod) - 8 * delta_sum * (8 * delta_sum * delta_sum + 4 * delta_prod + 4 * nu * nu + 2 * c_2 - 5))
                recursion_coeffs[2] -= 2 * (delta + k - 3) * (delta + k - 4) * (8 * delta_prod + 40 * delta_sum * delta_sum + 48 * delta_sum + 3 * c_2 + 2 * nu * nu + 7 * nu - 10)
                recursion_coeffs[2] += 2 * (delta + k - 3) * (delta + k - 4) * (delta + k - 5) * (7 - 2 * nu - 16 * delta_sum)
                recursion_coeffs[2] += 3 * (delta + k - 3) * (delta + k - 4) * (delta + k - 5) * (delta + k - 6)
                recursion_coeffs[3] -= 3 * c_4 + 2 * c_2 * (16 * delta_sum * delta_sum + 2 * nu - 3) + 16 * delta_prod * delta_sum * (8 * delta_sum + 2 * nu + 5)
                recursion_coeffs[3] -= 2 * (delta + k - 4) * (12 + 4 * nu - 8 * nu * nu - c_2 * (2 * nu + 5) - 8 * delta_sum * (8 * delta_sum * delta_sum + 8 * delta_sum * nu + 6 * delta_sum + 4 * nu * nu + 2 * c_2 - 5) + 4 * delta_prod * (2 * nu - 5 - 8 * delta_sum))
                recursion_coeffs[3] -= 2 * (delta + k - 4) * (delta + k - 5) * (3 * c_2 + 2 * nu * nu + 7 * nu - 10 + 8 * delta_prod + delta_sum * (40 * delta_sum - 18 * nu + 21))
                recursion_coeffs[3] -= 2 * (delta + k - 4) * (delta + k - 5) * (delta + k - 6) * (16 * delta_sum + 2 * nu + 11)
                recursion_coeffs[3] -= 3 * (delta + k - 4) * (delta + k - 5) * (delta + k - 6) * (delta + k - 7)
                recursion_coeffs[4] -= 3 * c_4 + 2 * c_2 * (4 * delta_sum * (4 * delta_sum + 2 * nu + 1) + 2 * nu - 3) - 8 * delta_prod * (2 * delta_sum * (1 - 6 * nu) + 6 * nu * nu - 5 * nu)
                recursion_coeffs[4] -= 2 * (delta + k - 5) * (12 + 4 * nu - 8 * nu * nu - c_2 * (2 * nu + 5 - 8 * delta_sum) + 2 * delta_prod * (3 - 10 * nu + 16 * delta_sum) - 8 * delta_sum * (2 * nu * nu + 5 * nu + 3 + 8 * delta_sum * nu + 6 * delta_sum))
                recursion_coeffs[4] -= 2 * (delta + k - 5) * (delta + k - 6) * (22 + 5 * nu - 2 * nu * nu - 3 * c_2 - 4 * delta_prod - 4 * delta_sum * (10 * delta_sum + 6 * nu + 9))
                recursion_coeffs[4] += 2 * (delta + k - 5) * (delta + k - 6) * (delta + k - 7) * (8 * delta_sum - 2 * nu - 11)
                recursion_coeffs[4] -= 3 * (delta + k - 5) * (delta + k - 6) * (delta + k - 7) * (delta + k - 8)
                recursion_coeffs[5] -= 2 * c_2 * (2 * nu + 1) * (4 * delta_sum + 1) - c_4 + 8 * delta_prod * nu * (2 * nu + 1)
                recursion_coeffs[5] -= 2 * (delta + k - 6) * ((2 * nu + 3) * (c_2 - 2 * nu - 2) + 6 * delta_prod * (2 * nu + 1) + 8 * delta_sum * (c_2 - 2 * nu * nu - 5 * nu - 3))
                recursion_coeffs[5] -= 2 * (delta + k - 6) * (delta + k - 7) * (c_2 + 4 * delta_prod - (2 * nu + 3) * (nu + 4 + 12 * delta_sum))
                recursion_coeffs[5] += 2 * (delta + k - 6) * (delta + k - 7) * (delta + k - 8) * (2 * nu + 5 + 8 * delta_sum)
                recursion_coeffs[5] += 1 * (delta + k - 6) * (delta + k - 7) * (delta + k - 8) * (delta + k - 9)
                recursion_coeffs[6] = (k + 2 * nu - 5) * (2 * delta + k - 7) * (delta + k - l - 6) * (delta + k + l + 2 * nu - 6)
                recursion_coeffs[5] = recursion_coeffs[5].subs(ell, l)
                recursion_coeffs[4] = recursion_coeffs[4].subs(ell, l)
                recursion_coeffs[3] = recursion_coeffs[3].subs(ell, l)
                recursion_coeffs[2] = recursion_coeffs[2].subs(ell, l)
                recursion_coeffs[1] = recursion_coeffs[1].subs(ell, l)
                recursion_coeffs[0] = recursion_coeffs[0].subs(ell, l)

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
            l += step

        l = 0
        while l <= l_max and effective_power == 2:
            frob_coeffs = [1]
            conformal_blocks.append([])
            self.table.append(PolynomialVector([], [l, 0], pole_set[l // step]))

            for k in range(2, k_max + 1, 2):
                recursion_coeffs = [0, 0, 0]
                recursion_coeffs[0] += 3 * c_4 + 2 * c_2 * (2 * nu - 3)
                recursion_coeffs[0] += 2 * (delta + k - 2) * (2 * nu - 1) * (c_2 + 2 * nu - 2)
                recursion_coeffs[0] += 2 * (delta + k - 2) * (delta + k - 3) * (10 - 7 * nu - 2 * nu * nu - 3 * c_2)
                recursion_coeffs[0] += 2 * (delta + k - 2) * (delta + k - 3) * (delta + k - 4) * (7 - 2 * nu)
                recursion_coeffs[0] += 3 * (delta + k - 2) * (delta + k - 3) * (delta + k - 4) * (delta + k - 5)
                recursion_coeffs[1] += 2 * c_2 * (3 - 2 * nu) - 3 * c_4
                recursion_coeffs[1] += 2 * (delta + k - 4) * (c_2 * (2 * nu + 5) + 8 * nu * nu - 4 * nu - 12)
                recursion_coeffs[1] += 2 * (delta + k - 4) * (delta + k - 5) * (3 * c_2 + 2 * nu * nu - 5 * nu - 22)
                recursion_coeffs[1] -= 2 * (delta + k - 4) * (delta + k - 5) * (delta + k - 6) * (2 * nu + 11)
                recursion_coeffs[1] -= 3 * (delta + k - 4) * (delta + k - 5) * (delta + k - 6) * (delta + k - 7)
                recursion_coeffs[2] = (k + 2 * nu - 4) * (2 * delta + k - 6) * (delta + k - l - 5) * (delta + k + l + 2 * nu - 5)
                recursion_coeffs[1] = recursion_coeffs[1].subs(ell, l)
                recursion_coeffs[0] = recursion_coeffs[0].subs(ell, l)

                pole_prod = 1
                frob_coeffs.append(0)
                for i in range(0, min(k / 2, 3)):
                    frob_coeffs[k / 2] += recursion_coeffs[i] * pole_prod * frob_coeffs[(k / 2) - i - 1] / eval_mpfr(2 * k, prec)
                    frob_coeffs[k / 2] = frob_coeffs[k / 2].expand()
                    if i + 1 < min(k / 2, 3):
                        pole_prod *= (delta - pole_set[l // step][3 * ((k / 2) - i - 2)]) * (delta - pole_set[l // step][3 * ((k / 2) - i - 2) + 1]) * (delta - pole_set[l // step][3 * ((k / 2) - i - 2) + 2])

            pole_prod = 1
            for k in range(k_max // 2, -1, -1):
                frob_coeffs[k] *= pole_prod
                frob_coeffs[k] = frob_coeffs[k].expand()
                if k > 0:
                    pole_prod *= (delta - pole_set[l // step][3 * k - 1]) * (delta - pole_set[l // step][3 * k - 2]) * (delta - pole_set[l // step][3 * k - 3])

            conformal_blocks[l // step] = [0] * (m_max + 1)
            for k in range(0, (k_max // 2) + 1):
                prod = 1
                for m in range(0, m_max + 1):
                    conformal_blocks[l // step][m] += prod * frob_coeffs[k] * (r_cross ** (2 * k - m))
                    conformal_blocks[l // step][m] = conformal_blocks[l // step][m].expand()
                    prod *= (delta + 2 * k - m)
            l += step

        (rules1, rules2, self.m_order, self.n_order) = rules(m_max, 0)
        chain_rule_single(self.m_order, rules1, self.table, conformal_blocks, lambda l, i: conformal_blocks[l][i])

        # Find the superfluous poles (including possible triple poles) to cancel
        for l in range(0, len(self.table)):
            cancel_poles(self.table[l])
