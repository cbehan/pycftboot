def delta_pole(nu, k, l, series):
    """
    Returns the pole of a meromorphic global conformal block given by the
    parameters in arXiv:1406.4858 by Kos, Poland and Simmons-Duffin.

    Parameters
    ----------
    nu:     `(d - 2) / 2` where d is the spatial dimension.
    k:      The parameter k indexing the various poles. As described in
            arXiv:1406.4858, it may be any positive integer unless `series` is 3.
    l:      The spin.
    series: The parameter i desribing the three types of poles in arXiv:1406.4858.
    """
    if series == 1:
        pole = 1 - l - k
    elif series == 2:
        pole = 1 + nu - k
    else:
        pole = 1 + l + 2 * nu - k

    return eval_mpfr(pole, prec)

def delta_residue(nu, k, l, delta_12, delta_34, series):
    """
    Returns the residue of a meromorphic global conformal block at a particular
    pole in `delta`. These residues were found by Kos, Poland and Simmons-Duffin
    in arXiv:1406.4858.

    Parameters
    ----------
    nu:       `(d - 2) / 2` where d is the spatial dimension. This must be
              different from an integer.
    k:        The parameter k indexing the various poles. As described in
              arXiv:1406.4858, it may be any positive integer unless `series`
              is 3.
    l:        The spin.
    delta_12: The difference between the external scaling dimensions of operator
              1 and operator 2.
    delta_34: The difference between the external scaling dimensions of operator
              3 and operator 4.
    series:   The parameter i desribing the three types of poles in
              arXiv:1406.4858.
    """
    two = eval_mpfr(2, prec)
    # Time saving special case
    if series != 2 and k % 2 != 0 and delta_12 == 0 and delta_34 == 0:
        return 0

    if series == 1:
        ret = - ((k * (-4) ** k) / (factorial(k) ** 2)) * sympy.rf((1 - k + delta_12) / two, k) * sympy.rf((1 - k + delta_34) / two, k)
        if l == 0 and nu == 0:
            # Take l to 0, then nu
            return ret * 2
        else:
            return ret * (sympy.rf(l + 2 * nu, k) / sympy.rf(l + nu, k))
    elif series == 2:
        factors = [l + nu + 1 - delta_12, l + nu + 1 + delta_12, l + nu + 1 - delta_34, l + nu + 1 + delta_34]
        ret = ((k * sympy.rf(nu + 1, k - 1)) / (factorial(k) ** 2)) * ((l + nu - k) / (l + nu + k))
        ret *= sympy.rf(-nu, k + 1) / ((sympy.rf((l + nu - k + 1) / 2, k) * sympy.rf((l + nu - k) / 2, k)) ** 2)

        for f in factors:
            ret *= sympy.rf((f - k) / 2, k)
        return ret
    else:
        return - ((k * (-4) ** k) / (factorial(k) ** 2)) * (sympy.rf(1 + l - k, k) * sympy.rf((1 - k + delta_12) / two, k) * sympy.rf((1 - k + delta_34) / two, k) / sympy.rf(1 + nu + l - k, k))

class LeadingBlockVector:
    def __init__(self, dim, l, m_max, n_max, delta_12, delta_34):
        self.spin = l
        self.m_max = m_max
        self.n_max = n_max
        self.chunks = []

        r = Symbol('r')
        eta = Symbol('eta')
        nu = (dim / Integer(2)) - 1
        derivative_order = m_max + 2 * n_max

        # With only a derivatives, we never need eta derivatives
        off_diag_order = derivative_order
        if n_max == 0:
            off_diag_order = 0

        # We cache derivatives as we go
        # This is because csympy can only compute them one at a time, but it's faster anyway
        old_expression = self.leading_block(nu, r, eta, l, delta_12, delta_34)

        for n in range(0, off_diag_order + 1):
            chunk = []
            for m in range(0, derivative_order - n + 1):
                if n == 0 and m == 0:
                    expression = old_expression
                elif m == 0:
                    old_expression = old_expression.diff(eta)
                    expression = old_expression
                else:
                    expression = expression.diff(r)

                chunk.append(expression.subs({r : r_cross, eta : 1}))
            self.chunks.append(DenseMatrix(len(chunk), 1, chunk))

    def leading_block(self, nu, r, eta, l, delta_12, delta_34):
        if self.n_max == 0:
            ret = 1
        elif nu == 0:
            ret = sympy.chebyshevt(l, eta)
        else:
            ret = factorial(l) * sympy.gegenbauer(l, nu, eta) / sympy.rf(2 * nu, l)

        one = eval_mpfr(1, prec)
        two = eval_mpfr(2, prec)

        # Time saving special case
        if delta_12 == delta_34:
            return ((-1) ** l) * ret / (((1 - r ** 2) ** nu) * sqrt((1 + r ** 2) ** 2 - 4 * (r * eta) ** 2))
        else:
            return ((-1) ** l) * ret / (((1 - r ** 2) ** nu) * ((1 + r ** 2 + 2 * r * eta) ** ((one + delta_12 - delta_34) / two)) * ((1 + r ** 2 - 2 * r * eta) ** ((one - delta_12 + delta_34) / two)))

class MeromorphicBlockVector:
    def __init__(self, leading_block):
        # A chunk is a set of r derivatives for one eta derivative
        # The matrix that should multiply a chunk is just R restricted to the right length
        self.chunks = []

        for j in range(0, len(leading_block.chunks)):
            rows = leading_block.chunks[j].nrows()
            self.chunks.append(DenseMatrix(rows, 1, [0] * rows))
            for n in range(0, rows):
                self.chunks[j].set(n, 0, leading_block.chunks[j].get(n, 0))

class ConformalBlockVector:
    def __init__(self, dim, l, delta_12, delta_34, derivative_order, kept_pole_order, s_matrix, leading_block, pol_list, res_list):
        self.large_poles = []
        self.small_poles = []
        self.chunks = []

        nu = (dim / Integer(2)) - 1
        old_list = MeromorphicBlockVector(leading_block)
        for k in range(0, len(pol_list)):
            pole = delta_pole(nu, pol_list[k][1], l, pol_list[k][3])

            if abs(float(res_list[k].chunks[0].get(0, 0))) < cutoff:
                self.small_poles.append(pole)
            else:
                self.large_poles.append(pole)

        matrix = []
        if self.small_poles != []:
            for i in range(0, len(self.large_poles) // 2):
                for j in range(0, len(self.large_poles)):
                    matrix.append(1 / ((cutoff + unitarity_bound(dim, l) - self.large_poles[j]) ** (i + 1)))
            for i in range(0, len(self.large_poles) - (len(self.large_poles) // 2)):
                for j in range(0, len(self.large_poles)):
                    matrix.append(1 / (((1 / cutoff) - self.large_poles[j]) ** (i + 1)))
            matrix = DenseMatrix(len(self.large_poles), len(self.large_poles), matrix)
            matrix = matrix.inv()

        for j in range(0, len(leading_block.chunks)):
            self.chunks.append(leading_block.chunks[j])
            for p in self.large_poles:
                self.chunks[j] = self.chunks[j].mul_scalar(delta - p)

        for k in range(0, len(pol_list)):
            pole = delta_pole(nu, pol_list[k][1], l, pol_list[k][3])

            if pole in self.large_poles:
                for j in range(0, len(self.chunks)):
                    self.chunks[j] = self.chunks[j].add_matrix(res_list[k].chunks[j].mul_scalar(omit_all(self.large_poles, [pole], delta)))
            else:
                vector = []
                for i in range(0, len(self.large_poles) // 2):
                    vector.append(1 / ((unitarity_bound(dim, l) - pole) ** (i + 1)))
                for i in range(0, len(self.large_poles) - (len(self.large_poles) // 2)):
                    vector.append(1 / (((1 / cutoff) - pole) ** (i + 1)))
                vector = DenseMatrix(len(self.large_poles), 1, vector)
                vector = matrix.mul_matrix(vector)
                for i in range(0, len(self.large_poles)):
                    for j in range(0, len(self.chunks)):
                        self.chunks[j] = self.chunks[j].add_matrix(res_list[k].chunks[j].mul_scalar(vector.get(i, 0) * omit_all(self.large_poles, [self.large_poles[i]], delta)))

        for j in range(0, len(self.chunks)):
            s_sub = s_matrix[0:derivative_order - j + 1, 0:derivative_order - j + 1]
            self.chunks[j] = s_sub.mul_matrix(self.chunks[j])

class ConformalBlockTableSeed:
    """
    A class which calculates tables of conformal block derivatives from scratch
    using the recursion relations with meromorphic versions of the blocks.
    Usually, it will not be necessary for the user to call it. Instead,
    `ConformalBlockTable` calls it automatically for `m_max = 3` and `n_max = 0`.
    For people wanting to call it with different values of `m_max` and `n_max`,
    the parameters and attributes are the same as those of `ConformalBlockTable`.
    It also supports the `dump` method.
    """
    def __init__(self, dim, k_max, l_max, m_max, n_max, delta_12 = 0, delta_34 = 0, odd_spins = False, name = None):
        self.dim = dim
        self.k_max = k_max
        self.l_max = l_max
        self.m_max = m_max
        self.n_max = n_max
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

        if name != None:
            dump_file = open(name, 'r')
            command = dump_file.read()
            exec(command)
            return

        derivative_order = m_max + 2 * n_max
        nu = (dim / Integer(2)) - 1

        # The matrix for how derivatives are affected when one multiplies by r
        r_powers = []
        identity = [0] * ((derivative_order + 1) ** 2)
        lower_band = [0] * ((derivative_order + 1) ** 2)

        for i in range(0, derivative_order + 1):
            identity[i * (derivative_order + 1) + i] = 1
        for i in range(1, derivative_order + 1):
            lower_band[i * (derivative_order + 1) + i - 1] = i

        identity = DenseMatrix(derivative_order + 1, derivative_order + 1, identity)
        lower_band = DenseMatrix(derivative_order + 1, derivative_order + 1, lower_band)
        r_matrix = identity.mul_scalar(r_cross).add_matrix(lower_band)
        r_powers.append(identity)
        r_powers.append(r_matrix)

        conformal_blocks = []
        leading_blocks = []
        pol_list = []
        res_list = []
        pow_list = []
        new_res_list = []

        # Find out which residues we will ever need to include
        for l in range(0, l_max + k_max + 1):
            lb = LeadingBlockVector(dim, l, m_max, n_max, delta_12, delta_34)
            leading_blocks.append(lb)
            current_pol_list = []

            for k in range(1, k_max + 1):
                if l <= l_max:
                    if delta_residue(nu, k, l, delta_12, delta_34, 1) != 0:
                        current_pol_list.append((k, k, l + k, 1))

                if k % 2 == 0:
                    if delta_residue(nu, k // 2, l, delta_12, delta_34, 2) != 0:
                        current_pol_list.append((k, k // 2, l, 2))

                if k <= l:
                    if delta_residue(nu, k, l, delta_12, delta_34, 3) != 0:
                        current_pol_list.append((k, k, l - k, 3))

                if l == 0:
                    r_powers.append(r_powers[k].mul_matrix(r_powers[1]))

            # These are in the format (n, k, l, series)
            pol_list.append(current_pol_list)
            res_list.append([])
            pow_list.append([])
            new_res_list.append([])

        # Initialize the residues at the appropriate leading blocks
        for l in range(0, l_max + k_max + 1):
            for i in range(0, len(pol_list[l])):
                l_new = pol_list[l][i][2]
                res_list[l].append(MeromorphicBlockVector(leading_blocks[l_new]))

                pow_list[l].append(0)
                new_res_list[l].append(0)

        for k in range(1, k_max + 1):
            for l in range(0, l_max + k_max + 1):
                for i in range(0, len(res_list[l])):
                    if pow_list[l][i] >= k_max:
                        continue

                    res = delta_residue(nu, pol_list[l][i][1], l, delta_12, delta_34, pol_list[l][i][3])
                    pow_list[l][i] += pol_list[l][i][0]

                    for j in range(0, len(res_list[l][i].chunks)):
                        r_sub = r_powers[pol_list[l][i][0]][0:derivative_order - j + 1, 0:derivative_order - j + 1]
                        res_list[l][i].chunks[j] = r_sub.mul_matrix(res_list[l][i].chunks[j]).mul_scalar(res)

            for l in range(0, l_max + k_max + 1):
                for i in range(0, len(res_list[l])):
                    if pow_list[l][i] >= k_max:
                        continue

                    l_new = pol_list[l][i][2]
                    new_res_list[l][i] = MeromorphicBlockVector(leading_blocks[l_new])
                    pole1 = delta_pole(nu, pol_list[l][i][1], l, pol_list[l][i][3]) + pol_list[l][i][0]

                    for i_new in range(0, len(res_list[l_new])):
                        pole2 = delta_pole(nu, pol_list[l_new][i_new][1], l_new, pol_list[l_new][i_new][3])

                        for j in range(0, len(new_res_list[l][i].chunks)):
                            new_res_list[l][i].chunks[j] = new_res_list[l][i].chunks[j].add_matrix(res_list[l_new][i_new].chunks[j].mul_scalar(1 / eval_mpfr(pole1 - pole2, prec)))

            for l in range(0, l_max + k_max + 1):
                for i in range(0, len(res_list[l])):
                    if pow_list[l][i] >= k_max:
                        continue

                    for j in range(0, len(res_list[l][i].chunks)):
                         res_list[l][i].chunks[j] = new_res_list[l][i].chunks[j]

        # Perhaps poorly named, S keeps track of a linear combination of derivatives
        # We get this by including the essential singularity, then stripping it off again
        s_matrix = DenseMatrix(derivative_order + 1, derivative_order + 1, [0] * ((derivative_order + 1) ** 2))
        for i in range(0, derivative_order + 1):
            new_element = 1
            for j in range(i, -1, -1):
                s_matrix.set(i, j, new_element)
                new_element *= (j / ((i - j + 1) * r_cross)) * (delta - (i - j))

        for l in range(0, l_max + 1, step):
            conformal_block = ConformalBlockVector(dim, l, delta_12, delta_34, m_max + 2 * n_max, k_max, s_matrix, leading_blocks[l], pol_list[l], res_list[l])
            conformal_blocks.append(conformal_block)
            self.table.append(PolynomialVector([], [l, 0], conformal_block.large_poles))

        a = Symbol('a')
        b = Symbol('b')
        hack = Symbol('hack')
        old_coeff_grid = []

        rules1 = []
        rules2 = []
        old_expression1 = sqrt(a ** 2 - b) / (hack + sqrt((hack - a) ** 2 - b) + hack * sqrt(hack - a + sqrt((hack - a) ** 2 - b)))
        old_expression2 = (hack - sqrt((hack - a) ** 2 - b)) / sqrt(a ** 2 - b)

        for n in range(0, m_max + 2 * n_max + 1):
            old_coeff_grid.append([0] * (m_max + 2 * n_max + 1))

        for n in range(0, n_max + 1):
            for m in range(0, 2 * (n_max - n) + m_max + 1):
                if n == 0 and m == 0:
                    expression1 = old_expression1
                    expression2 = old_expression2
                elif m == 0:
                    old_expression1 = old_expression1.diff(b)
                    old_expression2 = old_expression2.diff(b)
                    expression1 = old_expression1
                    expression2 = old_expression2
                else:
                    expression1 = expression1.diff(a)
                    expression2 = expression2.diff(a)

                rules1.append(expression1.subs({hack : eval_mpfr(2, prec), a : 1, b : 0}))
                rules2.append(expression2.subs({hack : eval_mpfr(2, prec), a : 1, b : 0}))
                self.m_order.append(m)
                self.n_order.append(n)

        # If b is always 0, then eta is always 1
        if n_max == 0:
            _x = Symbol('_x')
            r = function_symbol('r', a)
            g = function_symbol('g', r)

            for m in range(0, derivative_order + 1):
                if m == 0:
                    old_expression = g
                    g = function_symbol('g', _x)
                else:
                    old_expression = old_expression.diff(a)

                expression = old_expression
                for i in range(1, m + 1):
                    expression = expression.subs(Derivative(r, [a] * i), rules1[i])

                for l in range(0, len(conformal_blocks)):
                    new_deriv = expression
                    for i in range(1, m + 1):
                        new_deriv = new_deriv.subs(Subs(Derivative(g, [_x] * i), [_x], [r]), conformal_blocks[l].chunks[0].get(i, 0))
                    if m == 0:
                        new_deriv = conformal_blocks[l].chunks[0].get(0, 0)
                    self.table[l].vector.append(new_deriv.expand())

            # Prevent further execution
            n_max = -1

        r = function_symbol('r', a, b)
        eta = function_symbol('eta', a, b)
        old_coeff_grid[0][0] = 1
        order = 0

        for n in range(0, n_max + 1):
            for m in range(0, 2 * (n_max - n) + m_max + 1):
                # Hack implementation of the g(r(a, b), eta(a, b)) chain rule
                if n == 0 and m == 0:
                    coeff_grid = self.deepcopy(old_coeff_grid)
                elif m == 0:
                    for i in range(m + n - 1, -1, -1):
                        for j in range(m + n - i - 1, -1, -1):
                            coeff = old_coeff_grid[i][j]
                            if type(coeff) == type(1):
                                coeff_deriv = 0
                            else:
                                coeff_deriv = coeff.diff(b)
                            old_coeff_grid[i + 1][j] += coeff * r.diff(b)
                            old_coeff_grid[i][j + 1] += coeff * eta.diff(b)
                            old_coeff_grid[i][j] = coeff_deriv
                    coeff_grid = self.deepcopy(old_coeff_grid)
                else:
                    for i in range(m + n - 1, -1, -1):
                        for j in range(m + n - i - 1, -1, -1):
                            coeff = coeff_grid[i][j]
                            if type(coeff) == type(1):
                                coeff_deriv = 0
                            else:
                                coeff_deriv = coeff.diff(a)
                            coeff_grid[i + 1][j] += coeff * r.diff(a)
                            coeff_grid[i][j + 1] += coeff * eta.diff(a)
                            coeff_grid[i][j] = coeff_deriv

                # Replace r and eta derivatives with the rules found above
                deriv = self.deepcopy(coeff_grid)
                for l in range(order, 0, -1):
                    for i in range(0, m + n + 1):
                        for j in range(0, m + n - i + 1):
                            if type(deriv[i][j]) != type(1):
                                deriv[i][j] = deriv[i][j].subs(Derivative(r, [a] * self.m_order[l] + [b] * self.n_order[l]), rules1[l])
                                deriv[i][j] = deriv[i][j].subs(Derivative(r, [b] * self.n_order[l] + [a] * self.m_order[l]), rules1[l])
                                deriv[i][j] = deriv[i][j].subs(Derivative(eta, [a] * self.m_order[l] + [b] * self.n_order[l]), rules2[l])
                                deriv[i][j] = deriv[i][j].subs(Derivative(eta, [b] * self.n_order[l] + [a] * self.m_order[l]), rules2[l])

                # Replace conformal block derivatives similarly for each spin
                for l in range(0, len(conformal_blocks)):
                    new_deriv = 0
                    for i in range(0, m + n + 1):
                        for j in range(0, m + n - i + 1):
                            new_deriv += deriv[i][j] * conformal_blocks[l].chunks[j].get(i, 0)
                    self.table[l].vector.append(new_deriv.expand())
                order += 1

    def dump(self, name):
        dump_table_contents(self, name)

    def deepcopy(self, array):
        ret = []
        for el in array:
            ret.append(list(el))
        return ret
