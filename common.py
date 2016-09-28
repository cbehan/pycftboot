cutoff = 0
prec = 660
dec_prec = int((3.0 / 10.0) * prec)
mpmath.mp.dps = dec_prec
exec("tiny = eval_mpfr(1e-" + str(dec_prec // 2) + ", prec)")

rho_cross = 3 - 2 * mpmath.sqrt(2)
r_cross = eval_mpfr(3 - 2 * sqrt(2), prec)

ell = Symbol('ell')
delta  = Symbol('delta')
delta_ext = Symbol('delta_ext')

sdpb_path = "/usr/bin/sdpb"
sdpb_options = ["maxThreads", "checkpointInterval", "maxIterations", "maxRuntime", "dualityGapThreshold", "primalErrorThreshold", "dualErrorThreshold", "initialMatrixScalePrimal", "initialMatrixScaleDual", "feasibleCenteringParameter", "infeasibleCenteringParameter", "stepLengthReduction", "choleskyStabilizeThreshold", "maxComplementarity"]
sdpb_defaults = ["4", "3600", "500", "86400", "1e-30", "1e-30", "1e-30", "1e+20", "1e+20", "0.1", "0.3", "0.7", "1e-40", "1e+100"]

def deepcopy(array):
    """
    Copies a list of a list so that entries can be changed non-destructively.
    """
    ret = []
    for el in array:
        ret.append(list(el))
    return ret

def get_index(array, element):
    """
    Finds where an element occurs in an array or -1 if not present.
    """
    if element in array:
        return array.index(element)
    else:
        return -1

def get_index_approx(array, element):
    """
    Finds where an element numerically close to the one given occurs in an array
    or -1 if not present.
    """
    for i in range(0, len(array)):
        if abs(array[i] - element) < tiny:
            return i
    return -1

def extract_power(term):
    """
    Returns the degree of a single term in a polynomial. Symengine stores these
    as (coefficient, (delta, exponent)). This is helpful for sorting polynomials
    which are not sorted by default.
    """
    if not "args" in dir(term):
        return 0

    if term.args == ():
        return 0
    elif term.args[1].args == ():
        return 1
    else:
        return int(term.args[1].args[1])

def coefficients(polynomial):
    """
    Returns a sorted list of all coefficients in a polynomial starting with the
    constant term. Zeros are automatically added so that the length of the list
    is always one more than the degree.
    """
    if not "args" in dir(polynomial):
        return [polynomial]
    if polynomial.args == ():
        return [polynomial]

    coeff_list = sorted(polynomial.args, key = extract_power)
    degree = extract_power(coeff_list[-1])

    pos = 0
    ret = []
    for d in range(0, degree + 1):
        if extract_power(coeff_list[pos]) == d:
            if d == 0:
                ret.append(eval_mpfr(coeff_list[0], prec))
            else:
                ret.append(eval_mpfr(coeff_list[pos].args[0], prec))
            pos += 1
        else:
            ret.append(0)
    return ret

def unitarity_bound(dim, spin):
    """
    Returns the lower bound for conformal dimensions in a unitary theory for a
    given spatial dimension and spin.
    """
    if spin == 0:
        return (dim / Integer(2)) - 1
    else:
        return dim + spin - 2

def omit_all(poles, special_poles, var):
    """
    Instead of returning a product of poles where each pole is not in a special
    list, this returns a product where each pole is subtracted from some variable.
    """
    expression = 1
    for p in poles:
        if not p in special_poles:
            expression *= (var - p)
    return expression

def dump_table_contents(block_table, name):
    """
    This is called by `ConformalBlockTable` and `ConformalBlockTableSeed`. It
    writes executable Python code to a file designed to recreate the full set of
    the table's attributes as quickly as possible.
    """
    dump_file = open(name, 'w')

    dump_file.write("self.dim = " + block_table.dim.__str__() + "\n")
    dump_file.write("self.k_max = " + block_table.k_max.__str__() + "\n")
    dump_file.write("self.l_max = " + block_table.l_max.__str__() + "\n")
    dump_file.write("self.m_max = " + block_table.m_max.__str__() + "\n")
    dump_file.write("self.n_max = " + block_table.n_max.__str__() + "\n")
    dump_file.write("self.delta_12 = " + block_table.delta_12.__str__() + "\n")
    dump_file.write("self.delta_34 = " + block_table.delta_34.__str__() + "\n")
    dump_file.write("self.odd_spins = " + block_table.odd_spins.__str__() + "\n")
    dump_file.write("self.m_order = " + block_table.m_order.__str__() + "\n")
    dump_file.write("self.n_order = " + block_table.n_order.__str__() + "\n")
    dump_file.write("self.table = []\n")

    for l in range(0, len(block_table.table)):
        dump_file.write("derivatives = []\n")
        for i in range(0, len(block_table.table[0].vector)):
            poly_string = block_table.table[l].vector[i].__str__()
            poly_string = re.sub("([0-9]+\.[0-9]+e?-?[0-9]+)", r"eval_mpfr(\1, prec)", poly_string)
            dump_file.write("derivatives.append(" + poly_string + ")\n")
        dump_file.write("self.table.append(PolynomialVector(derivatives, " + block_table.table[l].label.__str__() + ", " + block_table.table[l].poles.__str__() + "))\n")

    dump_file.close()

def rules(m_max, n_max):
    """
    This takes the radial and angular co-ordinates, defined by Hogervorst and
    Rychkov in arXiv:1303.1111, and differentiates them with respect to the
    diagonal `a` and off-diagonal `b`. It returns a quadruple where the first
    two entries store radial and angular derivatives respectively evaluated at
    the crossing symmetric point. The third entry is a list stating the number of
    `a` derivatives to which a given position corresponds and the fourth entry
    does the same for `b` derivatives.
    """
    a = Symbol('a')
    b = Symbol('b')
    hack = Symbol('hack')

    rules1 = []
    rules2 = []
    m_order = []
    n_order = []
    old_expression1 = sqrt(a ** 2 - b) / (hack + sqrt((hack - a) ** 2 - b) + hack * sqrt(hack - a + sqrt((hack - a) ** 2 - b)))
    old_expression2 = (hack - sqrt((hack - a) ** 2 - b)) / sqrt(a ** 2 - b)

    if n_max == 0:
        old_expression1 = old_expression1.subs(b, 0)
        old_expression2 = b

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
            m_order.append(m)
            n_order.append(n)

    return (rules1, rules2, m_order, n_order)

def chain_rule_single_symengine(m_order, rules, table, conformal_blocks, accessor):
    """
    This reads a conformal block list where each spin's entry is a list of radial
    derivatives. It converts these to diagonal `a` derivatives using the rules
    given. Once these are calculated, the passed `table` is populated. Here,
    `accessor` is a hack to get around the fact that different parts of the code
    like to index in different ways.
    """
    _x = Symbol('_x')
    a = Symbol('a')
    r = function_symbol('r', a)
    g = function_symbol('g', r)
    m_max = max(m_order)

    for m in range(0, m_max + 1):
        if m == 0:
            old_expression = g
            g = function_symbol('g', _x)
        else:
            old_expression = old_expression.diff(a)

        expression = old_expression
        for i in range(1, m + 1):
            expression = expression.subs(Derivative(r, [a] * m_order[i]), rules[i])

        for l in range(0, len(conformal_blocks)):
            new_deriv = expression
            for i in range(1, m + 1):
                new_deriv = new_deriv.subs(Subs(Derivative(g, [_x] * i), [_x], [r]), accessor(l, i))
            if m == 0:
                new_deriv = accessor(l, 0)
            table[l].vector.append(new_deriv.expand())

def chain_rule_single(m_order, rules, table, conformal_blocks, accessor):
    """
    This implements the same thing except in Python which should not be faster
    but it is.
    """
    a = Symbol('a')
    r = function_symbol('r', a)
    m_max = max(m_order)

    old_coeff_grid = [0] * (m_max + 1)
    old_coeff_grid[0] = 1
    order = 0

    for m in range(0, m_max + 1):
        if m == 0:
            coeff_grid = old_coeff_grid[:]
        else:
            for i in range(m - 1, -1, -1):
                coeff = coeff_grid[i]
                if type(coeff) == type(1):
                    coeff_deriv = 0
                else:
                    coeff_deriv = coeff.diff(a)
                coeff_grid[i + 1] += coeff * r.diff(a)
                coeff_grid[i] = coeff_deriv

        deriv = coeff_grid[:]
        for l in range(order, 0, -1):
            for i in range(0, m + 1):
                if type(deriv[i]) != type(1):
                    deriv[i] = deriv[i].subs(Derivative(r, [a] * m_order[l]), rules[l])

        for l in range(0, len(conformal_blocks)):
            new_deriv = 0
            for i in range(0, m + 1):
                new_deriv += deriv[i] * accessor(l, i)
            table[l].vector.append(new_deriv.expand())
        order += 1

def chain_rule_double_symengine(m_order, n_order, rules1, rules2, table, conformal_blocks):
    """
    This reads a conformal block list where each spin has a chunk for a given
    number of angular derivatives and different radial derivatives within each
    chunk. It converts these to diagonal and off-diagonal `a` and `b` derivatives
    using the two sets of rules given. Once these are calculated, the passed
    `table` is populated.
    """
    _x = Symbol('_x')
    __x = Symbol('__x')
    a = Symbol('a')
    b = Symbol('b')
    r = function_symbol('r', a, b)
    eta = function_symbol('eta', a, b)
    g = function_symbol('g', r, eta)
    n_max = max(n_order)
    m_max = max(m_order) - 2 * n_max
    order = 0

    for n in range(0, n_max + 1):
        for m in range(0, 2 * (n_max - n) + m_max + 1):
            if n == 0 and m == 0:
                old_expression = g
                expression = old_expression
                g0 = function_symbol('g', __x, _x)
                g1 = function_symbol('g', _x, __x)
                g2 = function_symbol('g', _x, eta)
                g3 = function_symbol('g', r, _x)
                g4 = function_symbol('g', r, eta)
            elif m == 0:
                old_expression = old_expression.diff(b)
                expression = old_expression
            else:
                expression = expression.diff(a)

            deriv = expression
            for l in range(order, 0, -1):
                deriv = deriv.subs(Derivative(r, [a] * m_order[l] + [b] * n_order[l]), rules1[l])
                deriv = deriv.subs(Derivative(r, [b] * n_order[l] + [a] * m_order[l]), rules1[l])
                deriv = deriv.subs(Derivative(eta, [a] * m_order[l] + [b] * n_order[l]), rules2[l])
                deriv = deriv.subs(Derivative(eta, [b] * n_order[l] + [a] * m_order[l]), rules2[l])

            for l in range(0, len(conformal_blocks)):
                new_deriv = deriv
                for i in range(1, m + n + 1):
                    for j in range(1, m + n - i + 1):
                        new_deriv = new_deriv.subs(Subs(Derivative(g1, [_x] * i + [__x] * j), [_x, __x], [r, eta]), conformal_blocks[l].chunks[j].get(i, 0))
                        new_deriv = new_deriv.subs(Subs(Derivative(g0, [_x] * j + [__x] * i), [_x, __x], [eta, r]), conformal_blocks[l].chunks[j].get(i, 0))
                for i in range(1, m + n + 1):
                    new_deriv = new_deriv.subs(Subs(Derivative(g2, [_x] * i), [_x], [r]), conformal_blocks[l].chunks[0].get(i, 0))
                for j in range(1, m + n + 1):
                    new_deriv = new_deriv.subs(Subs(Derivative(g3, [_x] * j), [_x], [eta]), conformal_blocks[l].chunks[j].get(0, 0))
                new_deri = new_deriv.subs(g4, conformal_blocks[l].chunks[0].get(0, 0))
                table[l].vector.append(new_deriv.expand())
            order += 1

def chain_rule_double(m_order, n_order, rules1, rules2, table, conformal_blocks):
    """
    This implements the same thing except in Python which should not be faster
    but it is.
    """
    a = Symbol('a')
    b = Symbol('b')
    r = function_symbol('r', a, b)
    eta = function_symbol('eta', a, b)
    n_max = max(n_order)
    m_max = max(m_order) - 2 * n_max

    old_coeff_grid = []
    for n in range(0, m_max + 2 * n_max + 1):
        old_coeff_grid.append([0] * (m_max + 2 * n_max + 1))
    old_coeff_grid[0][0] = 1
    order = 0

    for n in range(0, n_max + 1):
        for m in range(0, 2 * (n_max - n) + m_max + 1):
            # Hack implementation of the g(r(a, b), eta(a, b)) chain rule
            if n == 0 and m == 0:
                coeff_grid = deepcopy(old_coeff_grid)
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
                coeff_grid = deepcopy(old_coeff_grid)
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
            deriv = deepcopy(coeff_grid)
            for l in range(order, 0, -1):
                for i in range(0, m + n + 1):
                    for j in range(0, m + n - i + 1):
                        if type(deriv[i][j]) != type(1):
                            deriv[i][j] = deriv[i][j].subs(Derivative(r, [a] * m_order[l] + [b] * n_order[l]), rules1[l])
                            deriv[i][j] = deriv[i][j].subs(Derivative(r, [b] * n_order[l] + [a] * m_order[l]), rules1[l])
                            deriv[i][j] = deriv[i][j].subs(Derivative(eta, [a] * m_order[l] + [b] * n_order[l]), rules2[l])
                            deriv[i][j] = deriv[i][j].subs(Derivative(eta, [b] * n_order[l] + [a] * m_order[l]), rules2[l])

            # Replace conformal block derivatives similarly for each spin
            for l in range(0, len(conformal_blocks)):
                new_deriv = 0
                for i in range(0, m + n + 1):
                    for j in range(0, m + n - i + 1):
                        new_deriv += deriv[i][j] * conformal_blocks[l].chunks[j].get(i, 0)
                table[l].vector.append(new_deriv.expand())
            order += 1
