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
        return term.args[1].args[1]

def coefficients(polynomial):
    """
    Returns a sorted list of all coefficients in a polynomial starting with the
    constant term. Zeros are automatically added so that the length of the list
    is always one more than the degree.
    """
    if not "args" in dir(polynomial):
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
