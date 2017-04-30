def juliboots_read(block_table, name):
    """
    This reads in a block table produced by JuliBoots, the program by Miguel
    Paulos. Whether to call it is determined by `ConformalBlockTable`
    automatically. The two attributes of `ConformalBlockTable` that do not appear
    in the JuliBoots specification are delta_12 and delta_34. The user just has to
    remember them.
    """
    tab_file = open(name, 'r')
    nu = float(next(tab_file))

    block_table.n_max = int(next(tab_file))
    block_table.m_max = int(next(tab_file))
    block_table.l_max = int(next(tab_file))
    odds = int(next(tab_file))
    prec = int(next(tab_file))
    comp = int(next(tab_file))

    block_table.dim = 2 * nu + 2
    if odds == 0:
        step = 2
        block_table.odd_spins = False
    else:
        step = 1
        block_table.odd_spins = True

    block_table.m_order = []
    block_table.n_order = []
    for n in range(0, block_table.n_max + 1):
        for m in range(0, 2 * (block_table.n_max - n) + block_table.m_max + 1):
            block_table.m_order.append(m)
            block_table.n_order.append(n)

    block_table.table = []
    for l in range(0, block_table.l_max + 1, step):
        artifact = float(next(tab_file))
        degree = int(next(tab_file)) - 1

        derivatives = []
        for i in range(0, comp):
            poly = 0
            for k in range(0, degree + 1):
                exec("coeff = eval_mpfr(" + next(tab_file)[:-1] + ", prec)")
                poly += coeff * (delta ** k)
            derivatives.append(poly.expand())

        single_poles = [0] * int(next(tab_file))
        for p in range(0, len(single_poles)):
            exec("single_poles[p] = eval_mpfr(" + next(tab_file)[:-1] + ", prec)")

        # We add coeff / (delta - p) summed over all poles
        # This just puts it over a common denominator automatically
        for i in range(0, len(derivatives)):
            prod1 = 1
            single_pole_term = 0
            for p in single_poles:
                exec("coeff = eval_mpfr(" + next(tab_file)[:-1] + ", prec)")
                single_pole_term = single_pole_term * (delta - p) + coeff * prod1
                single_pole_term = single_pole_term.expand()
                prod1 *= (delta - p)
                prod1 = prod1.expand()
            derivatives[i] = derivatives[i] * prod1 + single_pole_term
            derivatives[i] = derivatives[i].expand()

        double_poles = [0] * int(next(tab_file))
        for p in range(0, len(double_poles)):
            exec("double_poles[p] = eval_mpfr(" + next(tab_file)[:-1] + ", prec)")

        # Doing this for double poles is the same if we remember to square everything
        # We also need the product of single poles to come in at the end
        for i in range(0, len(derivatives)):
            prod2 = 1
            double_pole_term = 0
            for p in double_poles:
                exec("coeff = eval_mpfr(" + next(tab_file)[:-1] + ", prec)")
                double_pole_term = double_pole_term * ((delta - p) ** 2) + coeff * prod2
                double_pole_term = double_pole_term.expand()
                prod2 *= (delta - p) ** 2
                prod2 = prod2.expand()
            derivatives[i] = derivatives[i] * prod2 + double_pole_term * prod1
            derivatives[i] = derivatives[i] / (eval_mpfr(2, prec) ** (block_table.m_order[i] + 2 * block_table.n_order[i]))
            derivatives[i] = derivatives[i].expand()

        poles = single_poles + (double_poles * 2)
        block_table.table.append(PolynomialVector(derivatives, [l, 0], poles))
        block_table.k_max = len(poles)
    tab_file.close()
