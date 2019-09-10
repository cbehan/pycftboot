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
                coeff = RealMPFR(next(tab_file)[:-1], prec)
                poly += coeff * (delta ** k)
            derivatives.append(poly.expand())

        single_poles = [0] * int(next(tab_file))
        for p in range(0, len(single_poles)):
            single_poles[p] = RealMPFR(next(tab_file)[:-1], prec)

        # We add coeff / (delta - p) summed over all poles
        # This just puts it over a common denominator automatically
        for i in range(0, len(derivatives)):
            prod1 = 1
            single_pole_term = 0
            for p in single_poles:
                coeff = RealMPFR(next(tab_file)[:-1], prec)
                single_pole_term = single_pole_term * (delta - p) + coeff * prod1
                single_pole_term = single_pole_term.expand()
                prod1 *= (delta - p)
                prod1 = prod1.expand()
            derivatives[i] = derivatives[i] * prod1 + single_pole_term
            derivatives[i] = derivatives[i].expand()

        double_poles = [0] * int(next(tab_file))
        for p in range(0, len(double_poles)):
            double_poles[p] = RealMPFR(next(tab_file)[:-1], prec)

        # Doing this for double poles is the same if we remember to square everything
        # We also need the product of single poles to come in at the end
        for i in range(0, len(derivatives)):
            prod2 = 1
            double_pole_term = 0
            for p in double_poles:
                coeff = RealMPFR(next(tab_file)[:-1], prec)
                double_pole_term = double_pole_term * ((delta - p) ** 2) + coeff * prod2
                double_pole_term = double_pole_term.expand()
                prod2 *= (delta - p) ** 2
                prod2 = prod2.expand()
            derivatives[i] = derivatives[i] * prod2 + double_pole_term * prod1
            derivatives[i] = derivatives[i] / (RealMPFR("2", prec) ** (block_table.m_order[i] + 2 * block_table.n_order[i]))
            derivatives[i] = derivatives[i].expand()

        poles = single_poles + (double_poles * 2)
        block_table.table.append(PolynomialVector(derivatives, [l, 0], poles))
        block_table.k_max = len(poles)
    tab_file.close()

def juliboots_write(block_table, name):
    """
    This writes out a block table in the format expected by JuliBoots. It is
    triggered when a `ConformalBlockTable` is dumped with the right format string.
    """
    tab_file = open(name, 'w')
    tab_file.write(str((block_table.dim / Integer(2)) - 1) + "\n")
    tab_file.write(str(block_table.n_max) + "\n")
    tab_file.write(str(block_table.m_max) + "\n")
    tab_file.write(str(block_table.l_max) + "\n")

    alternate = 1
    if block_table.odd_spins:
        tab_file.write("1\n")
    else:
        tab_file.write("0\n")
    tab_file.write(str(prec) + "\n")
    tab_file.write(str(len(block_table.table[0].vector)) + "\n")

    # Print delta_12 or delta_34 when we get the chance
    # If the file is going to have unused bits, we might as well use them
    for l in range(0, len(block_table.table)):
        if alternate == 1:
            tab_file.write(str(block_table.delta_12) + "\n")
        else:
            tab_file.write(str(block_table.delta_34) + "\n")

        max_degree = 0
        for poly in block_table.table[l].vector:
            coeff_list = sorted(poly.args, key = extract_power)
            degree = extract_power(coeff_list[-1])
            max_degree = max(max_degree, degree - len(block_table.table[l].poles))
        tab_file.write(str(max_degree + 1) + "\n")

        series = 1
        for p in block_table.table[l].poles:
            term = build_polynomial([1] * (max_degree + 1))
            term = term.subs(delta, p * delta)
            series *= term
            series = series.expand()
            series = build_polynomial(coefficients(series)[:max_degree + 1])

        # Above, delta functions as 1 / delta
        # We need to multiply by the numerator with reversed coefficients to get the entire part
        for i in range(0, len(block_table.table[l].vector)):
            poly = block_table.table[l].vector[i]
            coeff_list = coefficients(poly)
            coeff_list.reverse()
            poly = build_polynomial(coeff_list)
            poly = poly * series
            poly = poly.expand()
            coeff_list = coefficients(poly)
            # We get the numerator degree by subtracting the degree of series
            # The difference between this and the number of poles is the degree of the polynomial we write
            degree = len(coeff_list) - max_degree - len(block_table.table[l].poles) - 1
            factor = RealMPFR("2", prec) ** (block_table.m_order[i] + 2 * block_table.n_order[i])
            for k in range(0, max_degree + 1):
                index = degree - k
                if index >= 0:
                    tab_file.write(str(factor * coeff_list[index]) + "\n")
                else:
                    tab_file.write("0\n")

        single_poles = []
        double_poles = []
        for p in block_table.table[l].poles:
            if p in single_poles:
                single_poles.remove(p)
                double_poles.append(p)
            else:
                single_poles.append(p)

        # The single pole part of the partial fraction decomposition is easier
        tab_file.write(str(len(single_poles)) + "\n")
        for p in single_poles:
            tab_file.write(str(p) + "\n")

        for i in range(0, len(block_table.table[l].vector)):
            poly = block_table.table[l].vector[i]
            factor = RealMPFR("2", prec) ** (block_table.m_order[i] + 2 * block_table.n_order[i])
            for p in single_poles:
                num = poly.subs(delta, p)
                denom = omit_all(block_table.table[l].poles, [p], p)
                tab_file.write(str(factor * num / denom) + "\n")

        # The double pole part is identical
        tab_file.write(str(len(double_poles)) + "\n")
        for p in double_poles:
            tab_file.write(str(p) + "\n")

        for i in range(0, len(block_table.table[l].vector)):
            poly = block_table.table[l].vector[i]
            factor = RealMPFR("2", prec) ** (block_table.m_order[i] + 2 * block_table.n_order[i])
            for p in double_poles:
                num = poly.subs(delta, p)
                denom = omit_all(block_table.table[l].poles, [p], p)
                tab_file.write(str(factor * num / denom) + "\n")

        alternate *= -1
    tab_file.close()
