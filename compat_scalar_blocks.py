class ConvolvedBlockTable2:
    """
    This is exactly like `ConvolvedBlockTable` except it uses derivatives with
    respect to (z, zb) instead of (a, b).
    """
    def __init__(self, block_table, odd_spins = True, symmetric = False):
        self.dim = block_table.dim
        self.k_max = block_table.k_max
        self.l_max = block_table.l_max
        self.m_max = block_table.m_max
        self.n_max = block_table.n_max
        self.delta_12 = block_table.delta_12
        self.delta_34 = block_table.delta_34
        self.m_order = []
        self.n_order = []
        self.table = []

        if odd_spins == False and block_table.odd_spins == True:
            self.odd_spins = False
        else:
            self.odd_spins = block_table.odd_spins
        if block_table.odd_spins == True:
            step = 1
        else:
            step = 2
        if self.odd_spins == True:
            spin_list = range(0, self.l_max + 1, 1)
        else:
            spin_list = range(0, self.l_max + 1, 2)

        symbol_array = []
        for m in range(0, block_table.m_max + 1):
            symbol_list = []
            for n in range(0, min(m, block_table.m_max - m) + 1):
                symbol_list.append(Symbol('g_' + m.__str__() + '_' + n.__str__()))
            symbol_array.append(symbol_list)

        derivatives = []
        for m in range(0, block_table.m_max + 1):
            for n in range(0, min(m, block_table.m_max - m) + 1):
                if (symmetric == False and (m + n) % 2 == 0) or (symmetric == True and (m + n) % 2 == 1):
                    continue

                self.m_order.append(m)
                self.n_order.append(n)

                expression = 0
                old_coeff = eval_mpfr(Integer(1) / Integer(4), prec) ** delta_ext
                for i in range(0, m + 1):
                    coeff = old_coeff
                    for j in range(0, n + 1):
                        expression += coeff * symbol_array[max(m - i, n - j)][min(m - i, n - j)]
                        coeff *= 2 * (j - delta_ext) * (n - j) / (j + 1)
                    old_coeff *= 2 * (i - delta_ext) * (m - i) / (i + 1)
                deriv = expression / (factorial(m) * factorial(n))
                derivatives.append(deriv)

        for spin in spin_list:
            l = spin // step
            new_derivs = []
            for i in range(0, len(derivatives)):
                deriv = derivatives[i]
                for j in range(len(block_table.table[l].vector) - 1, 0, -1):
                    deriv = deriv.subs(symbol_array[max(block_table.m_order[j], block_table.n_order[j])][min(block_table.m_order[j], block_table.n_order[j])], block_table.table[l].vector[j])
                new_derivs.append(2 * deriv.subs(symbol_array[0][0], block_table.table[l].vector[0]))
            self.table.append(PolynomialVector(new_derivs, [spin, 0], block_table.table[l].poles))

def scalar_blocks_read(block_table, name):
    """
    This reads in a block table produced by scalar_blocks, the program by Walter
    Landry. Whether to call it is determined by `ConformalBlockTable`
    automatically.
    """
    files1 = os.listdir(name)
    files0 = sorted(files1)
    files = sorted(files0, key = len)
    # A cheap way to get alphanumeric sort
    info = files[0]

    # Parsing is annoying because '-' is used in the numbers and the delimiters
    delta12_negative = info.split("-delta12--")
    delta12_positive = info.split("-delta12-")
    if len(delta12_negative) > 1:
        block_table.delta_12 = float(delta12_negative[1].split('-')[0])
        info = info.replace("-delta12--", "-delta12-")
    else:
        block_table.delta_12 = float(delta12_positive[1].split('-')[0])
    delta34_negative = info.split("-delta34--")
    delta34_positive = info.split("-delta34-")
    if len(delta34_negative) > 1:
        block_table.delta_34 = float(delta34_negative[1].split('-')[0])
        info = info.replace("-delta34--", "-delta34-")
    else:
        block_table.delta_34 = float(delta34_positive[1].split('-')[0])

    info = info.split('-')
    block_table.dim = float(info[1][1:])
    block_table.k_max = int(info[8][13:])
    block_table.n_max = int(info[7][4:]) - 1
    block_table.m_max = 2 * block_table.n_max + 1
    block_table.l_max = len(files) - 1
    block_table.odd_spins = False
    block_table.m_order = []
    block_table.n_order = []
    for m in range(0, block_table.m_max + 1):
        for n in range(0, min(m, block_table.m_max - m) + 1):
            block_table.m_order.append(m)
            block_table.n_order.append(n)

    block_table.table = []
    for f in files:
        info = f.replace('--', '-')
        full = name + "/" + f
        l = int(info.split('-')[6][1:])
        if l % 2 == 1:
            block_table.odd_spins = True
        if l > block_table.l_max:
            block_table.l_max = l
        derivatives = []
        vector = open(full, 'r').read().replace('{', '').replace('}', '')
        vector = re.sub("zzbDeriv\[[0-9]+,[0-9]+\]", "", vector).split(',\n')
        for el in vector:
            poly = 0
            poly_lines = el.split('\n')
            # Watch out for the blank line at the end
            for k in range(0, len(poly_lines)):
                if len(poly_lines[k]) == 0:
                    continue
                if k == 0:
                    coeff = poly_lines[k].split('->')[1]
                else:
                    coeff = poly_lines[k].split('*')[0][5:]
                poly += RealMPFR(coeff, prec) * (delta ** k)
            # It turns out that the scalars come with a shift of d - 2 which is not the unitarity bound
            # All shifts, scalar or not, are undone here as we prefer to have this handle this step during XML writing
            derivatives.append(poly.subs(delta, delta - block_table.dim - l + 2).expand())

        poles = []
        nu = (block_table.dim - 2.0) / 2.0
        # Poles are not saved in the file so we have to reconstruct them
        for k in range(1, block_table.k_max + 1):
            # This program appears to use all the poles even when the scalars are identical
            poles.append(delta_pole(nu, k, l, 1))
            if k % 2 == 0:
                poles.append(delta_pole(nu, k // 2, l, 2))
            if k <= l:
                poles.append(delta_pole(nu, k, l, 3))
        block_table.table.append(PolynomialVector(derivatives, [l, 0], poles))
