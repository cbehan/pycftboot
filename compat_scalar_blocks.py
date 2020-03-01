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

    # The convolution functions to support both can be found in the git history
    if info[:13] == "zzbDerivTable":
        print("Please rerun scalar_blocks with --output-ab")
        return
    elif info[:12] != "abDerivTable":
        print("Unknown convention for derivatives")
        return

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
    block_table.m_max = 1
    block_table.l_max = len(files) - 1
    block_table.odd_spins = False
    block_table.m_order = []
    block_table.n_order = []
    for n in range(0, block_table.n_max + 1):
        for m in range(0, 2 * (block_table.n_max - n) + 2):
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
        vector = re.sub("abDeriv\[[0-9]+,[0-9]+\]", "", vector).split(',\n')
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
            # All shifts, scalar or not, are undone here as we prefer to handle this step during XML writing
            derivatives.append(poly.subs(delta, delta - block_table.dim - l + 2).expand())

        # The block for scalar exchange should not give zero for the identity
        if l == 0:
            for i in range(0, len(derivatives)):
                poly = 0
                coeffs = coefficients(derivatives[i])
                for c in range(1, len(coeffs)):
                    poly += coeffs[c] * (delta ** (c - 1))
                derivatives[i] = poly

        poles = []
        nu = (block_table.dim - 2.0) / 2.0
        # Poles are not saved in the file so we have to reconstruct them
        for k in range(1, block_table.k_max + 1):
            # This program appears to use all the poles even when the scalars are identical
            if k > 1 or l > 0:
                poles.append(delta_pole(nu, k, l, 1))
            if k % 2 == 0:
                poles.append(delta_pole(nu, k // 2, l, 2))
            if k <= l:
                poles.append(delta_pole(nu, k, l, 3))
        block_table.table.append(PolynomialVector(derivatives, [l, 0], poles))
