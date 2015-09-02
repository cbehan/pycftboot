#!/usr/bin/env python2
import xml.dom.minidom
import mpmath
import re
import os

# Use regular sympy sparingly because it is slow
# Every time we explicitly use it, we should consider implementing such a line in C++
from symengine import *
from symengine.lib.symengine_wrapper import *
import sympy

prec = 660
mpmath.mp.dps = int((3.0 / 10.0) * prec)

s_matrix = []
r_powers = []
dual_poles = []
leading_blocks = []

rho_cross = 3 - 2 * mpmath.sqrt(2)
r_cross = eval_mpfr(3 - 2 * sqrt(2), prec)
z_cross = eval_mpfr(sympy.Rational(1, 2), prec)

delta  = symbols('delta')
delta_ext = symbols('delta_ext')

def delta_pole(nu, k, l, series):
    if series == 1:
        return 1 - l - 2 * k
    elif series == 2:
	return 1 + nu - k
    else:
	return 1 + l + 2 * nu - 2 * k

def delta_residue(nu, k, l, series):
    if series == 1:
        ret = - ((k * factorial(2 * k) ** 2) / (2 ** (4 * k - 1) * factorial(k) ** 4))
	if l == 0 and nu == 0:
	    # Take l to 0, then nu
	    return ret * 2
	else:
	    return ret * (sympy.rf(l + 2 * nu, 2 * k) / sympy.rf(l + nu, 2 * k))
    elif series == 2:
	return - sympy.rf(nu, k) * sympy.rf(1 - nu, k) * (sympy.rf((nu + l + 1 - k) / 2, k) ** 2 / sympy.rf((nu + l - k) / 2, k) ** 2) * (k / factorial(k) ** 2) * ((nu + l - k) / (nu + l + k))
    else:
	return - (sympy.rf(1 + l - 2 * k, 2 * k) / sympy.rf(1 + nu + l - 2 * k, 2 * k)) * ((k * factorial(2 * k) ** 2) / (2 ** (4 * k - 1) * factorial(k) ** 4))

def get_poles(dim, l, kept_pole_order):
    nu = sympy.Rational(dim, 2) - 1

    k = 1
    ret = []
    while (2 * k) <= kept_pole_order:
        if delta_residue(nu, k, l, 1) != 0:
	    ret.append(delta_pole(nu, k, l, 1))
	    
	# Nonzero but it might be infinite
	if delta_residue(nu, k, l, 2) != 0:
	    ret.append(delta_pole(nu, k, l, 2))
	    
	if k <= (l / 2):
	    if delta_residue(nu, k, l, 3) != 0:
	        ret.append(delta_pole(nu, k, l, 3))

	k += 1

    return ret

def omit_all(poles, special_pole):
    expression = 1
    for p in poles:
        if p != special_pole:
	    expression *= (delta - p)
    return expression

def leading_block(nu, R, Eta, l):
    if nu == 0:
        ret = sympy.chebyshevt(l, Eta)
    else:
        ret = factorial(l) * sympy.gegenbauer(l, nu, Eta) / sympy.rf(2 * nu, l)
    return ret / (((1 - R ** 2) ** nu) * sqrt((1 + R ** 2) ** 2 - 4 * (R * Eta) ** 2))

class LeadingBlockVector:
    def __init__(self, dim, derivative_order, l):
	self.spin = l
	self.derivative_order = derivative_order
	self.chunks = []
	
	r = symbols('r')
	eta = symbols('eta')
	nu = sympy.Rational(dim, 2) - 1
	
	# We cache derivatives as we go
	# This is because csympy can only compute them one at a time, but it's faster anyway
	old_expression = leading_block(nu, r, eta, l)
	    
	for m in range(0, derivative_order + 1):
	    chunk = []
	    for n in range(0, derivative_order - m + 1):
	        if n == 0 and m == 0:
		    expression = old_expression
		elif n == 0:
		    old_expression = old_expression.diff(eta)
		    expression = old_expression
		else:
		    expression = expression.diff(r)
		    
		chunk.append(expression.subs({r : r_cross, eta : 1}))
	    self.chunks.append(DenseMatrix(len(chunk), 1, chunk))

class MeromorphicBlockVector:
    def __init__(self, dim, Delta, l, derivative_order, kept_pole_order, top, old_pair, old_series):
        global r_powers
	global dual_poles
        self.chunks = []
	summation = []
	nu = sympy.Rational(dim, 2) - 1
        k = 1
	
        # Reuse leading block vectors that have already been calculated
	if len(leading_blocks) == 0:
	    lb = LeadingBlockVector(dim, derivative_order, l)
	    leading_blocks.append(lb)
	else:
            for lb in leading_blocks:
                if lb.spin == l and lb.derivative_order == derivative_order:
	            break
            if lb.spin != l:
                lb = LeadingBlockVector(dim, derivative_order, l)
	        leading_blocks.append(lb)
        for i in range(0, derivative_order + 1):
	    summation.append(lb.chunks[i])
	
	# Take the same strategy with powers of the R matrix
	if top == True:
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
	
	# Top says we have not recursed yet and our expression is still expected to have denominators
        # with the free variable delta. Cancelling them later is slow so we do it now.
        if top == True:
            poles = get_poles(dim, l, kept_pole_order)
	    for p in poles:
	        for i in range(0, derivative_order + 1):
	            summation[i] = summation[i].mul_scalar(delta - p)
	elif old_pair[0] == delta and old_pair[1] in dual_poles:
	    for i in range(0, derivative_order + 1):
	        summation[i] = summation[i].mul_scalar(old_pair[0] - old_pair[1])
	
	if dim % 2 != 0:
	    dual_poles = []
	
	# Preparing for infinite residues that may be encountered soon
        if top == True and dim % 2 == 0:
	    dual_poles = []
	    while (2 * k) <= kept_pole_order:
	        if k >= nu + l and delta_residue(nu, k, l, 2) != 0:
	            dual_poles.append(delta_pole(nu, k, l, 2))
		    dual_poles.append(delta_pole(nu, k, l, 2))
	        k += 1
	    k = 1
	
        while (2 * k) <= kept_pole_order:
	    if len(r_powers) < 2 * k + 1:
	        r_powers.append(r_powers[2 * k - 1].mul_matrix(r_powers[1]))
		r_powers.append(r_powers[2 * k].mul_matrix(r_powers[1]))
		
            res = delta_residue(nu, k, l, 1)
	    if res != 0:
	        pole = delta_pole(nu, k, l, 1)
		new_block = MeromorphicBlockVector(dim, pole + 2 * k, l + 2 * k, derivative_order, kept_pole_order - 2 * k, False, (Delta, pole), 1)
		if old_pair[0] == delta and old_pair[1] in dual_poles and Delta != pole:
	            res *= (old_pair[0] - old_pair[1])
		
		if top == True:
		    res *= omit_all(poles, pole)
		elif Delta != pole:
		    res /= Delta - pole
		else:
		    current_series = 1
		    sign = sympy.Rational(2 - old_series, old_series - current_series)
		    if old_pair[0] == delta:
		        res *= sign
		    else:
		        res /= (old_pair[0] - old_pair[1]) / sign
				
		for i in range(0, derivative_order + 1):
		    r_sub = r_powers[2 * k].submatrix(0, derivative_order - i, 0, derivative_order - i)
		    summation[i] = summation[i].add_matrix(r_sub.mul_matrix(new_block.chunks[i]).mul_scalar(res))
	    
	    # We don't REALLY skip these parts for k >= nu + l
	    # It's just that whenever this happens, the same pole has shown up in one of the other two sections
	    # The fact that it did will be signalled by a divergence that the program runs into
	    # It will handle this divergence in a way equivalent to keeping this term and taking the limit
	    if k < nu + l or dim % 2 != 0:
	        res = delta_residue(nu, k, l, 2)
	        if res != 0:
	            pole = delta_pole(nu, k, l, 2)
		    new_block = MeromorphicBlockVector(dim, pole + 2 * k, l, derivative_order, kept_pole_order - 2 * k, False, (Delta, pole), 2)
		    if old_pair[0] == delta and old_pair[1] in dual_poles and Delta != pole:
	                res *= (old_pair[0] - old_pair[1])
		    
		    if top == True:
		        res *= omit_all(poles, pole)
		    else:
		        res /= Delta - pole
		    
		    for i in range(0, derivative_order + 1):
		        r_sub = r_powers[2 * k].submatrix(0, derivative_order - i, 0, derivative_order - i)
		        summation[i] = summation[i].add_matrix(r_sub.mul_matrix(new_block.chunks[i]).mul_scalar(res))
	    
	    if k <= (l / 2):
	        res = delta_residue(nu, k, l, 3)
	        if res != 0:
		    pole = delta_pole(nu, k, l, 3)
		    new_block = MeromorphicBlockVector(dim, pole + 2 * k, l - 2 * k, derivative_order, kept_pole_order - 2 * k, False, (Delta, pole), 3)
		    if old_pair[0] == delta and old_pair[1] in dual_poles and Delta != pole:
	                res *= (old_pair[0] - old_pair[1])
		    
		    if top == True:
		        res *= omit_all(poles, pole)
		    elif Delta != pole:
		        res /= Delta - pole
		    else:
		        current_series = 3
			sign = sympy.Rational(2 - old_series, old_series - current_series)
			if old_pair[0] == delta:
		            res *= sign
		        else:
		            res /= (old_pair[0] - old_pair[1]) / sign
		    
		    for i in range(0, derivative_order + 1):
		        r_sub = r_powers[2 * k].submatrix(0, derivative_order - i, 0, derivative_order - i)
		        summation[i] = summation[i].add_matrix(r_sub.mul_matrix(new_block.chunks[i]).mul_scalar(res))
	    
	    k += 1
	
	# A chunk is a set of r derivatives for one eta derivative
	# The matrix that should multiply a chunk is just R restricted to the right length
	for i in range(0, derivative_order + 1):
	    self.chunks.append(summation[i])

class ConformalBlockVector:
    def __init__(self, dim, l, derivative_order, kept_pole_order):
        global s_matrix
	self.chunks = []
	
	# Perhaps poorly named, S keeps track of a linear combination of derivatives
	# We get this by including the essential singularity, then stripping it off again
	if s_matrix == []:
	    s_matrix = DenseMatrix(derivative_order + 1, derivative_order + 1, [0] * ((derivative_order + 1) ** 2))
	    for i in range(0, derivative_order + 1):
	        new_element = 1
	        for j in range(i, -1, -1):
		    s_matrix.set(i, j, new_element)
		    new_element *= (j / ((i - j + 1) * r_cross)) * (delta - (i - j))
	
	meromorphic_block = MeromorphicBlockVector(dim, delta, l, derivative_order, kept_pole_order, True, (0, 0), 0)
	for i in range(0, derivative_order + 1):
	    s_sub = s_matrix.submatrix(0, derivative_order - i, 0, derivative_order - i)
	    self.chunks.append(s_sub.mul_matrix(meromorphic_block.chunks[i]))

class SDPVector:
    def __init__(self, derivatives, l):
        self.vector = derivatives
	self.spin = l

class ConformalBlockTable:
    def __init__(self, dim, derivative_order, kept_pole_order, l_max, odd_spins = False, name = None):
	self.dim = dim
	self.derivative_order = derivative_order
	self.kept_pole_order = kept_pole_order
	self.l_max = l_max
	self.odd_spins = odd_spins
	self.m_order = []
	self.n_order = []
	self.table = []
	
	if odd_spins:
	    step = 1
	else:
	    step = 2
	conformal_blocks = []
	
	if name != None:
	    dump_file = open(name, 'r')
	    command = dump_file.read()
	    exec command
	    return
	
	print "Preparing blocks"
	for l in range(0, l_max + 1, step):
	    conformal_blocks.append(ConformalBlockVector(dim, l, derivative_order, kept_pole_order))
	    self.table.append([])
	
	z_norm = symbols('z_norm')
	z_conj = symbols('z_conj')
	r = function_symbol('r', z_norm, z_conj)
	eta = function_symbol('eta', z_norm, z_conj)
	old_coeff_grid = []
	
	rules1 = []
	rules2 = []
	old_expression1 = sqrt(z_norm * z_conj) / ((1 + sqrt(1 - z_norm)) * (1 + sqrt(1 - z_conj)))
	old_expression2 = (sqrt(z_norm / z_conj) * ((1 + sqrt(1 - z_conj)) / (1 + sqrt(1 - z_norm))) + sqrt(z_conj / z_norm) * ((1 + sqrt(1 - z_norm)) / (1 + sqrt(1 - z_conj)))) / 2
	
	print "Differentiating radial co-ordinates"
	for m in range(0, derivative_order + 1):
	    old_coeff_grid.append([0] * (derivative_order - m + 1))
	
	for m in range(0, derivative_order + 1):
	    for n in range(0, min(((derivative_order + 1) / 2), derivative_order - m) + 1):
		if n == 0 and m == 0:
		    expression1 = old_expression1
		    expression2 = old_expression2
		elif n == 0:
		    old_expression1 = old_expression1.diff(z_norm)
		    old_expression2 = old_expression2.diff(z_norm)
		    expression1 = old_expression1
		    expression2 = old_expression2
		else:
		    expression1 = expression1.diff(z_conj)
		    expression2 = expression2.diff(z_conj)
		
		rules1.append(expression1.subs({z_norm : z_cross, z_conj : z_cross}))
		rules2.append(expression2.subs({z_norm : z_cross, z_conj : z_cross}))
		self.m_order.append(m)
		self.n_order.append(n)
	
	print "Putting them together"
	old_coeff_grid[0][0] = 1
	order = 0
	
	for m in range(0, derivative_order + 1):
	    for n in range(0, min(((derivative_order + 1) / 2), derivative_order - m) + 1):
	        # Hack implementation of the g(r(z_norm, z_conj), eta(z_norm, z_conj)) chain rule
	        if n == 0 and m == 0:
		    coeff_grid = self.deepcopy(old_coeff_grid)
		elif n == 0:
		    for i in range(m + n - 1, -1, -1):
		        for j in range(m + n - i - 1, -1, -1):
			    coeff = old_coeff_grid[i][j]
			    if type(coeff) == type(1):
			        coeff_deriv = 0
			    else:
			        coeff_deriv = coeff.diff(z_norm)
			    old_coeff_grid[i + 1][j] += coeff * r.diff(z_norm)
			    old_coeff_grid[i][j + 1] += coeff * eta.diff(z_norm)
			    old_coeff_grid[i][j] = coeff_deriv
		    coeff_grid = self.deepcopy(old_coeff_grid)
		else:
		    for i in range(m + n - 1, -1, -1):
		        for j in range(m + n - i - 1, -1, -1):
			    coeff = coeff_grid[i][j]
			    if type(coeff) == type(1):
			        coeff_deriv = 0
			    else:
			        coeff_deriv = coeff.diff(z_conj)
			    coeff_grid[i + 1][j] += coeff * r.diff(z_conj)
			    coeff_grid[i][j + 1] += coeff * eta.diff(z_conj)
			    coeff_grid[i][j] = coeff_deriv
		
		# Replace r and eta derivatives with the rules found above
		deriv = self.deepcopy(coeff_grid)
	    	for l in range(order, 0, -1):
		    for i in range(0, m + n + 1):
		        for j in range(0, m + n - i + 1):
			    if type(deriv[i][j]) != type(1):
		                deriv[i][j] = deriv[i][j].subs(Derivative(r, [z_norm] * self.m_order[l] + [z_conj] * self.n_order[l]), rules1[l])
		                deriv[i][j] = deriv[i][j].subs(Derivative(eta, [z_norm] * self.m_order[l] + [z_conj] * self.n_order[l]), rules2[l])
		
		# Replace conformal block derivatives similarly for each spin
		for l in range(0, len(conformal_blocks)):
		    new_deriv = 0
		    for i in range(0, m + n + 1):
		        for j in range(0, m + n - i + 1):
			    new_deriv += deriv[i][j] * conformal_blocks[l].chunks[j].get(i, 0)
		    self.table[l].append(new_deriv.expand())
		order += 1
    
    def dump(self, name):
        dump_file = open(name, 'w')
	
	dump_file.write("self.dim = " + self.dim.__str__() + "\n")
	dump_file.write("self.derivative_order = " + self.derivative_order.__str__() + "\n")
	dump_file.write("self.kept_pole_order = " + self.kept_pole_order.__str__() + "\n")
	dump_file.write("self.l_max = " + self.l_max.__str__() + "\n")
	dump_file.write("self.odd_spins = " + self.odd_spins.__str__() + "\n")
	dump_file.write("self.m_order = " + self.m_order.__str__() + "\n")
	dump_file.write("self.n_order = " + self.n_order.__str__() + "\n")
	
        for l in range(0, len(self.table)):
	    dump_file.write("derivatives = []\n")
	    for i in range(0, len(self.table[0])):
	        poly_string = self.table[l][i].__str__()
		poly_string = re.sub("([0-9]+\.[0-9]+e?-?[0-9]+)", r"eval_mpfr(\1, prec)", poly_string)
	        dump_file.write("derivatives.append(" + poly_string + ")\n")
	    dump_file.write("self.table.append(derivatives)\n")
	
	dump_file.close()
    
    def deepcopy(self, array):
        ret = []
	for el in array:
	    ret.append(list(el))
	return ret

class ConvolvedBlockTable:
    def __init__(self, block_table, odd_spins = True):
        # Copying everything but the unconvolved table is fine from a memory standpoint
        self.dim = block_table.dim
	self.derivative_order = block_table.derivative_order
	self.kept_pole_order = block_table.kept_pole_order
	self.l_max = block_table.l_max
	self.table = []
	self.unit = []
	
	# We can restrict to even spin when the provided table has odd spin but not vice-versa
	if odd_spins == False and block_table.odd_spins == True:
	    self.odd_spins = False
	    step = 2
	else:
	    self.odd_spins = block_table.odd_spins
	    step = 1
	
	z_norm = symbols('z_norm')
        z_conj = symbols('z_conj')
	
	symbol_array = []
	for m in range(0, block_table.derivative_order + 1):
	    symbol_list = []
	    for n in range(0, min(((block_table.derivative_order + 1) / 2), block_table.derivative_order - m) + 1):
	        symbol_list.append(symbols('g_' + m.__str__() + '_' + n.__str__()))
	    symbol_array.append(symbol_list)
	
	derivatives = []
	for m in range(0, block_table.derivative_order + 1):
	    for n in range(0, min(m, block_table.derivative_order - m) + 1):
	        # Skip even derivatives
		if (m + n) % 2 == 0:
		    continue
		
		expression = 0
		old_coeff = (z_cross * z_cross) ** delta_ext
		for i in range(0, m + 1):
		    coeff = old_coeff
		    for j in range(0, n + 1):
		        expression += coeff * symbol_array[max(m - i, n - j)][min(m - i, n - j)]
		        coeff *= (j - delta_ext) * (n - j) / ((j + 1) * z_cross)
		    old_coeff *= (i - delta_ext) * (m - i) / ((i + 1) * z_cross)
	        
		deriv = expression / (factorial(m) * factorial(n))
		derivatives.append(deriv)
		
		for i in range(len(block_table.table[0]) - 1, 0, -1):
		    deriv = deriv.subs(symbol_array[block_table.m_order[i]][block_table.n_order[i]], 0)
		self.unit.append(2 * deriv.subs(symbol_array[0][0], 1))
	
	for l in range(0, len(block_table.table), step):
	    new_derivs = []
	    for i in range(0, len(derivatives)):
	        deriv = derivatives[i]
	        for j in range(len(block_table.table[0]) - 1, 0, -1):
		    deriv = deriv.subs(symbol_array[block_table.m_order[j]][block_table.n_order[j]], block_table.table[l][j])
		deriv = deriv.subs(symbol_array[0][0], block_table.table[l][0])
		new_derivs.append(2 * deriv.subs({z_norm : z_cross, z_conj : z_cross}))
	    self.table.append(new_derivs)

class SDP:
    def __init__(self, conv_block_table, dim_ext):
        # Same story here
        self.dim = conv_block_table.dim
	self.derivative_order = conv_block_table.derivative_order
	self.kept_pole_order = conv_block_table.kept_pole_order
	self.l_max = conv_block_table.l_max
	self.odd_spins = conv_block_table.odd_spins
	
	self.unit = []
	self.table = []
	self.points = []
	
	for i in range(0, len(conv_block_table.table[0])):
	    unit = conv_block_table.unit[i].subs(delta_ext, dim_ext)
	    self.unit.append(unit)

	for l in range(0, len(conv_block_table.table)):
	    if self.odd_spins:
	        spin = l
	    else:
	        spin = 2 * l
	    
	    derivatives = []
	    for i in range(0, len(conv_block_table.table[l])):
	        derivatives.append(conv_block_table.table[l][i].subs(delta_ext, dim_ext))
	    self.table.append(SDPVector(derivatives, spin))
	
	self.bounds = [0.0] * len(self.table)
	self.set_bound()
    
    def add_point(self, spin, dimension):
        self.points.append((spin, dimension))
    
    # Defaults to unitarity bounds if there are missing arguments
    def set_bound(self, gapped_spin = -1, delta_min = -1):
        if gapped_spin == -1:
	    self.bounds[0] = sympy.Rational(self.dim, 2) - 1
	    for l in range(1, len(self.bounds)):
	        if self.odd_spins:
		    spin = l
		else:
		    spin = 2 * l
	        self.bounds[l] = self.dim + spin - 2
	else:
	    if self.odd_spins:
	        l = gapped_spin
	    else:
	        l = gapped_spin / 2
	    
	    if delta_min == -1 and l == 0:
	        self.bounds[0] = sympy.Rational(self.dim, 2) - 1
	    elif delta_min == -1:
	        self.bounds[l] = self.dim + gapped_spin - 2
	    else:
	        self.bounds[l] = delta_min
    
    # Translate between the mathematica definition and the bootstrap definition of SDP
    def reshuffle_with_normalization(self, vector, norm):
        max_index = 0
	#max_index = norm.index(max(norm, key = abs))
	const = vector[max_index] / norm[max_index]
	ret = []
	
	for i in range(0, len(norm)):
	    ret.append(vector[i] - const * norm[i])
	
	ret = [const] + ret[:max_index] + ret[max_index + 1:]
	return ret
    
    def get_index(self, array, element):
        if element in array:
            return array.index(element)
        else:
            return -1
    
    # Polynomials in csympy are not sorted
    # This determines sorting order from the (coefficient, (delta, exponent)) representation
    def extract_power(self, term):
        if term.args == ():
            return 0
        elif term.args[1].args == ():
            return 1
        else:
            return term.args[1].args[1]
    
    def make_laguerre_points(self, degree):
        ret = []
        for d in range(0, degree + 1):
	    point = -(pi ** 2) * ((4 * d - 1) ** 2) / (64 * (degree + 1) * log(r_cross))
	    ret.append(eval_mpfr(point, prec))
	return ret

    def shifted_prefactor(self, poles, base, x, shift):
        product = 1
        for p in poles:
            product *= x - (p - shift)
        return (base ** (x + shift)) / product
    
    def integral(self, pos, shift, poles):
        single_poles = []
	double_poles = []
	ret = mpmath.mpf(0)
	
	for p in poles:
	    if (p - shift) in single_poles:
	        single_poles.remove(p - shift)
		double_poles.append(p - shift)
	    elif (p - shift) < 0:
	        single_poles.append(p - shift)
	
	for i in range(0, len(single_poles)):
	    denom = mpmath.mpf(1)
	    pole = single_poles[i]
	    other_single_poles = single_poles[:i] + single_poles[i + 1:]
	    for p in other_single_poles:
	        denom *= pole - p
	    for p in double_poles:
	        denom *= (pole - p) ** 2
	    ret += (mpmath.mpf(1) / denom) * (rho_cross ** pole) * ((-pole) ** pos) * mpmath.factorial(pos) * mpmath.gammainc(-pos, a = pole * mpmath.log(rho_cross))
	
	for i in range(0, len(double_poles)):
	    denom = mpmath.mpf(1)
	    pole = double_poles[i]
	    other_double_poles = double_poles[:i] + double_poles[i + 1:]
	    for p in other_double_poles:
	        denom *= (pole - p) ** 2
	    for p in single_poles:
	        denom *= pole - p
	    # Contribution of the most divergent part
	    ret += (mpmath.mpf(1) / (pole * denom)) * ((-1) ** (pos + 1)) * mpmath.factorial(pos) * ((mpmath.log(rho_cross)) ** (-pos))
	    ret -= (mpmath.mpf(1) / denom) * (rho_cross ** pole) * ((-pole) ** (pos - 1)) * mpmath.factorial(pos) * mpmath.gammainc(-pos, a = pole * mpmath.log(rho_cross)) * (pos + pole * mpmath.log(rho_cross))
	    
	    factor = 0
	    for p in other_double_poles:
	        factor -= mpmath.mpf(2) / (pole - p)
	    for p in single_poles:
	        factor -= mpmath.mpf(1) / (pole - p)
	    # Contribution of the least divergent part
	    ret += (factor / denom) * (rho_cross ** pole) * ((-pole) ** pos) * mpmath.factorial(pos) * mpmath.gammainc(-pos, a = pole * mpmath.log(rho_cross))
	
	return (rho_cross ** shift) * ret
    
    def write_xml(self, obj, norm):
        obj = self.reshuffle_with_normalization(obj, norm)
        laguerre_points = []
	laguerre_degrees = []
	extra_vectors = []
	
	# Handle discretely added points
	print "Adding isolated points"
	for p in self.points:
	    new_vector = []
	    if self.odd_spins:
	        l = p[0]
	    else:
	        l = p[0] / 2
	    
	    for i in range(0, len(self.table[0].vector)):
	        new_vector.append(self.table[l].vector[i].subs(delta, p[1]))
	    extra_vectors.append(SDPVector(new_vector, p[0]))
	self.table += extra_vectors
	
	doc = xml.dom.minidom.Document()
	root_node = doc.createElement("sdp")
	doc.appendChild(root_node)
	
	objective_node = doc.createElement("objective")
	matrices_node = doc.createElement("polynomialVectorMatrices")
	root_node.appendChild(objective_node)
	root_node.appendChild(matrices_node)
	
	# Here, we use indices that match the SDPB specification
	for n in range(0, len(obj)):
	    elt_node = doc.createElement("elt")
	    elt_node.appendChild(doc.createTextNode(obj[n].__str__()))
	    objective_node.appendChild(elt_node)
	
	for j in range(0, len(self.table)):
	    spin = self.table[j].spin
	    
	    matrix_node = doc.createElement("polynomialVectorMatrix")
	    rows_node = doc.createElement("rows")
	    cols_node = doc.createElement("cols")
	    elements_node = doc.createElement("elements")
	    sample_point_node = doc.createElement("samplePoints")
	    sample_scaling_node = doc.createElement("sampleScalings")
	    bilinear_basis_node = doc.createElement("bilinearBasis")
	    rows_node.appendChild(doc.createTextNode("1"))
	    cols_node.appendChild(doc.createTextNode("1"))
	    
	    degree = 0
	    if j >= len(self.bounds):
	        delta_min = 0
	    else:
	        delta_min = self.bounds[j]
	    polynomial_vector = self.reshuffle_with_normalization(self.table[j].vector, norm)
	    
	    vector_node = doc.createElement("polynomialVector")
	    for n in range(0, len(polynomial_vector)):
	        expression = polynomial_vector[n].expand()
		# Impose unitarity bounds and the specified gap
		expression = expression.subs(delta, delta + delta_min).expand()
		
		if type(expression) == type(eval_mpfr(1, 10)):
		    coeff_list = [expression]
		else:
		    coeff_list = sorted(expression.args, key = self.extract_power)
		degree = max(degree, len(coeff_list) - 1)
		
	        polynomial_node = doc.createElement("polynomial")
		for d in range(0, len(coeff_list)):
		    if d == 0:
		        coeff = eval_mpfr(coeff_list[0], prec)
		    else:
		        coeff = eval_mpfr(coeff_list[d].args[0], prec)
		    
		    coeff_node = doc.createElement("coeff")
		    coeff_node.appendChild(doc.createTextNode(coeff.__str__()))
		    polynomial_node.appendChild(coeff_node)
		vector_node.appendChild(polynomial_node)
	    elements_node.appendChild(vector_node)
	    
	    print "Getting points"
	    poles = get_poles(self.dim, spin, self.kept_pole_order)
	    index = self.get_index(laguerre_degrees, degree)
	    if j >= len(self.bounds):
	        points = [self.points[j - len(self.bounds)][1]]
	    elif index == -1:
	        points = self.make_laguerre_points(degree)
		laguerre_points.append(points)
		laguerre_degrees.append(degree)
	    else:
	        points = laguerre_points[index]
	    
	    print "Evaluating them"
	    for d in range(0, degree + 1):
	        elt_node = doc.createElement("elt")
		elt_node.appendChild(doc.createTextNode(points[d].__str__()))
		sample_point_node.appendChild(elt_node)
		damped_rational = self.shifted_prefactor(poles, r_cross, points[d], eval_mpfr(delta_min, prec))
		elt_node = doc.createElement("elt")
		elt_node.appendChild(doc.createTextNode(damped_rational.__str__()))
		sample_scaling_node.appendChild(elt_node)
	    
	    # We have now finished using delta_min in csympy
	    # It's time to convert it to a more precise mpmath type for this part
	    delta_min = mpmath.mpf(delta_min.__str__())
	    
	    bands = []
	    matrix = []
	    # One place where arbitrary precision really matters
	    print "Getting bands"
	    for d in range(0, 2 * (degree / 2) + 1):
	        result = self.integral(d, delta_min, poles)
		bands.append(result)
	    for r in range(0, (degree / 2) + 1):
	        new_entries = []
	        for s in range(0, (degree / 2) + 1):
		    new_entries.append(bands[r + s])
		matrix.append(new_entries)
	    matrix = mpmath.matrix(matrix)
	    print "Decomposing matrix of size " + str(degree) + " / 2"
	    matrix = mpmath.cholesky(matrix, tol = mpmath.mpf(1e-200))
	    print "Inverting matrix"
	    matrix = mpmath.inverse(matrix)
	    
	    for d in range(0, (degree / 2) + 1):
		polynomial_node = doc.createElement("polynomial")
		for q in range(0, d + 1):
		    coeff_node = doc.createElement("coeff")
		    coeff_node.appendChild(doc.createTextNode(matrix[d, q].__str__()))
		    polynomial_node.appendChild(coeff_node)
		bilinear_basis_node.appendChild(polynomial_node)
	    
	    matrix_node.appendChild(rows_node)
	    matrix_node.appendChild(cols_node)
	    matrix_node.appendChild(elements_node)
	    matrix_node.appendChild(sample_point_node)
	    matrix_node.appendChild(sample_scaling_node)
	    matrix_node.appendChild(bilinear_basis_node)
	    matrices_node.appendChild(matrix_node)
	
	self.table = self.table[:len(self.bounds)]
	xml_file = open("mySDP.xml", 'wb')
	doc.writexml(xml_file, addindent = "    ", newl = '\n')
	xml_file.close()
	doc.unlink()
    
    def bisect(self, lower, upper, threshold, spin):
        test = (lower + upper) / 2.0
        if abs(upper - lower) < threshold:
	    return lower
	else:
	    print "Trying " + str(test)
	    obj = [0.0] * len(self.table[0].vector)
	    self.set_bound(spin, test)
	    self.write_xml(obj, self.unit)
	    os.spawnlp(os.P_WAIT, "/usr/bin/sdpb", "sdpb", "-s", "mySDP.xml", "--findPrimalFeasible", "--findDualFeasible", "--noFinalCheckpoint")
	    out_file = open("mySDP.out", 'r')
	    terminate_line = out_file.next()
	    terminate_reason = terminate_line.partition(" = ")[-1]
	    out_file.close()
	    
	    if terminate_reason == '"found dual feasible solution";\n':
	        return self.bisect(lower, test, threshold, spin)
	    else:
	        return self.bisect(test, upper, threshold, spin)
    
    def opemax(self, dimension, spin):
        if self.odd_spins:
	    j = spin
	else:
	    j = spin / 2
	
	norm = []
	for i in range(0, len(self.table[j])):
	    norm.append(self.table[j].vector[i].subs(delta, dimension))
	
	# Impose no gap
	self.write_xml(self.unit, norm)
	os.spawnlp(os.P_WAIT, "/usr/bin/sdpb", "sdpb", "-s", "mySDP.xml", "--noFinalCheckpoint")
	out_file = open("mySDP.out", 'r')
	out_file.next()
	primal_line = out_file.next()
	out_file.close()
	
	primal_value = primal_line.partition(" = ")[-1][:-2]
	return float(primal_value)
