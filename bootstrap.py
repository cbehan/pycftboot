#!/usr/bin/env python2
import xml.dom.minidom
import numpy.polynomial
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

delta  = symbols('delta')
delta_ext = symbols('delta_ext')

def is_nonzero(num):
    if type(num) == type(1.0):
        return abs(num) < 1e-10
    elif type(num) == type(eval_mpfr(1, 10)):
        return num > 1e-10 or num < -1e-10
    else:
        return num != 0

def delta_pole(nu, k, l, series):
    if series == 1:
        return 1 - l - k
    elif series == 2:
	return 1 + nu - k
    else:
	return 1 + l + 2 * nu - k

def delta_residue(nu, k, l, delta_12, delta_34, series):
    # Time saving special case
    if series != 2 and k % 2 != 0 and delta_12 == 0 and delta_34 == 0:
        return 0
    
    if series == 1:
        ret = - ((k * (-4) ** k) / (factorial(k) ** 2)) * sympy.rf((1 - k + delta_12) / 2.0, k) * sympy.rf((1 - k + delta_34) / 2.0, k)
	if l == 0 and nu == 0:
	    # Take l to 0, then nu
	    return ret * 2
	else:
	    return ret * (sympy.rf(l + 2 * nu, k) / sympy.rf(l + nu, k))
    elif series == 2:
        # There should be expressions for all integer dimension differences that
	# make it clear when their Pochhammer symbols cancel divergences in ret.
        if delta_12 == 0 and delta_34 == 0:
	    return - sympy.rf(nu, k) * sympy.rf(1 - nu, k) * (sympy.rf((nu + l + 1 - k) / 2, k) ** 2 / sympy.rf((nu + l - k) / 2, k) ** 2) * (k / factorial(k) ** 2) * ((nu + l - k) / (nu + l + k))
	else:
            ret = - ((k * (-16) ** k) / (factorial(k) ** 2)) * (sympy.rf(nu - k, 2 * k) / (sympy.rf(l + nu - k, 2 * k) * sympy.rf(l + nu + 1 - k, 2 * k)))
	    return ret * sympy.rf((1 - k + l + nu + delta_12) / 2.0, k) * sympy.rf((1 - k + l + nu + delta_34) / 2.0, k) * sympy.rf((1 - k + l + nu - delta_12) / 2.0, k) * sympy.rf((1 - k + l + nu - delta_34) / 2.0, k)
    else:
	return - ((k * (-4) ** k) / (factorial(k) ** 2)) * (sympy.rf(1 + l - k, k) * sympy.rf((1 - k + delta_12) / 2.0, k) * sympy.rf((1 - k + delta_34) / 2.0, k) / sympy.rf(1 + nu + l - k, k))

def get_poles(dim, l, delta_12, delta_34, kept_pole_order):
    nu = sympy.Rational(dim, 2) - 1

    k = 1
    ret = []
    while k <= kept_pole_order:
        if is_nonzero(delta_residue(nu, k, l, delta_12, delta_34, 1)):
	    ret.append(delta_pole(nu, k, l, 1))
	    
	# Nonzero but it might be infinite
	if k % 2 == 0:
	    if is_nonzero(delta_residue(nu, k / 2, l, delta_12, delta_34, 2)):
	        ret.append(delta_pole(nu, k / 2, l, 2))
	    
	if k <= l:
	    if is_nonzero(delta_residue(nu, k, l, delta_12, delta_34, 3)):
	        ret.append(delta_pole(nu, k, l, 3))

	k += 1

    return ret

def omit_all(poles, special_pole):
    expression = 1
    for p in poles:
        if p != special_pole:
	    expression *= (delta - p)
    return expression

def leading_block(nu, r, eta, l, delta_12, delta_34):
    if nu == 0:
        ret = sympy.chebyshevt(l, eta)
    else:
        ret = factorial(l) * sympy.gegenbauer(l, nu, eta) / sympy.rf(2 * nu, l)
    
    # Time saving special case
    if delta_12 == delta_34:
        return ret / (((1 - r ** 2) ** nu) * sqrt((1 + r ** 2) ** 2 - 4 * (r * eta) ** 2))
    else:
        return ((-1) ** l) * ret / (((1 - r ** 2) ** nu) * ((1 + r ** 2 + 2 * r * eta) ** ((1.0 + delta_12 - delta_34) / 2.0)) * ((1 + r ** 2 - 2 * r * eta) ** ((1.0 - delta_12 + delta_34) / 2.0)))

def dump_table_contents(block_table, name):
    dump_file = open(name, 'w')

    dump_file.write("self.dim = " + block_table.dim.__str__() + "\n")
    dump_file.write("self.l_max = " + block_table.l_max.__str__() + "\n")
    dump_file.write("self.m_max = " + block_table.m_max.__str__() + "\n")
    dump_file.write("self.n_max = " + block_table.n_max.__str__() + "\n")
    dump_file.write("self.kept_pole_order = " + block_table.kept_pole_order.__str__() + "\n")
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
	dump_file.write("self.table.append(PolynomialVector(derivatives, [" + block_table.table[l].label[0].__str__() + ", 0]))\n")

    dump_file.close()

class LeadingBlockVector:
    def __init__(self, dim, l, delta_12, delta_34, derivative_order):
	self.spin = l
	self.derivative_order = derivative_order
	self.chunks = []
	
	r = symbols('r')
	eta = symbols('eta')
	nu = sympy.Rational(dim, 2) - 1
	
	# We cache derivatives as we go
	# This is because csympy can only compute them one at a time, but it's faster anyway
	old_expression = leading_block(nu, r, eta, l, delta_12, delta_34)
	    
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
    def __init__(self, dim, Delta, l, delta_12, delta_34, derivative_order, kept_pole_order, top, old_pair, old_series):
        global r_powers
	global dual_poles
        self.chunks = []
	summation = []
	nu = sympy.Rational(dim, 2) - 1
        k = 1
	
        # Reuse leading block vectors that have already been calculated
	if len(leading_blocks) == 0:
	    lb = LeadingBlockVector(dim, l, delta_12, delta_34, derivative_order)
	    leading_blocks.append(lb)
	else:
            for lb in leading_blocks:
                if lb.spin == l and lb.derivative_order == derivative_order:
	            break
            if lb.spin != l:
                lb = LeadingBlockVector(dim, l, delta_12, delta_34, derivative_order)
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
            poles = get_poles(dim, l, delta_12, delta_34, kept_pole_order)
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
	        if k >= nu + l and is_nonzero(delta_residue(nu, k, l, delta_12, delta_34, 2)):
	            dual_poles.append(delta_pole(nu, k, l, 2))
		    dual_poles.append(delta_pole(nu, k, l, 2))
	        k += 1
	    k = 1
	
        while k <= kept_pole_order:
	    if len(r_powers) < k + 1:
	        r_powers.append(r_powers[k - 1].mul_matrix(r_powers[1]))
		
            res = delta_residue(nu, k, l, delta_12, delta_34, 1)
	    if is_nonzero(res):
	        pole = delta_pole(nu, k, l, 1)
		new_block = MeromorphicBlockVector(dim, pole + k, l + k, delta_12, delta_34, derivative_order, kept_pole_order - k, False, (Delta, pole), 1)
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
		    r_sub = r_powers[k].submatrix(0, derivative_order - i, 0, derivative_order - i)
		    summation[i] = summation[i].add_matrix(r_sub.mul_matrix(new_block.chunks[i]).mul_scalar(res))
	    
	    # We don't REALLY skip these parts for (k / 2) >= nu + l
	    # It's just that whenever this happens, the same pole has shown up in one of the other two sections
	    # The fact that it did will be signalled by a divergence that the program runs into
	    # It will handle this divergence in a way equivalent to keeping this term and taking the limit
	    if (k % 2 == 0) and ((k / 2) < nu + l or dim % 2 != 0):
	        res = delta_residue(nu, k / 2, l, delta_12, delta_34, 2)
	        if is_nonzero(res):
	            pole = delta_pole(nu, k / 2, l, 2)
		    new_block = MeromorphicBlockVector(dim, pole + k, l, delta_12, delta_34, derivative_order, kept_pole_order - k, False, (Delta, pole), 2)
		    if old_pair[0] == delta and old_pair[1] in dual_poles and Delta != pole:
	                res *= (old_pair[0] - old_pair[1])
		    
		    if top == True:
		        res *= omit_all(poles, pole)
		    else:
		        res /= Delta - pole
		    
		    for i in range(0, derivative_order + 1):
		        r_sub = r_powers[k].submatrix(0, derivative_order - i, 0, derivative_order - i)
		        summation[i] = summation[i].add_matrix(r_sub.mul_matrix(new_block.chunks[i]).mul_scalar(res))
	    
	    if k <= l:
	        res = delta_residue(nu, k, l, delta_12, delta_34, 3)
	        if is_nonzero(res):
		    pole = delta_pole(nu, k, l, 3)
		    new_block = MeromorphicBlockVector(dim, pole + k, l - k, delta_12, delta_34, derivative_order, kept_pole_order - k, False, (Delta, pole), 3)
		    if old_pair[0] == delta and old_pair[1] in dual_poles and Delta != pole:
	                res *= (old_pair[0] - old_pair[1])
		    
		    if top == True:
		        res *= omit_all(poles, pole)
		    elif Delta != pole:
		        res /= Delta - pole
		    else:
		        current_series = 3
			sign = sympy.Rational(old_series - 2, old_series - current_series)
			if old_pair[0] == delta:
		            res *= sign
		        else:
		            res /= (old_pair[0] - old_pair[1]) / sign
		    
		    for i in range(0, derivative_order + 1):
		        r_sub = r_powers[k].submatrix(0, derivative_order - i, 0, derivative_order - i)
		        summation[i] = summation[i].add_matrix(r_sub.mul_matrix(new_block.chunks[i]).mul_scalar(res))
	    
	    k += 1
	
	# A chunk is a set of r derivatives for one eta derivative
	# The matrix that should multiply a chunk is just R restricted to the right length
	for i in range(0, derivative_order + 1):
	    self.chunks.append(summation[i])

class ConformalBlockVector:
    def __init__(self, dim, l, delta_12, delta_34, derivative_order, kept_pole_order):
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
	
	meromorphic_block = MeromorphicBlockVector(dim, delta, l, delta_12, delta_34, derivative_order, kept_pole_order, True, (0, 0), 0)
	for i in range(0, derivative_order + 1):
	    s_sub = s_matrix.submatrix(0, derivative_order - i, 0, derivative_order - i)
	    self.chunks.append(s_sub.mul_matrix(meromorphic_block.chunks[i]))

class PolynomialVector:
    def __init__(self, derivatives, spin_irrep):
        if type(spin_irrep) == type(1):
	    spin_irrep = [spin_irrep, 0]
        self.vector = derivatives
	self.label = spin_irrep

class ConformalBlockTableSeed:
    def __init__(self, dim, l_max, m_max, n_max, kept_pole_order, delta_12 = 0, delta_34 = 0, odd_spins = False, name = None):
	self.dim = dim
	self.l_max = l_max
	self.m_max = m_max
	self.n_max = n_max
	self.kept_pole_order = kept_pole_order
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
	    conformal_blocks.append(ConformalBlockVector(dim, l, delta_12, delta_34, m_max + 2 * n_max, kept_pole_order))
	    self.table.append(PolynomialVector([], [l, 0]))
	
	a = symbols('a')
	b = symbols('b')
	hack = symbols('hack')
	r = function_symbol('r', a, b)
	eta = function_symbol('eta', a, b)
	old_coeff_grid = []
	
	rules1 = []
	rules2 = []
	old_expression1 = sqrt(a ** 2 - b) / (hack + sqrt((hack - a) ** 2 - b) + hack * sqrt(hack - a + sqrt((hack - a) ** 2 - b)))
	old_expression2 = (hack - sqrt((hack - a) ** 2 - b)) / sqrt(a ** 2 - b)
	
	print "Differentiating radial co-ordinates"
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
	
	print "Putting them together"
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
		                deriv[i][j] = deriv[i][j].subs(Derivative(eta, [a] * self.m_order[l] + [b] * self.n_order[l]), rules2[l])
		
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

class ConformalBlockTable:
    def __init__(self, dim, l_max, m_max, n_max, kept_pole_order, delta_12 = 0, delta_34 = 0, odd_spins = False, name = None):
	self.dim = dim
	self.l_max = l_max
	self.m_max = m_max
	self.n_max = n_max
	self.kept_pole_order = kept_pole_order
	self.odd_spins = odd_spins
	
	if name != None:
	    dump_file = open(name, 'r')
	    command = dump_file.read()
	    exec command
	    return
	
	small_table = ConformalBlockTableSeed(dim, l_max, min(m_max + 2 * n_max, 3), 0, kept_pole_order, delta_12, delta_34, odd_spins)
	self.m_order = small_table.m_order
	self.n_order = small_table.n_order
	self.table = small_table.table
	
	a = symbols('a')
	l = symbols('l')
	nu = eval_mpfr(sympy.Rational(dim, 2) - 1, prec)
	c_2 = (l * (l + 2 * nu) + delta * (delta - 2 * nu - 2)) / 2
	c_4 = l * (l + 2 * nu) * (delta - 1) * (delta - 2 * nu - 1)
	polys = [0, 0, 0, 0, 0]
	poly_derivs = [[], [], [], [], []]
	delta_prod = delta_12 * delta_34 / (-2.0)
	delta_sum = (delta_12 - delta_34) / (-2.0)
	
	# Polynomial 0 goes with the lowest order derivative on the right hand side
	# Polynomial 3 goes with the highest order derivative on the right hand side
	# Polynomial 4 goes with the derivative for which we are solving
	polys[0] += (a ** 0) * (16 * c_2 * (2 * nu + 1) - 8 * c_4)
	polys[0] += (a ** 1) * (4 * (c_4 + 2 * (2 * nu + 1) * (c_2 * delta_sum - c_2 + nu * delta_prod)))
	polys[0] += (a ** 2) * (2 * (delta_sum - nu) * (c_2 * (2 * delta_sum - 1) + delta_prod * (6 * nu - 1)))
	polys[0] += (a ** 3) * (2 * delta_prod * (delta_sum - nu) * (delta_sum - nu + 1))
	polys[1] += (a ** 1) * (-16 * c_2 * (2 * nu + 1))
	polys[1] += (a ** 2) * (4 * delta_prod - 24 * nu * delta_prod + 8 * nu * (2 * nu - 1) * (2 * delta_sum + 1) + 4 * c_2 * (1 - 4 * delta_sum + 6 * nu))
	polys[1] += (a ** 3) * (2 * c_2 * (4 * delta_sum - 2 * nu + 1) + 4 * (2 * nu - 1) * (2 * delta_sum + 1) * (delta_sum - nu + 1) + 2 * delta_prod * (10 * nu - 5 - 4 * delta_sum))
	polys[1] += (a ** 4) * ((delta_sum - nu + 1) * (4 * delta_prod + (2 * delta_sum + 1) * (delta_sum - nu + 2)))
	polys[2] += (a ** 2) * (16 * c_2 + 16 * nu - 32 * nu * nu)
	polys[2] += (a ** 3) * (8 * delta_prod - 8 * (3 * delta_sum - nu + 3) * (2 * nu - 1) - 16 * c_2 - 8 * nu + 16 * nu * nu)
	polys[2] += (a ** 4) * (4 * (c_2 - delta_prod + (3 * delta_sum - nu + 3) * (2 * nu - 1)) - 4 * delta_prod - 2 * (delta_sum - nu + 2) * (5 * delta_sum - nu + 5))
	polys[2] += (a ** 5) * (2 * delta_prod + (delta_sum - nu + 2) * (5 * delta_sum - nu + 5))
	polys[3] += (a ** 3) * (32 * nu - 16)
	polys[3] += (a ** 4) * (16 - 32 * nu + 4 * (4 * delta_sum - 2 * nu + 7))
	polys[3] += (a ** 5) * (4 * (2 * nu - 1) - 4 * (4 * delta_sum - 2 * nu + 7))
	polys[3] += (a ** 6) * (4 * delta_sum - 2 * nu + 7)
	polys[4] += (a ** 7) - 6 * (a ** 6) + 12 * (a ** 5) - 8 * (a ** 4)
	
	# Store all possible derivatives of these polynomials
	for i in range(0, 5):
	    for j in range(0, i + 4):
	        poly_derivs[i].append(polys[i].subs(a, 1))
		polys[i] = polys[i].diff(a)

	for m in range(self.m_order[-1] + 1, m_max + 2 * n_max + 1):
	    for j in range(0, len(small_table.table)):
		new_deriv = 0
		for i in range(m - 1, max(m - 8, -1), -1):
		    coeff = 0
		    index = max(m - i - 4, 0)
		    
		    prefactor = eval_mpfr(1, prec)
		    for k in range(0, index):
		        prefactor *= (m - 4 - k)
			prefactor /= k + 1
		    
		    k = max(4 + i - m, 0)
		    while k <= 4 and index <= (m - 4):
		        coeff += prefactor * poly_derivs[k][index]
			prefactor *= (m - 4 - index)
			prefactor /= index + 1
			index += 1
			k += 1
		    
		    if type(coeff) != type(1):
		        coeff = coeff.subs(l, small_table.table[j].label[0])
		    new_deriv -= coeff * self.table[j].vector[i]
		
		new_deriv = new_deriv / poly_derivs[4][0]
		self.table[j].vector.append(new_deriv.expand())
	    
	    self.m_order.append(m)
	    self.n_order.append(0)

	# This is just an alternative to storing derivatives as a doubly-indexed list
	index = m_max + 2 * n_max + 1
	index_map = [range(0, m_max + 2 * n_max + 1)]
	
	for n in range(1, n_max + 1):
	    index_map.append([])
	    for m in range(0, 2 * (n_max - n) + m_max + 1):
	        index_map[n].append(index)
		
	        coeff1 = m * (-1) * (2 - 4 * n - 4 * nu)
		coeff2 = m * (m - 1) * (2 - 4 * n - 4 * nu)
		coeff3 = m * (m - 1) * (m - 2) * (2 - 4 * n - 4 * nu)
		coeff4 = 1
		coeff5 = (-6 + m + 4 * n - 2 * nu - 2 * delta_sum)
		coeff6 = (-1) * (4 * c_2 + m * m + 8 * m * n - 5 * m + 4 * n * n - 2 * n - 2 - 4 * nu * (1 - m - n) + delta_sum * (m + 2 * n - 2) + 2 * delta_prod)
		coeff7 = m * (-1) * (m * m + 12 * m * n - 13 * m + 12 * n * n - 34 * n + 22 - 2 * nu * (2 * n - m - 1) + 2 * delta_sum * (m + 4 * n - 5) + 4 * delta_prod)
		coeff8 = (1 - n)
		coeff9 = (1 - n) * (-6 + 3 * m + 4 * n - 2 * nu + 2 * delta_sum)
		
	        for j in range(0, len(small_table.table)):
		    new_deriv = 0
		    
		    if m > 0:
		        new_deriv += coeff1 * self.table[j].vector[index_map[n][m - 1]]
		    if m > 1:
		        new_deriv += coeff2 * self.table[j].vector[index_map[n][m - 2]]
		    if m > 2:
		        new_deriv += coeff3 * self.table[j].vector[index_map[n][m - 3]]
		    
		    new_deriv += coeff4 * self.table[j].vector[index_map[n - 1][m + 2]]
		    new_deriv += coeff5 * self.table[j].vector[index_map[n - 1][m + 1]]
		    new_deriv += coeff6.subs(l, small_table.table[j].label[0]) * self.table[j].vector[index_map[n - 1][m]]
		    new_deriv += coeff7 * self.table[j].vector[index_map[n - 1][m - 1]]
		    
		    if n > 1:
		        new_deriv += coeff8 * self.table[j].vector[index_map[n - 2][m + 2]]
			new_deriv += coeff9 * self.table[j].vector[index_map[n - 2][m + 1]]
		    
		    new_deriv = new_deriv / (2 - 4 * n - 4 * nu)
		    self.table[j].vector.append(new_deriv.expand())
		
	        self.m_order.append(m)
	        self.n_order.append(n)
		index += 1
    
    def dump(self, name):
        dump_table_contents(self, name)

class ConvolvedBlockTable:
    def __init__(self, block_table, odd_spins = True, symmetric = False):
        # Copying everything but the unconvolved table is fine from a memory standpoint
        self.dim = block_table.dim
	self.l_max = block_table.l_max
	self.m_max = block_table.m_max
	self.n_max = block_table.n_max
	self.kept_pole_order = block_table.kept_pole_order
	
	self.m_order = []
	self.n_order = []
	self.table = []
	self.unit = []
	
	# We can restrict to even spin when the provided table has odd spin but not vice-versa
	if odd_spins == False and block_table.odd_spins == True:
	    self.odd_spins = False
	    step = 2
	else:
	    self.odd_spins = block_table.odd_spins
	    step = 1
	
	symbol_array = []
	for n in range(0, block_table.n_max + 1):
	    symbol_list = []
	    for m in range(0, 2 * (block_table.n_max - n) + block_table.m_max + 1):
	        symbol_list.append(symbols('g_' + n.__str__() + '_' + m.__str__()))
	    symbol_array.append(symbol_list)
	
	derivatives = []
	for n in range(0, block_table.n_max + 1):
	    for m in range(0, 2 * (block_table.n_max - n) + block_table.m_max + 1):
	        # Skip the ones that will vanish
		if (symmetric == False and m % 2 == 0) or (symmetric == True and m % 2 == 1):
		    continue
		
		self.m_order.append(m)
		self.n_order.append(n)
		
		expression = 0
		old_coeff = eval_mpfr(sympy.Rational(1, 4), prec) ** delta_ext
		for j in range(0, n + 1):
		    coeff = old_coeff
		    for i in range(0, m + 1):
		        expression += coeff * symbol_array[n - j][m - i]
		        coeff *= (i + 2 * j - 2 * delta_ext) * (m - i) / (i + 1)
		    old_coeff *= (j - delta_ext) * (n - j) / (j + 1)
	        
		deriv = expression / (factorial(m) * factorial(n))
		derivatives.append(deriv)
		
		for i in range(len(block_table.table[0].vector) - 1, 0, -1):
		    deriv = deriv.subs(symbol_array[block_table.n_order[i]][block_table.m_order[i]], 0)
		self.unit.append(2 * deriv.subs(symbol_array[0][0], 1))
	
	for l in range(0, len(block_table.table), step):
	    new_derivs = []
	    for i in range(0, len(derivatives)):
	        deriv = derivatives[i]
	        for j in range(len(block_table.table[0].vector) - 1, 0, -1):
		    deriv = deriv.subs(symbol_array[block_table.n_order[j]][block_table.m_order[j]], block_table.table[l].vector[j])
		new_derivs.append(2 * deriv.subs(symbol_array[0][0], block_table.table[l].vector[0]))
	    self.table.append(PolynomialVector(new_derivs, block_table.table[l].label))

class SDP:
    def __init__(self, dim_ext, conv_block_table_anti, conv_block_table_symm = None, vector_types = [[[[1, -1]], 0, 0]]):
        # Same story here
        self.dim = conv_block_table_anti.dim
	self.l_max = conv_block_table_anti.l_max
	self.m_max = conv_block_table_anti.m_max
	self.n_max = conv_block_table_anti.n_max
	self.kept_pole_order = conv_block_table_anti.kept_pole_order
	self.odd_spins = False
	
	# Just in case these are different
	if conv_block_table_symm != None:
	    self.l_max = max(self.l_max, conv_block_table_symm.l_max)
	    self.m_max = max(self.m_max, conv_block_table_symm.m_max)
	    self.n_max = max(self.n_max, conv_block_table_symm.n_max)
	    self.kept_pole_order = max(self.kept_pole_order, conv_block_table_symm.kept_pole_order)
	
	self.points = []
	self.m_order = []
	self.n_order = []
	self.table = []
	self.unit = []
	
	# We must assume the 0th element put in vector_types corresponds to the singlet channel
	# This is because we must harvest the identity from it
	for pair in vector_types[0][0]:
	    if pair[1] == -1:
	        tab = conv_block_table_anti
	    else:
	        tab = conv_block_table_symm
	    
	    for i in range(0, len(tab.unit)):
	        unit = pair[0] * tab.unit[i].subs(delta_ext, dim_ext)
		self.m_order.append(tab.m_order[i])
		self.n_order.append(tab.n_order[i])
		self.unit.append(unit)
	
	# Looping over types and spins gives "0 - S", "0 - T", "1 - A" and so on
	for vec in vector_types:
	    if (vec[1] % 2) == 1:
	        self.odd_spins = True
	        start = 1
	    else:
	        start = 0
	    
	    derivatives = []
	    for l in range(start, self.l_max, 2):
	        derivatives = []
	        # This loop switches between adding F derivatives and adding H derivatives
	        for pair in vec[0]:
		    if pair[1] == -1:
		        tab = conv_block_table_anti
		    else:
		        tab = conv_block_table_symm
		    
		    if tab.odd_spins:
		        index = l
		    else:
		        index = l / 2
		    
		    for i in range(0, len(tab.table[index].vector)):
		        derivatives.append(pair[0] * tab.table[index].vector[i].subs(delta_ext, dim_ext))
	        self.table.append(PolynomialVector(derivatives, [l, vec[2]]))
	
	self.bounds = [0.0] * len(self.table)
	self.set_bound()
    
    def add_point(self, spin_irrep, dimension):
        if type(spin_irrep) == type(1):
	    spin_irrep = [spin_irrep, 0]
        self.points.append((spin_irrep, dimension))
    
    def get_bound(self, gapped_spin_irrep):
        for l in range(0, len(self.table)):
	        if self.table[l].label == gapped_spin_irrep:
		    return self.bounds[l]
    
    # Defaults to unitarity bounds if there are missing arguments
    def set_bound(self, gapped_spin_irrep = -1, delta_min = -1):
        if gapped_spin_irrep == -1:
	    for l in range(0, len(self.table)):
	        spin = self.table[l].label[0]
		if spin == 0:
		    self.bounds[l] = sympy.Rational(self.dim, 2) - 1
		else:
		    self.bounds[l] = self.dim + spin - 2
	else:
	    if type(gapped_spin_irrep) == type(1):
	        gapped_spin_irrep = [gapped_spin_irrep, 0]
	    
	    for l in range(0, len(self.table)):
	        if self.table[l].label == gapped_spin_irrep:
		    break
	    spin = gapped_spin_irrep[0]
	    
	    if delta_min == -1 and spin == 0:
	        self.bounds[l] = sympy.Rational(self.dim, 2) - 1
	    elif delta_min == -1:
	        self.bounds[l] = self.dim + spin - 2
	    else:
	        self.bounds[l] = delta_min
    
    # Translate between the mathematica definition and the bootstrap definition of SDP
    def reshuffle_with_normalization(self, vector, norm):
	norm_hack = []
	for el in norm:
	    norm_hack.append(float(el))
	
	#max_index = 0
	max_index = norm_hack.index(max(norm_hack, key = abs))
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
	    for l in range(0, len(self.table)):
	        if self.table[l].label == p[0]:
	            break
	    
	    for i in range(0, len(self.table[l].vector)):
	        new_vector.append(self.table[l].vector[i].subs(delta, p[1]))
	    extra_vectors.append(PolynomialVector(new_vector, p[0]))
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
		if coeff_list == []:
		    coeff_list = [0.0]
		
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
	    poles = get_poles(self.dim, self.table[j].label[0], 0, 0, self.kept_pole_order)
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
    
    def bisect(self, lower, upper, threshold, spin_irrep):
        if type(spin_irrep) == type(1):
	    spin_irrep = [spin_irrep, 0]
	
        test = (lower + upper) / 2.0
        if abs(upper - lower) < threshold:
	    return lower
	else:
	    print "Trying " + str(test)
	    
	    obj = [0.0] * len(self.table[0].vector)
	    old = self.get_bound(spin_irrep)
	    self.set_bound(spin_irrep, test)
	    self.write_xml(obj, self.unit)
	    self.set_bound(spin_irrep, old)
	    
	    os.spawnlp(os.P_WAIT, "/usr/bin/sdpb", "sdpb", "-s", "mySDP.xml", "--findPrimalFeasible", "--findDualFeasible", "--noFinalCheckpoint")
	    out_file = open("mySDP.out", 'r')
	    terminate_line = out_file.next()
	    terminate_reason = terminate_line.partition(" = ")[-1]
	    out_file.close()
	    
	    if terminate_reason == '"found dual feasible solution";\n':
	        return self.bisect(lower, test, threshold, spin_irrep)
	    else:
	        return self.bisect(test, upper, threshold, spin_irrep)
    
    def opemax(self, dimension, spin_irrep):
        if type(spin_irrep) == type(1):
	    spin_irrep = [spin_irrep, 0]
	
	for l in range(0, len(self.table)):
	    if self.table[l].label == spin_irrep:
	        break
	
	norm = []
	for i in range(0, len(self.table[l].vector)):
	    norm.append(self.table[l].vector[i].subs(delta, dimension))
	
	self.write_xml(self.unit, norm)
	os.spawnlp(os.P_WAIT, "/usr/bin/sdpb", "sdpb", "-s", "mySDP.xml", "--noFinalCheckpoint")
	out_file = open("mySDP.out", 'r')
	out_file.next()
	primal_line = out_file.next()
	out_file.close()
	
	primal_value = primal_line.partition(" = ")[-1][:-2]
	return float(primal_value)
    
    def solution_functional(self, dimension, spin_irrep):
        if type(spin_irrep) == type(1):
	    spin_irrep = [spin_irrep, 0]
	
	obj = [0.0] * len(self.table[0].vector)
	old = self.get_bound(spin_irrep)
	self.set_bound(spin_irrep, dimension)
	self.write_xml(obj, self.unit)
	self.set_bound(spin_irrep, old)
	
	os.spawnlp(os.P_WAIT, "/usr/bin/sdpb", "sdpb", "-s", "mySDP.xml", "--noFinalCheckpoint")
	out_file = open("mySDP.out", 'r')
	for i in range(0, 7):
	    out_file.next()
	y_line = out_file.next()
	y_line = y_line.partition(" = ")[-1][1:-3]
	
	component_strings = y_line.split(", ")
	components = [eval_mpfr(1.0, prec)]
	for num in component_strings:
	    command = "components.append(eval_mpfr(" + num + ", prec))"
	    exec command
	
	return PolynomialVector(components, [0, 0])
    
    def extremal_coefficients(self, dimensions, spin_irreps):
        zeros = min(len(dimensions), len(spin_irreps))
	for i in range(0, zeros):
	    if type(spin_irreps[i]) == type(1):
	        spin_irreps[i] = [spin_irreps[i], 0]
	
	extremal_blocks = []
	for i in range(0, zeros):
	    for j in range(0, zeros):
	        for l in range(0, len(self.table)):
		    if self.table[l].label == spin_irreps[j]:
	                break
		
		polynomial_vector = self.table[l].vector[i].subs(delta, dimensions[j])
		extremal_blocks.append(float(polynomial_vector))
	
	identity = DenseMatrix(zeros, 1, self.unit)
	extremal_matrix = DenseMatrix(zeros, zeros, extremal_blocks)
	inverse = extremal_matrix.inv()
	
	return inverse.mul_matrix(identity)
    
    def extremal_dimensions(self, functional, spin_irrep):
        if type(spin_irrep) == type(1):
	    spin_irrep = [spin_irrep, 0]
	
	for l in range(0, len(self.table)):
	    if self.table[l].label == spin_irrep:
	        break
	
	inner_product = 0.0
	polynomial_vector = self.reshuffle_with_normalization(self.table[l].vector, self.unit)
	
	for i in range(0, len(self.table[l].vector)):
	    inner_product += functional.vector[i] * polynomial_vector[i]
	    inner_product = inner_product.expand()
	
	if type(inner_product) == type(eval_mpfr(1, 10)):
	    coeff_list = [inner_product]
	else:
	    coeff_list = sorted(inner_product.args, key = self.extract_power)
	if coeff_list == []:
	    coeff_list = [0.0]
	
	coeffs = []
	for d in range(0, len(coeff_list)):
	    if d == 0:
	        coeffs.append(float(coeff_list[0]))
	    else:
	        coeffs.append(float(coeff_list[d].args[0]))
	
	poly = numpy.polynomial.Polynomial(coeffs)
	roots = poly.roots()
	
	ret = []
	bound = self.get_bound(spin_irrep)
	for dim in roots:
	    if dim == dim.conj() and dim.real > (bound - 0.01):
	        ret.append(dim.real)
	return ret
