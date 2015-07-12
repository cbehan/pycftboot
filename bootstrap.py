#!/usr/bin/env python2

# Use regular sympy sparingly because it is slow
# Every time we explicitly use it, we should consider implementing such a line in C++
from csympy import *
from csympy.lib.csympy_wrapper import *
import sympy

z_norm = symbols('z_norm')
z_conj = symbols('z_conj')
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
	    
	if k < nu + l or dim % 2 == 1:
	    if delta_residue(nu, k, l, 2) != 0:
	        ret.append(delta_pole(nu, k, l, 2))
	    
	if k <= (l / 2):
	    if delta_residue(nu, k, l, 3) != 0:
	        ret.append(delta_pole(nu, k, l, 3))

	k += 1

    # This probably won't change anything
    if nu == 0 and l == 0:
        ret.append(-1)

    return ret

def omit_product(poles, special_pole):
    expression = 1
    omitted = False
    for p in poles:
        if p == special_pole and omitted == False:
	    omitted = True
	else:
	    expression *= (delta - p)
    return expression

def leading_block(nu, r, eta, l):
    if nu == 0:
        ret = sympy.chebyshevt(l, eta)
    else:
        ret = factorial(l) * sympy.gegenbauer(l, nu, eta) / sympy.rf(2 * nu, l)
    return ret / (((1 - r ** 2) ** nu) * sqrt((1 + r ** 2) ** 2 - 4 * (r * eta) ** 2))

def meromorphic_block(dim, r, eta, Delta, l_new, l, kept_pole_order, top):
    k = 1
    nu = sympy.Rational(dim, 2) - 1
    # When the recursion relation shifts l, this does not affect the appropriate poles and
    # residues to use which are still determined by the original spin.
    summation = leading_block(nu, r, eta, l_new)
    
    # Top says we have not recursed yet and our expression is still expected to have denominators
    # with the free variable delta. Cancelling them later is slow so we do it now.
    if top == True:
        poles = get_poles(dim, l, kept_pole_order)
	for p in poles:
	    summation *= (delta - p)
    
    while (2 * k) <= kept_pole_order:
        res = delta_residue(nu, k, l, 1)
	if res != 0:
	    pole = delta_pole(nu, k, l, 1)
	    if top == True:
	        new_term = res * (r ** (2 * k)) * omit_product(poles, pole)
	    else:
	        new_term = res * (r ** (2 * k)) / (Delta - pole)
	    summation += new_term * meromorphic_block(dim, r, eta, pole + 2 * k, l + 2 * k, l, kept_pole_order - 2 * k, False)
	
	if k < nu + l or dim % 2 == 1:
	    res = delta_residue(nu, k, l, 2)
	    if res != 0:
	        pole = delta_pole(nu, k, l, 2)
		if top == True:
	            new_term = res * (r ** (2 * k)) * omit_product(poles, pole)
	        else:
	            new_term = res * (r ** (2 * k)) / (Delta - pole)
	        summation += new_term * meromorphic_block(dim, r, eta, pole + 2 * k, l, l, kept_pole_order - 2 * k, False)
	
	if k <= (l / 2):
	    res = delta_residue(nu, k, l, 3)
	    if res != 0:
	        pole = delta_pole(nu, k, l, 3)
	        if top == True:
	            new_term = res * (r ** (2 * k)) * omit_product(poles, pole)
	        else:
	            new_term = res * (r ** (2 * k)) / (Delta - pole)
	        summation += new_term * meromorphic_block(dim, r, eta, pole + 2 * k, l - 2 * k, l, kept_pole_order - 2 * k, False)

	k += 1
    return summation

def conformal_block(dim, rho_norm, rho_conj, Delta, l, kept_pole_order):
    r = sqrt(rho_norm * rho_conj)
    eta = (rho_norm + rho_conj) / (2 * r)
    return (r ** Delta) * meromorphic_block(dim, r, eta, Delta, l, l, kept_pole_order, True)

class ConformalBlockTable:
    def __init__(self, dim, derivative_order, kept_pole_order, l_max, odd_spins = False):
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
	
	z = symbols('z')
	rho_n = function_symbol('rho_n', z_norm)
	rho_c = function_symbol('rho_c', z_conj)
	rho = z / (1 + sqrt(1 - z)) ** 2
	
	rules = []
	for i in range(0, derivative_order + 1):
	    rules.append(rho.subs(z, 0.5))
	    rho = rho.diff(z)
	
	# We cache derivatives as we go
	# This is because csympy can only compute them one at a time, but it's faster anyway
	for l in range(0, l_max + 1, step):
	    derivatives = []
	    expressions = []
	    
	    # Multiplying a list twice is weird in Python
	    # If this were a numpy array, it would have to store floats
	    for i in range(0, derivative_order + 1):
	        expressions.append([0] * (derivative_order + 1))
	    
	    expressions[0][0] = conformal_block(dim, rho_n, rho_c, delta, l, kept_pole_order)
	    for m in range(0, derivative_order + 1):
		for n in range(0, min(m, derivative_order - m) + 1):
		    if n == 0 and m != 0:
		        expressions[m][0] = expressions[m - 1][0].diff(z_norm)
		    elif n > 0:
		        expressions[m][n] = expressions[m][n - 1].diff(z_conj)
		    
		    # We need to expand before we evaluate to have the essential singularity cancel
		    deriv = expressions[m][n] * (sqrt(rho_n * rho_c) ** (-delta))
		    deriv = deriv.expand()
		    
		    for i in range(m, 0, -1):
		        deriv = deriv.subs(Derivative(rho_n, [z_norm] * i), rules[i])
		    for j in range(n, 0, -1):
		        deriv = deriv.subs(Derivative(rho_c, [z_conj] * j), rules[j])
		    deriv = deriv.subs(rho_n, rules[0]).subs(rho_c, rules[0])
		    
		    derivatives.append(deriv)
		    # For the 27th element of the list, say what m derivative and what n derivative it corresponds to
		    if l == 0:
		        self.m_order.append(m)
		        self.n_order.append(n)
	    self.table.append(derivatives)

class ConvolvedBlockTable:
    def __init__(self, block_table):
        # Copying everything but the unconvolved table is fine from a memory standpoint
        self.dim = block_table.dim
	self.derivative_order = block_table.derivative_order
	self.kept_pole_order = block_table.kept_pole_order
	self.l_max = block_table.l_max
	self.odd_spins = block_table.odd_spins
	self.table = []
	
	g = function_symbol('g', z_norm, z_conj)
	f = (((1 - z_norm) * (1 - z_conj)) ** delta_ext) * g
	
	# Same comments apply here
	for l in range(0, len(block_table.table)):
	    derivatives = []
	    expressions = []
	    
	    for i in range(0, block_table.derivative_order + 1):
	        expressions.append([0] * (block_table.derivative_order + 1))
	    
	    expressions[0][0] = f
	    for m in range(0, block_table.derivative_order + 1):
		for n in range(0, min(m, block_table.derivative_order - m) + 1):
		    if n == 0 and m != 0:
		        expressions[m][0] = expressions[m - 1][0].diff(z_norm)
		    elif n > 0:
		        expressions[m][n] = expressions[m][n - 1].diff(z_conj)
		    
		    # Now we replace abstract derivatives with the expressions in the unconvolved table
		    deriv = expressions[m][n] / (factorial(m) * factorial(n))
		    for i in range(len(block_table.table[0]) - 1, 0, -1):
		        deriv = deriv.subs(Derivative(g, [z_norm] * block_table.m_order[i] + [z_conj] * block_table.n_order[i]), block_table.table[l][i])
		    
		    deriv = deriv.subs(g, block_table.table[l][0])
		    derivatives.append(deriv.subs({z_norm : 0.5, z_conj : 0.5}))
	    self.table.append(derivatives)
