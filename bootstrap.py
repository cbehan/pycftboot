#!/usr/bin/env python2
from sympy import *
import time

z_norm = symbols('z_norm')
z_conj = symbols('z_conj')
delta  = symbols('delta')

def damped_rational(const, poles, base, delta):
    product = 1
    for p in poles:
        product *= delta - p
    return const * (base ** delta) / product

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
	    return ret * (rf(l + 2 * nu, 2 * k) / rf(l + nu, 2 * k))
    elif series == 2:
	return - rf(nu, k) * rf(1 - nu, k) * (rf((nu + l + 1 - k) / 2, k) ** 2 / rf((nu + l - k) / 2, k) ** 2) * (k / factorial(k) ** 2) * ((nu + l - k) / (nu + l + k))
    else:
	return - (rf(1 + l - 2 * k, 2 * k) / rf(1 + nu + l - 2 * k, 2 * k)) * ((k * factorial(2 * k) ** 2) / (2 ** (4 * k - 1) * factorial(k) ** 4))

def get_poles(dim, l, kept_pole_order):
    nu = Rational(dim, 2) - 1

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
	ret = chebyshevt(l, eta)
    else:
        ret = factorial(l) * gegenbauer(l, nu, eta) / rf(2 * nu, l)
    return ret / (((1 - r ** 2) ** nu) * sqrt((1 + r ** 2) ** 2 - 4 * (r * eta) ** 2))

def meromorphic_block(dim, r, eta, Delta, l_new, l, kept_pole_order, top):
    k = 1
    nu = Rational(dim, 2) - 1
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
	rho_n = Function('rho_n')
	rho_c = Function('rho_c')
	rho = z / (1 + sqrt(1 - z)) ** 2
	
	rules = []
	for i in range(0, derivative_order + 1):
	    rules.append(diff(rho, z, i).subs(z, 0.5))

	# This should be modified for when we need even_derivatives
	for l in range(0, l_max + 1, step):
	    derivatives = []
	    for m in range(0, derivative_order + 1):
		for n in range(1 - (m % 2), min(m, derivative_order + 1 - m), 2):
		    expression = diff(conformal_block(dim, rho_n(z_norm), rho_c(z_conj), delta, l, kept_pole_order), z_norm, m, z_conj, n).subs((rho_n(z_norm) * rho_c(z_conj)) ** (delta / 2), 1)
		    for i in range(m, -1, -1):
		        expression = expression.subs(Derivative(rho_n(z_norm), z_norm, i), rules[i])
		    for j in range(n, -1, -1):
		        expression = expression.subs(Derivative(rho_c(z_conj), z_conj, j), rules[j])
		    
		    # We loop backwards because high derivatives depend on low derivatives
		    # Using replace above instead of subs would be slow
		    derivatives.append(expand(expression))
		    # For the 27th element of the list, say what m derivative and what n derivative it corresponds to
		    if l == 0:
		        self.m_order.append(m)
		        self.n_order.append(n)
	    self.table.append(derivatives)
    
    def get_poles(self, l):
	nu = Rational(self.dim, 2) - 1
	
	k = 1
	ret = []
	while (2 * k) <= self.kept_pole_order:
	    if delta_residue(nu, k, l, 1) != 0:
	        ret.append(delta_pole(nu, k, l, 1))
	    
	    if k < nu + l or self.dim % 2 == 1:
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

class SDP:
    def __init__(self, delta_ext, block_table):
        # Copying everything but the unconvolved table is fine from a memory standpoint
        self.dim = block_table.dim
	self.derivative_order = block_table.derivative_order
	self.kept_pole_order = block_table.kept_pole_order
	self.l_max = block_table.l_max
	self.odd_spins = block_table.odd_spins
	
        self.unitop = 1
	self.bounds = []
	self.table = []
	self.prefactors = []
	
	if block_table.odd_spins:
	    step = 1
	else:
	    step = 2
	
	# Sets up the unitarity bounds
	for l in range(0, block_table.l_max + 1, step):
	    if l == 0:
	        self.bounds.append(Rational(block_table.dim, 2) - 1)
	    else:
	        self.bounds.append(block_table.dim + l - 2)
	
	g = Function('g')
	f = (((1 - z_norm) * (1 - z_conj)) ** delta_ext) * g(z_norm, z_conj)
	
	for l in range(0, block_table.l_max + 1, step):
	    derivatives = []
	    for m in range(0, block_table.derivative_order + 1):
		for n in range(1 - (m % 2), min(m, block_table.derivative_order + 1 - m, 2)):
		    expression = (1 / (factorial(m) * factorial(n))) * diff(f, z_norm, m, z_conj, n)
		    # Now we replace abstract derivatives with the expressions in the unconvolved table
		    for i in range(0, len(block_table.table[0])):
			    expression = expression.replace(Derivative(g(z_norm, z_conj), z_norm, block_table.m_order[i], z_conj, block_table.n_order[i]), block_table.table[l][i])
		    # Putting subs inside evalf appears not to work. Instead we set the precision after subs.
		    derivatives.append(expression.subs([(z_norm, 0.5), (z_conj, 0.5)]))
	    self.table.append(derivatives)
	
	# Separate each element into a polynomial and a prefactor
	for l in range(0, block_table.l_max + 1, step):
	    self.prefactors.append(damped_rational(1, block_table.get_poles(l), 3 - 2 * sqrt(2), delta))
