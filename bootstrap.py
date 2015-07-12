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

def conformal_block(dim, Z_norm, Z_conj, Delta, l, kept_pole_order):
    rho_norm = Z_norm / (1 + sqrt(1 - Z_norm)) ** 2
    rho_conj = Z_conj / (1 + sqrt(1 - Z_conj)) ** 2
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
	
	rho_norm = z_norm / (1 + sqrt(1 - z_norm)) ** 2
        rho_conj = z_conj / (1 + sqrt(1 - z_conj)) ** 2
        r = sqrt(rho_norm * rho_conj)

	# This should be modified for when we need even_derivatives
	for l in range(0, l_max + 1, step):
	    derivatives = []
	    for m in range(0, derivative_order + 1):
		for n in range(1 - (m % 2), min(m, derivative_order + 1 - m), 2):
		    # Differentiate with the power of r, then strip it off
		    # It will be added once more when we make the prefactor
		    # The poles will be part of it too
		    expression = diff(conformal_block(dim, z_norm, z_conj, delta, l, kept_pole_order), z_norm, m, z_conj, n).subs(r ** delta, 1)
		    expression = expression.subs([(z_norm, 0.5), (z_conj, 0.5)])
		    expression = expand(expression)
		    # If setting the exponent to 1 works, we only need to expand which we can do with floating points
		    derivatives.append(expression)
		    # For the 27th element of the list, say what m derivative and what n derivative it corresponds to
		    if l == 0:
		        self.m_order.append(m)
		        self.n_order.append(n)
	    self.table.append(derivatives)
