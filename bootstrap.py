#!/usr/bin/env python2
from sympy import *

dim = 2
nu = Rational(dim, 2) - 1

def damped_rational(const, poles, base, delta):
    product = 1
    for p in poles:
        product *= delta - p
    return const * (base ** delta) / product

def delta_pole(k, l, series):
    if series == 1:
        return 1 - l - 2 * k
    elif series == 2:
	return 1 + nu - k
    else:
	return 1 + l + 2 * nu - 2 * k

def delta_residue(k, l, series):
    if series == 1:
        ret = - ((k * factorial(2 * k) ** 2) / (2 ** (4 * k - 1) * factorial(k) ** 4))
	if l == 0 and nu == 0:
	    # Check if this needs to be 2
	    return ret
	else:
	    return ret * (rf(l + 2 * nu, 2 * k) / rf(l + nu, 2 * k))
    elif series == 2:
	return - rf(nu, k) * rf(1 - nu, k) * (rf((nu + l + 1 - k) / 2, k) ** 2 / rf((nu + l - k) / 2, k) ** 2) * (k / factorial(k) ** 2) * ((nu + l - k) / (nu + l + k))
    else:
	return - (rf(1 + l - 2 * k, 2 * k) / rf(1 + nu + l - 2 * k, 2 * k)) * ((k * factorial(2 * k) ** 2) / (2 ** (4 * k - 1) * factorial(k) ** 4))

def leading_block(r, eta, l):
    if nu == 0:
        #ret = cos(2 * l * asin(sqrt((1 - eta) / 2)))
	ret = chebyt(l, eta)
    else:
        ret = factorial(l) * gegenbauer(l, nu, eta) / rf(2 * nu, l)
    return ret / (((1 - r ** 2) ** nu) * sqrt((1 + r ** 2) ** 2 - 4 * (r * eta) ** 2))

def meromorphic_block(r, eta, delta, l, kept_pole_order):
    k = 1
    summation = leading_block(r, eta, l)
    while (2 * k) <= kept_pole_order:
	summation += delta_residue(k, l, 1) * (r ** (2 * k)) * meromorphic_block(r, eta, delta_pole(k, l, 1) + 2 * k, l + 2 * k, kept_pole_order - 2 * k) / (delta - delta_pole(k, l, 1))
	
	if k < nu + l or dim % 2 == 1:
	    summation += delta_residue(k, l, 2) * (r ** (2 * k)) * meromorphic_block(r, eta, delta_pole(k, l, 2) + 2 * k, l, kept_pole_order - 2 * k) / (delta - delta_pole(k, l, 2))
	
	if k <= (l / 2):
	    summation += delta_residue(k, l, 3) * (r ** (2 * k)) * meromorphic_block(r, eta, delta_pole(k, l, 3) + 2 * k, l - 2 * k, kept_pole_order - 2 * k) / (delta - delta_pole(k, l, 3))
	k += 1
    return summation

def conformal_block(z_norm, z_conj, delta, l, kept_pole_order):
    rho_norm = z_norm / (1 + sqrt(1 - z_norm)) ** 2
    rho_conj = z_conj / (1 + sqrt(1 - z_conj)) ** 2
    r = sqrt(rho_norm * rho_conj)
    eta = (rho_norm + rho_conj) / (2 * r)
    return (r ** delta) * meromorphic_block(r, eta, delta, l, kept_pole_order)
