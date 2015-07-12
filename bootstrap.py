#!/usr/bin/env python2
import xml.dom.minidom
import scipy.integrate
import scipy.linalg
import scipy
import os

# Use regular sympy sparingly because it is slow
# Every time we explicitly use it, we should consider implementing such a line in C++
from csympy import *
from csympy.lib.csympy_wrapper import *
import sympy

# A bug sometimes occurs when shifting the variable of a polynomial
# It is caused by a constant term, so we include an extra monomial and set it to unity at the end
z_norm = symbols('z_norm')
z_conj = symbols('z_conj')
delta  = symbols('delta')
delta_ext = symbols('delta_ext')
fudge = symbols('fudge')
rho_cross = 0.17157287525380990239

def get_index(array, element):
    if element in array:
        return array.index(element)
    else:
        return -1

def shifted_prefactor(poles, base, x):
    product = 1
    for p in poles:
        product *= x - p
    return (base ** x) / product

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

def leading_block(nu, rho_norm, rho_conj, l):
    if nu == 0:
        ret = sympy.chebyshevt(l, (rho_norm + rho_conj) / (2 * sqrt(rho_norm * rho_conj)))
    else:
        ret = factorial(l) * sympy.gegenbauer(l, nu, (rho_norm + rho_conj) / (2 * sqrt(rho_norm * rho_conj))) / sympy.rf(2 * nu, l)
    return ret / (((1 - (rho_norm * rho_conj)) ** nu) * sqrt((1 + (rho_norm * rho_conj)) ** 2 - (rho_norm + rho_conj) ** 2))

def meromorphic_block(dim, rho_norm, rho_conj, Delta, l_new, l, kept_pole_order, top):
    k = 1
    nu = sympy.Rational(dim, 2) - 1
    # When the recursion relation shifts l, this does not affect the appropriate poles and
    # residues to use which are still determined by the original spin.
    summation = leading_block(nu, rho_norm, rho_conj, l_new)
    
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
	        new_term = res * ((rho_norm * rho_conj) ** k) * omit_product(poles, pole)
	    else:
	        new_term = res * ((rho_norm * rho_conj) ** k) / (Delta - pole)
	    summation += new_term * meromorphic_block(dim, rho_norm, rho_conj, pole + 2 * k, l + 2 * k, l, kept_pole_order - 2 * k, False)
	
	if k < nu + l or dim % 2 == 1:
	    res = delta_residue(nu, k, l, 2)
	    if res != 0:
	        pole = delta_pole(nu, k, l, 2)
		if top == True:
	            new_term = res * ((rho_norm * rho_conj) ** k) * omit_product(poles, pole)
	        else:
	            new_term = res * ((rho_norm * rho_conj) ** k) / (Delta - pole)
	        summation += new_term * meromorphic_block(dim, rho_norm, rho_conj, pole + 2 * k, l, l, kept_pole_order - 2 * k, False)
	
	if k <= (l / 2):
	    res = delta_residue(nu, k, l, 3)
	    if res != 0:
	        pole = delta_pole(nu, k, l, 3)
	        if top == True:
	            new_term = res * ((rho_norm * rho_conj) ** k) * omit_product(poles, pole)
	        else:
	            new_term = res * ((rho_norm * rho_conj) ** k) / (Delta - pole)
	        summation += new_term * meromorphic_block(dim, rho_norm, rho_conj, pole + 2 * k, l - 2 * k, l, kept_pole_order - 2 * k, False)

	k += 1
    return summation

def conformal_block(dim, rho_norm, rho_conj, Delta, l, kept_pole_order):
    return ((rho_norm * rho_conj) ** (Delta / 2)) * meromorphic_block(dim, rho_norm, rho_conj, Delta, l, l, kept_pole_order, True)

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
            print "Spin = " + str(l)
	    derivatives = []
	    old_expression = conformal_block(dim, rho_n, rho_c, delta, l, kept_pole_order)
	    
	    for m in range(0, derivative_order + 1):
		for n in range(0, min(m, derivative_order - m) + 1):
		    # Each loop has one expansion that it can do without
		    if n == 0 and m == 0:
		        expression = old_expression
		    elif n == 0:
		        old_expression = old_expression.diff(z_norm).expand()
			expression = old_expression
		    else:
		        expression = expression.diff(z_conj).expand()
		    
		    deriv = expression * ((rho_n * rho_c) ** (-delta / 2))
		    for i in range(m, 0, -1):
		        deriv = deriv.subs(Derivative(rho_n, [z_norm] * i), rules[i])
		    for j in range(n, 0, -1):
		        deriv = deriv.subs(Derivative(rho_c, [z_conj] * j), rules[j])
		    deriv = deriv.subs(rho_n, rules[0]).subs(rho_c, rules[0])
		    derivatives.append(fudge * deriv.expand())
		    
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
	self.norm = []
	
	g = function_symbol('g', z_norm, z_conj)
	f = (((1 - z_norm) * (1 - z_conj)) ** delta_ext) * g
	
	# Same comments apply here
	for l in range(0, len(block_table.table)):
	    derivatives = []
	    old_expression = f
	    
	    for m in range(0, block_table.derivative_order + 1):
		for n in range(0, min(m, block_table.derivative_order - m) + 1):
		    if n == 0 and m == 0:
		        expression = old_expression
		    elif n == 0:
		        old_expression = old_expression.diff(z_norm).expand()
			expression = old_expression
		    else:
		        expression = expression.diff(z_conj).expand()
		    
		    # Skip even derivatives
		    if (m + n) % 2 == 0:
		        continue
		    
		    # Now we replace abstract derivatives with the expressions in the unconvolved table
		    deriv = expression / (factorial(m) * factorial(n))
		    for i in range(len(block_table.table[0]) - 1, 0, -1):
		        deriv = deriv.subs(Derivative(g, [z_norm] * block_table.m_order[i] + [z_conj] * block_table.n_order[i]), block_table.table[l][i])
		    
		    deriv = deriv.subs(g, block_table.table[l][0])
		    derivatives.append(2 * deriv.subs({z_norm : 0.5, z_conj : 0.5}))
		    
		    # Do this once for the unit operator too
		    if l == 0:
		        deriv = expression / (factorial(m) * factorial(n))
			deriv = deriv.subs(Derivative(g, [z_norm]), 0).subs(Derivative(g, [z_conj]), 0).subs(g, 1)
			self.norm.append(2 * deriv.subs({z_norm : 0.5, z_conj : 0.5}))
	    self.table.append(derivatives)

class SDP:
    def __init__(self, conv_block_table, dim_ext):
        # Same story here
        self.dim = conv_block_table.dim
	self.derivative_order = conv_block_table.derivative_order
	self.kept_pole_order = conv_block_table.kept_pole_order
	self.l_max = conv_block_table.l_max
	self.odd_spins = conv_block_table.odd_spins
	self.norm = []
	self.table = []
	
	max_index = 0
	max_unit = 0

	for l in range(0, len(conv_block_table.table)):
	    derivatives = []
	    for i in range(0, len(conv_block_table.table[l])):
	        derivatives.append(conv_block_table.table[l][i].subs(delta_ext, dim_ext))
		if l == 0:
		    unit = conv_block_table.norm[i].subs(delta_ext, dim_ext)
		    if unit > max_unit:
		        max_unit = unit
			max_index = i
		    self.norm.append(unit)
	    self.table.append(derivatives)
	
	# Translate between the mathematica definition and the bootstrap definition of SDP
	for l in range(0, len(conv_block_table.table)):
	    const = self.table[l][max_index] / self.norm[max_index]
	    for i in range(0, len(self.table[l])):
	        self.table[l][i] -= const * self.norm[i]
	    self.table[l] = [const] + self.table[l][:max_index] + self.table[l][max_index + 1:]
    
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
	    point = -(sympy.pi ** 2) * ((4 * d - 1) ** 2) / (64 * (degree + 1) * sympy.log(3 - 2 * sympy.sqrt(2)))
	    ret.append(point)
	return ret
    
    def integrand(self, x, pos, shift, poles):
        return (x ** pos) * shifted_prefactor(poles, rho_cross, x + shift)
    
    def write_xml(self, gap, gapped_spin):
        laguerre_points = []
	laguerre_degrees = []
	
	doc = xml.dom.minidom.Document()
	root_node = doc.createElement("sdp")
	doc.appendChild(root_node)
	
	objective_node = doc.createElement("objective")
	matrices_node = doc.createElement("polynomialVectorMatrices")
	root_node.appendChild(objective_node)
	root_node.appendChild(matrices_node)
	
	# Here, we use indices that match the SDPB specification
	for n in range(0, len(self.table[0])):
	    elt_node = doc.createElement("elt")
	    elt_node.appendChild(doc.createTextNode("0"))
	    objective_node.appendChild(elt_node)
	
	for j in range(0, len(self.table)):
	    if self.odd_spins:
	        spin = j
	    else:
	        spin = 2 * j
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
	    if spin == gapped_spin:
	        delta_min = gap
	    elif spin == 0:
	        delta_min = sympy.Rational(self.dim, 2) - 1
	    else:
	        delta_min = self.dim + spin - 2
	    vector_node = doc.createElement("polynomialVector")
	    for n in range(0, len(self.table[j])):
	        expression = self.table[j][n].expand()
		# Impose unitarity bounds and the specified gap
		expression = expression.subs(delta, delta + delta_min).expand()
		expression = expression.subs(fudge, 1).expand()
		coeff_list = sorted(expression.args, key = self.extract_power)
		degree = max(degree, len(coeff_list) - 1)
		
	        polynomial_node = doc.createElement("polynomial")
		for d in range(0, len(coeff_list)):
		    if d == 0:
		        coeff = eval_double(coeff_list[0])
		    else:
		        coeff = eval_double(coeff_list[d].args[0])
		    
		    coeff_node = doc.createElement("coeff")
		    coeff_node.appendChild(doc.createTextNode(str.format('{0:.40f}', coeff)))
		    polynomial_node.appendChild(coeff_node)
		vector_node.appendChild(polynomial_node)
	    elements_node.appendChild(vector_node)
	    
	    poles = get_poles(self.dim, spin, self.kept_pole_order)
	    index = get_index(laguerre_degrees, degree)
	    if index == -1:
	        points = self.make_laguerre_points(degree)
		laguerre_points.append(points)
		laguerre_degrees.append(degree)
	    else:
	        points = laguerre_points[index]
	    
	    for d in range(0, degree + 1):
	        elt_node = doc.createElement("elt")
		elt_node.appendChild(doc.createTextNode(str.format('{0:.40f}', points[d].evalf())))
		sample_point_node.appendChild(elt_node)
		damped_rational = shifted_prefactor(poles, rho_cross, points[d] + delta_min)
		elt_node = doc.createElement("elt")
		elt_node.appendChild(doc.createTextNode(str.format('{0:.40f}', damped_rational.evalf())))
		sample_scaling_node.appendChild(elt_node)
	    
	    bands = []
	    matrix = []
	    # We numerically integrate to find the moment matrix for now
	    for d in range(0, 2 * (degree / 2) + 1):
	        result = scipy.integrate.quad(self.integrand, 0, scipy.Inf, args = (d, delta_min, poles))
	        bands.append(result[0])
	    for r in range(0, (degree / 2) + 1):
	        new_entries = []
	        for s in range(0, (degree / 2) + 1):
		    new_entries.append(bands[r + s])
		matrix.append(new_entries)
	    matrix = scipy.linalg.cholesky(matrix)
	    matrix = scipy.linalg.inv(matrix)
	    
	    for d in range(0, (degree / 2) + 1):
		polynomial_node = doc.createElement("polynomial")
		for q in range(0, d + 1):
		    coeff_node = doc.createElement("coeff")
		    coeff_node.appendChild(doc.createTextNode(str.format('{0:.40f}', matrix[q][d])))
		    polynomial_node.appendChild(coeff_node)
		bilinear_basis_node.appendChild(polynomial_node)
	    
	    matrix_node.appendChild(rows_node)
	    matrix_node.appendChild(cols_node)
	    matrix_node.appendChild(elements_node)
	    matrix_node.appendChild(sample_point_node)
	    matrix_node.appendChild(sample_scaling_node)
	    matrix_node.appendChild(bilinear_basis_node)
	    matrices_node.appendChild(matrix_node)
	    
	xml_file = open("mySDP.xml", 'wb')
	doc.writexml(xml_file, addindent = "    ", newl = '\n')
	xml_file.close()
	doc.unlink()
    
    def bisect(self, lower, upper, threshold, spin):
        test = (lower + upper) / 2.0
        if abs(upper - lower) < threshold:
	    return upper
	else:
	    print "Trying " + str(test)
	    self.write_xml(test, spin)
	    os.spawnlp(os.P_WAIT, "sdpb", "-s", "mySDP.xml", "--findPrimalFeasible", "--findDualFeasible", "--noFinalCheckpoint")
	    out_file = open("mySDP.out", 'r')
	    terminate_line = out_file.next()
	    terminate_reason = terminate_line.partition(" = ")[-1]
	    out_file.close()
	    
	    excluded = False
	    if terminate_reason == '"found dual feasible solution";\n':
	        excluded = True
	    if excluded == True:
	        return self.bisect(lower, test, threshold, spin)
	    else:
	        return self.bisect(test, upper, threshold, spin)
