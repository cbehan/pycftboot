#!/usr/bin/env python2
"""
This is a tutorial that applies conformal bootstrap logic with PyCFTBoot to make
non-trivial statements about four different conformal field theories. These are
taken from the examples in section 4 of arXiv:1602.02810. More information can
be found by reading the other sections or the documentation strings of PyCFTBoot.
These can be accessed by running `pydoc -g`.
"""
# Imports the package
import pycftboot as bootstrap
# The conformal blocks needed for a given run are calculated as a sum over poles.
# Demand that poles with small residues are approximated by poles with large ones.
bootstrap.cutoff = 1e-10

def cprint(message):
    print("\033[94m" + message + "\033[0m")

print("Welcome to the PyCFTBoot tutorial!")
print("Please read the comments and watch for coloured text.")
print("Which theory would you like to study?")
print("These take increasingly long amounts of time.")
print("1. 3D Ising model (even sector only).")
print("2. 3D O(3) model.")
print("3. 3D Ising model (odd sector as well).")
print("4. 4D N = 1 supersymmetric model.")
choice = int(input("Choice: "))

if choice == 1:
    # Concentrates on external scalars of 0.518, close to the 3D Ising value.
    dim_phi = 0.518
    cprint("Finding basic bound at external dimension " + str(dim_phi) + "...")
    # Spatial dimension.
    dim = 3
    # Dictates the number of poles to keep and therefore the accuracy of a conformal block.
    k_max = 20
    # Says that conformal blocks for spin-0 up to and including spin-14 should be computed.
    l_max = 14
    # Conformal blocks are functions of (a, b) and many derivatives of each should be kept for strong bounds.
    # This says to keep derivatives up to fourth order in b.
    n_max = 4
    # For a given n, this states how many a derivatives should be included beyond 2 * (n - n_max).
    m_max = 2
    # Generates the table.
    table1 = bootstrap.ConformalBlockTable(dim, k_max, l_max, m_max, n_max)
    # Computes the convolution.
    table2 = bootstrap.ConvolvedBlockTable(table1)
    # Sets up a semidefinite program that we can use to study this.
    sdp = bootstrap.SDP(dim_phi, table2)
    # We think it is perfectly fine for all internal scalars coupling to our external one to have dimension above 0.7.
    lower = 0.7
    # Conversely, we think it is a problem for crossing symmetry if they all have dimension above 1.7.
    upper = 1.7
    # The boundary between these regions will be found within an error of 0.01.
    tol = 0.01
    # The 0.7 and 1.7 are our guesses for scalars, not some other type of operator.
    channel = 0
    # Calls SDPB to compute the bound.
    result = sdp.bisect(lower, upper, tol, channel)
    cprint("If crossing symmetry and unitarity hold, the maximum gap we can have for Z2-even scalars is: " + str(result))
    cprint("Checking if (" + str(dim_phi) + ", " + str(result) + ") is still allowed if we require only one relevant Z2-even scalar...")
    # States that the continuum of internal scalars being checked starts at 3.
    sdp.set_bound(channel, float(dim))
    # States that the point near the boundary that we found should be the one exception.
    sdp.add_point(channel, result)
    # Calls SDPB to check.
    allowed = sdp.iterate()
    if (allowed):
        cprint("Yes")
    else:
        cprint("No")
    cprint("Checking if (" + str(dim_phi) + ", 1.2) is allowed under the same conditions...")
    # Removes the point we previously set by calling this with one missing argument.
    sdp.add_point(channel)
    # Adds a different one at a smaller dimension.
    sdp.add_point(channel, 1.2)
    # Checks again.
    allowed = sdp.iterate()
    if (allowed):
        cprint("Yes")
    else:
        cprint("No")

if choice == 2:
    # The 0.52 value turns out to be close to a kink.
    dim_phi1 = 0.52
    cprint("Finding basic bound on singlets at external dimension " + str(dim_phi1) + "...")
    # Parameters like those in example 1.
    dim = 3
    k_max = 20
    l_max = 15
    m_max = 1
    n_max = 3
    # This time we need to keep odd spins because O(N) models have antisymmetric tensors.
    table1 = bootstrap.ConformalBlockTable(dim, k_max, l_max, m_max, n_max, odd_spins = True)
    # Computes the two convolutions needed by the sum rule.
    table2 = bootstrap.ConvolvedBlockTable(table1, symmetric = True)
    table3 = bootstrap.ConvolvedBlockTable(table1)
    # Specializes to N = 3.
    N = 3.0
    # First vector: 0 * table3, 1 * table3, 1 * table2
    vec1 = [[0, 1], [1, 1], [1, 0]]
    # Second vector: 1 * table3, (1 - (2 / N)) * table3, -(1 + (2 / N)) * table2
    vec2 = [[1, 1], [1.0 - (2.0 / N), 1], [-(1.0 + (2.0 / N)), 0]]
    # Third vector: 1 * table3, -1 * table3, 1 * table2
    vec3 = [[1, 1], [-1, 1], [1, 0]]
    # The spins of these irreps (with arbitrary names) are even, even, odd.
    info = [[vec1, 0, "singlet"], [vec2, 0, "symmetric"], [vec3, 1, "antisymmetric"]]
    # Sets up an SDP here.
    sdp1 = bootstrap.SDP(dim_phi1, [table2, table3], vector_types = info)
    # This time channel needs two labels.
    channel1 = [0, "singlet"]
    result = sdp1.bisect(0.7, 1.8, 0.01, channel1)
    cprint("If crossing symmetry and unitarity hold, the maximum gap we can have for singlet scalars is: " + str(result))
    cprint("Bounding the OPE coefficient for the stress-energy tensor...")
    # The spin is now 2 and the dimension is 3.
    channel2 = [2, "singlet"]
    dim_t = dim
    # Calls SDPB to return a squared OPE coefficient bound.
    result1 = sdp1.opemax(dim_t, channel2)
    cprint("Bounding the same coefficient in the free theory to get a point of comparison...")
    # Sets up a new SDP where this time, the external scalar has a dimension very close to unitarity.
    dim_phi2 = 0.5001
    sdp2 = bootstrap.SDP(dim_phi2, [table2, table3], vector_types = info)
    result2 = sdp2.opemax(dim_t, channel2)
    # Uses the central charge formula which follows from the Ward identity to compute the ratio.
    ratio = ((result2 / result1) * (dim_phi1 / dim_phi2)) ** 2
    cprint("The central charge of the theory at " + str(dim_phi1) + " is " + str(ratio) + " times the free one.")

# A function used for the multi-correlator 3D Ising example.
def convolved_table_list(tab1, tab2, tab3):
    f_tab1a = bootstrap.ConvolvedBlockTable(tab1)
    f_tab1s = bootstrap.ConvolvedBlockTable(tab1, symmetric = True)
    f_tab2a = bootstrap.ConvolvedBlockTable(tab2)
    f_tab2s = bootstrap.ConvolvedBlockTable(tab2, symmetric = True)
    f_tab3 = bootstrap.ConvolvedBlockTable(tab3)
    return [f_tab1a, f_tab1s, f_tab2a, f_tab2s, f_tab3]

if choice == 3:
    cprint("Generating the tables needed to test two points...")
    dim = 3
    # Poles would be too approximate otherwise.
    bootstrap.cutoff = 0
    # First odd scalar, first even scalar.
    pair1 = [0.518, 1.412]
    pair2 = [0.53, 1.412]
    # Generates three tables, two of which depend on the dimension differences.
    g_tab1 = bootstrap.ConformalBlockTable(dim, 20, 20, 2, 4)
    g_tab2 = bootstrap.ConformalBlockTable(dim, 20, 20, 2, 4, pair1[1] - pair1[0], pair1[0] - pair1[1], odd_spins = True)
    g_tab3 = bootstrap.ConformalBlockTable(dim, 20, 20, 2, 4, pair1[0] - pair1[1], pair1[0] - pair1[1], odd_spins = True)
    # Uses the function above to return the convolved tables we need.
    tab_list1 = convolved_table_list(g_tab1, g_tab2, g_tab3)
    # One of the three tables above does not need to be regenerated for the next point.
    g_tab4 = bootstrap.ConformalBlockTable(dim, 20, 20, 2, 4, pair2[1] - pair2[0], pair2[0] - pair2[1], odd_spins = True)
    g_tab5 = bootstrap.ConformalBlockTable(dim, 20, 20, 2, 4, pair2[0] - pair2[1], pair2[0] - pair2[1], odd_spins = True)
    tab_list2 = convolved_table_list(g_tab1, g_tab4, g_tab5)
    # Saves and deletes tables that are no longer needed and might take up a lot of memory.
    for tab in [g_tab1, g_tab2, g_tab3, g_tab4, g_tab5]:
        # A somewhat descriptive name.
        tab.dump("tab_" + str(tab.delta_12) + "_" + str(tab.delta_34))
        del tab
    # Third vector: 0, 0, 1 * table4 with one of each dimension, -1 * table2 with only pair[0] dimensions, 1 * table3 with only pair[0] dimensions
    vec3 = [[0, 0, 0, 0], [0, 0, 0, 0], [1, 4, 1, 0], [-1, 2, 0, 0], [1, 3, 0, 0]]
    # Second vector: 0, 0, 1 * table4 with one of each dimension, 1 * table2 with only pair[0] dimensions, -1 * table3 with only pair[0] dimensions
    vec2 = [[0, 0, 0, 0], [0, 0, 0, 0], [1, 4, 1, 0], [1, 2, 0, 0], [-1, 3, 0, 0]]
    # The first vector has five components as well but they are matrices of quads, not just the quads themselves.
    m1 = [[[1, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0]]]
    m2 = [[[0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [1, 0, 1, 1]]]
    m3 = [[[0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0]]]
    m4 = [[[0, 0, 0, 0], [0.5, 0, 0, 1]], [[0.5, 0, 0, 1], [0, 0, 0, 0]]]
    m5 = [[[0, 1, 0, 0], [0.5, 1, 0, 1]], [[0.5, 1, 0, 1], [0, 1, 0, 0]]]
    vec1 = [m1, m2, m3, m4, m5]
    # Spins for these again go even, even, odd.
    info = [[vec1, 0, "z2-even-l-even"], [vec2, 0, "z2-odd-l-even"], [vec3, 1, "z2-odd-l-odd"]]
    cprint("Checking if (" + str(pair1[0]) + ", " + str(pair1[1]) + ") is allowed if we require only one relevant Z2-odd scalar...")
    sdp1 = bootstrap.SDP(pair1, tab_list1, vector_types = info)
    # The pair[1] scalar is Z2-even so have the corresponding channel start here.
    sdp1.set_bound([0, "z2-even-l-even"], pair1[1])
    # The Z2-odd scalars should start at 3 instead and just have pair[0] as a point given our assumption.
    sdp1.set_bound([0, "z2-odd-l-even"], dim)
    sdp1.add_point([0, "z2-odd-l-even"], pair1[0])
    # In this problem, a ruled out point may have primal error smaller than dual error unless we run for much longer.
    sdp1.set_option("dualErrorThreshold", 1e-15)
    allowed = sdp1.iterate()
    if (allowed):
        cprint("Yes")
    else:
        cprint("No")
    cprint("Checking if (" + str(pair2[0]) + ", " + str(pair2[1]) + ") is allowed under the same conditions...")
    # All bounds / points changed in the first SDP will be changed again so we may use it as a prototype.
    sdp2 = bootstrap.SDP(pair2, tab_list2, vector_types = info, prototype = sdp1)
    # Does the exact same testing for the second point.
    sdp2.set_bound([0, "z2-even-l-even"], pair2[1])
    sdp2.set_bound([0, "z2-odd-l-even"], dim)
    sdp2.add_point([0, "z2-odd-l-even"], pair2[0])
    sdp2.set_option("dualErrorThreshold", 1e-15)
    allowed = sdp2.iterate()
    if (allowed):
        cprint("Yes")
    else:
        cprint("No")

if choice == 4:
    # This is where a kink begins to appear.
    dim_phi = 1.4
    cprint("Finding a bound on the singlets...")
    # Generates a fairly demanding table in 3.99 dimensions.
    k_max = 25
    l_max = 26
    m_max = 3
    n_max = 5
    g_tab = bootstrap.ConformalBlockTable(3.99, k_max, l_max, m_max, n_max, odd_spins = True)
    # Bring reserved symbols into our namespace to avoid typing "bootstrap" in what follows.
    delta = bootstrap.delta
    ell = bootstrap.ell
    # Four coefficients that show up in the 4D N = 1 expression for superconformal blocks.
    c1 = (delta + ell + 1) * (delta - ell - 1) * (ell + 1)
    c2 = -(delta + ell) * (delta - ell - 1) * (ell + 2)
    c3 = -(delta - ell - 2) * (delta + ell + 1) * ell
    c4 = (delta + ell) * (delta - ell - 2) * (ell + 1)
    # We have c1 beside (delta + 0, ell + 0), c2 beside (delta + 1, ell + 1), c3 beside (delta + 1, ell - 1) and c4 beside (delta + 2, ell).
    combo1 = [[c1, 0, 0], [c2, 1, 1], [c3, 1, -1], [c4, 2, 0]]
    # The second linear combination has signs flipped on the parts with odd spin shift.
    combo2 = combo1
    combo2[1][0] *= -1
    combo2[2][0] *= -1
    # This makes all of the convolved block tables we need.
    f_tab1a = bootstrap.ConvolvedBlockTable(g_tab)
    f_tab1s = bootstrap.ConvolvedBlockTable(g_tab, symmetric = True)
    f_tab2a = bootstrap.ConvolvedBlockTable(g_tab, content = combo1)
    f_tab2s = bootstrap.ConvolvedBlockTable(g_tab, content = combo1, symmetric = True)
    f_tab3 = bootstrap.ConvolvedBlockTable(g_tab, content = combo2)
    tab_list = [f_tab1a, f_tab1s, f_tab2a, f_tab2s, f_tab3]
    # Sets up a vectorial sum rule just like in example 2.
    vec1 = [[1, 4], [1, 2], [1, 3]]
    vec2 = [[-1, 4], [1, 2], [1, 3]]
    vec3 = [[0, 0], [1, 0], [-1, 1]]
    info = [[vec1, 0, "singlet"], [vec2, 1, "antisymmetric"], [vec3, 0, "symmetric"]]
    # Allocates an SDP and makes it easier for a problem to be recognized as dual feasible.
    sdp = bootstrap.SDP(dim_phi, tab_list, vector_types = info)
    sdp.set_option("dualErrorThreshold", 1e-22)
    # Goes through all the spins and tells the symmetric channel to contain a BPS operator and then a gap.
    for l in range(0, l_max + 1, 2):
        sdp.add_point([l, "symmetric"], 2 * dim_phi + l)
        sdp.set_bound([l, "symmetric"], abs(2 * dim_phi - 3) + 3 + l)
    # Does a long test.
    result = sdp.bisect(3.0, 4.25, 0.01, [0, "singlet"])
    cprint("If crossing symmetry and unitarity hold, the maximum gap we can have for singlet scalars is: " + str(result))
