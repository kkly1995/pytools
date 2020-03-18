import pytools.math as pm
from math import isclose
import numpy as np

def func(x):
    """
    sum all the positions of the particles r1 + r2 + ...
    which should be a vector of the same dimension as e.g. r1
    then take the square norm
    this can be shown to be equivalent to just a matrix multiplication,
    then summing all the entries
    """
    return np.sum(np.matmul(x, x.transpose()))

def test_second_deriv():
    """
    test finite difference second derivative against func
    for all components this second derivative should be 2
    """
    r = np.random.rand(10, 5) #10 particles in 5d
    for i in range(10):
        for k in range(5):
            msg = 'test failed on square of sum'
            computed = pm.second_deriv(func, r, i, k, 0.1)
            assert isclose(2, computed), msg

def test_total_laplacian():
    """
    test total laplacian against func
    also checks to make sure that this is equivalent to second_deriv()
    upon summing over all i, k
    """
    r = np.random.rand(7, 4) #7 particles in 4d
    msg1 = 'test failed on square of sum'
    computed = pm.total_laplacian(func, r, 0.1)
    assert isclose(7*4*2, computed), msg1
    #check if summing over second_deriv gives the same result
    from_second_deriv = 0
    for i in range(7):
        for k in range(4):
            from_second_deriv += pm.second_deriv(func, r, i, k, 0.1)
    msg2 = 'total_laplacian and second_deriv dont match'
    assert isclose(computed, from_second_deriv), msg2

def test_gradient():
    """
    test gradient against func
    the result should be the same for all particles:
    the kth component of the gradient wrt particle i is just
    two times the sum of the kth component of every particle
    also check that it outputs the same shape as input
    """
    r = np.random.rand(5,3) #5 particles in 3d
    computed = pm.gradient(func, r, 0.01) #smaller step
    #first check that each gradient is the same
    rescaled = computed / computed[0] #divide every gradient by the first
    #since each gradient is the same, every entry should be 1
    msg1 = 'gradient doesnt give particle-independent result'
    assert np.isclose(np.ones(r.shape), rescaled).all(), msg1
    #now just check each component of the first gradient
    msg2 = 'gradient failed on square of sum'
    for k in range(3):
        assert isclose(sum(r[:,k])*2, computed[0,k], rel_tol = 0.01), msg2
        #relative tolerance is relaxed since forward difference sucks

def test_laplacian():
    """
    test laplacian against func
    the result should be 2*d where d is the dimension
    for each particle
    
    also test that summing the output of laplacian() matches
    the output of total_laplacian()
    """
    r = np.random.rand(7,2) #7 particles in 2d
    computed = pm.laplacian(func, r, 0.1)
    msg1 = 'laplacian failed on square of sum'
    for i in range(7):
        assert isclose(4, computed[i]), msg1
    msg2 = 'laplacian isnt consistent with total_laplacian'
    assert isclose(sum(computed), pm.total_laplacian(func, r, 0.1)), msg2

def test_displacement():
    """
    check that no component is greater than half the side length of cube
    check that in 1D, for two particles on opposite ends, the normal distance
    (i.e. not the minimum image) is recovered just by subtracting the side length
    """
    size = 6.9
    msg1 = 'minimum image is not inside first cell'
    #generate 69 random displacements
    r1 = np.random.rand(69,3)*size
    r2 = np.random.rand(69,3)*size
    r = r1 - r2
    computed = pm.minimum_image(r, size)
    assert (np.abs(computed) < size/2).all(), msg1
    #in 1D, place one particle near the left and and one near the right
    x1 = np.random.rand() - size/2
    x2 = (size/2) - np.random.rand()
    image = pm.minimum_image(x1 - x2, size)
    msg2 = 'minimum image failed in 1D'
    assert isclose(image - size, x1 - x2), msg2

def test_latticeVectorInfo():
    """
    check that this correctly gets the angles for fcc
    and hexagonal
    """
    a = 6.9 #lattice constant
    fcc1 = [a/2, a/2, 0]
    fcc2 = [a/2, 0, a/2]
    fcc3 = [0, a/2, a/2]
    msg1 = 'latticeVectorInfo failed on fcc'
    computed = pm.latticeVectorInfo(fcc1, fcc2, fcc3)
    for i in range(3):
        assert isclose(a / np.sqrt(2), computed[i]), msg1
        assert isclose(60, computed[i+3]), msg1
    #check hexagonal
    c = 1.69*a
    hex1 = [a, 0, 0]
    hex2 = [a/2, np.sqrt(3)*a/2, 0]
    hex3 = [0, 0, c]
    computed = pm.latticeVectorInfo(hex1, hex2, hex3)
    msg2 = 'latticeVectorInfo failed on simple hexagonal'
    for i in range(2):
        assert isclose(a, computed[i]), msg2
        assert isclose(90, computed[i+3]), msg2
    assert isclose(c, computed[2]), msg2
    assert isclose(60, computed[5]), msg2

def test_coulomb_potential():
    #this test also relies on electron and cell classes
    from pytools.pbc import electron, cell
    elec = electron(17, 19)
    elec.start_random()
    v = 6.9*np.eye(3) #cubic cell, length 6.9
    geometry = cell(v[0], v[1], v[2])
    up_displacement = geometry.crystal_to_cart(elec.up_table)
    down_displacement = geometry.crystal_to_cart(elec.down_table)
    updown_displacement = geometry.crystal_to_cart(elec.up_down_table)
    #begin computing potentials
    up_potential = pm.coulomb_potential(up_displacement)
    down_potential = pm.coulomb_potential(down_displacement)
    updown_potential = pm.coulomb_potential(updown_displacement, False)
    #computation using loops
    up_manual = 0
    down_manual = 0
    updown_manual = 0
    r_up = np.linalg.norm(up_displacement, axis=-1)
    r_down = np.linalg.norm(down_displacement, axis=-1)
    r_updown = np.linalg.norm(updown_displacement, axis=-1)
    for i in range(elec.N_up):
        for j in range(i):
            up_manual += 1./r_up[i,j]
    for i in range(elec.N_down):
        for j in range(i):
            down_manual += 1./r_down[i,j]
    for i in range(elec.N_up):
        for j in range(elec.N_down):
            updown_manual += 1./r_updown[i,j]
    assert isclose(up_potential, up_manual), \
            'coulomb_potential failed for up electrons'
    assert isclose(down_potential, down_manual), \
            'coulomb_potential failed for down electrons'
    assert isclose(updown_potential, updown_manual), \
            'coulomb_potential failed for up-down electrons'
