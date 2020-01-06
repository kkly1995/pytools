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
