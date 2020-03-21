import pytools.jastrow as pj
import numpy as np
from math import isclose

def test_egas():
    """
    check that as r goes to 0, u goes to A/F
    so set A and F to random numbers of order 1
    and fit u(r) for r from 0.0001 to 0.0002
    which should be linear

    because this relies on extrapolation, tolerance is relaxed
    for isclose()
    """
    A = np.random.rand() + 1
    F = np.random.rand() + 1
    r = np.linspace(0.0001, 0.0002)
    coef = np.polyfit(r, pj.egas(r, A, F), 1)
    assert isclose(coef[1], A/F, rel_tol=1e-3), \
            'electron-electron jastrow does not have correct r=0 behavior'

def test_egas_prime():
    """
    check that finite difference estimate of derivative of u
    matches u_prime

    again tolerance is relaxed for isclose()
    """
    A = np.random.rand() + 1
    F = np.random.rand() + 1
    r = np.linspace(1, 2)
    estimate = np.gradient(pj.egas(r, A, F), r[1] - r[0])
    actual = pj.egas_prime(r, A, F)
    #ignore the ends, whose finite difference estimates are way worse
    assert np.isclose(estimate[1:-1], actual[1:-1], rtol=1e-3).all(), \
            'electron-electron jastrow derivative is not correct'
            
def test_mcmillan():
    """
    check that for r = 1, the mcmillan jastrow gives the correct value
    """
    a1 = np.random.rand() + 1
    a2 = np.random.rand() + 1
    r = 1
    assert isclose(a1**a2, pj.mcmillan(r, a1, a2)), \
        'mcmillan jastrow failed for r = 1'

def test_mcmillan_prime():
    """
    check that finite difference estimate of derivative of mcmillan
    matches mcmillan_prime
    """
    a1 = np.random.rand() + 1
    a2 = np.random.rand() + 1
    r = np.linspace(1, 2)
    estimate = np.gradient(pj.mcmillan(r, a1, a2), r[1] - r[0])
    actual = pj.mcmillan_prime(r, a1, a2)
    assert np.isclose(estimate[1:-1], actual[1:-1], rtol=1e-3).all(), \
            'mcmillan jastrow derivative is not correct'
