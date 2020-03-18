"""
collection of jastrow factors (or rather, their logarithms)
"""
import numpy as np

def egas(r, A, F):
    """
    eq 4.5 in the review by Foulkes et al (2001)
    for electron gas jastrow
    """
    return A*(1 - np.exp(-r / F))/r

def egas_prime(r, A, F):
    """
    derivative of egas wrt to r,
    useful for computing the gradient of u
    as well as for cusp condition
    """
    val = F - np.exp(r / F)*F + r
    val *= A*np.exp(-r / F)/(F*r**2)
    return val

def laplacian_egas(r, A, F):
    numerator = A*np.exp(-r / F)
    denominator = r*F**2
    return -numerator / denominator

