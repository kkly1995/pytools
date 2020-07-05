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

def mcmillan(r, a1, a2):
    """
    jastrow factor of mcmillan (1965)
    this is -log(f) rather than f itself,
    where f is eq 5 in that paper
    """
    return (a1 / r)**a2

def mcmillan_prime(r, a1, a2):
    """
    derivative of mcmillan wrt r
    """
    return -(a2 / r)*mcmillan(r, a1, a2)

def laplacian_mcmillan(r, a1, a2):
    val = mcmillan(r, a1, a2)
    val *= (a2 - 1)*a2
    val /= r**2
    return val

def pade(r, a, b, c):
    """
    pade form
    see e.g. qmcpack manual
    note the sign (positive!)
    """
    numerator = a*r + c*r**2
    denominator = 1 + b*r
    return numerator/denominator

def pade_prime(r, a, b, c):
    numerator = a + c*r*(2 + b*r)
    denominator = (1 + b*r)**2
    return numerator/denominator

def laplacian_pade(r, a, b, c):
    numerator = 3 + b*r
    numerator = 3 + b*r*numerator
    numerator = a + c*r*numerator
    numerator *= 2
    denominator = r*(1 + b*r)**3
    return numerator/denominator
