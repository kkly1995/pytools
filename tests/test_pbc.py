import pytools.pbc as pbc
import numpy as np
from scipy.special import erfc
from math import isclose

def test_ewald():
    """
    currently just tests implementation against for loops
    for one_species=False (nonsymmetric displacement table)
    need to eventually check it against madelung constants
    """
    L = 6.9
    displacement = L*np.random.rand(11, 11, 3)
    r = np.linalg.norm(displacement, axis=-1)
    #make supercell to get kvecs
    v = L*np.eye(3)
    supercell = pbc.cell(v[0], v[1], v[2])
    k = supercell.fermi_sea(2)[1:,:3]
    k *= 2*np.pi/L
    kappa = 6/L
    volume = supercell.volume
    manual = 0
    for i in range(11):
        for j in range(11):
            manual += erfc(kappa*r[i,j]) / r[i,j]
    #begin longrange
    for i in range(11):
        for j in range(11):
            for l in range(len(k)):
                ksq = np.dot(k[l], k[l])
                kr = np.dot(k[l], displacement[i,j])
                prefactor = 4*np.pi/volume
                pw = np.cos(kr)
                manual += prefactor*np.exp(-ksq / (4*kappa**2))*pw/ksq
    #self-term, there are 22 charges
    manual -= kappa*22 / np.sqrt(np.pi)
    assert isclose(manual, pbc.ewald(displacement, kappa, k, volume))
