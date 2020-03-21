import pytools.pbc as pbc
import numpy as np
from scipy.special import erfc
from math import isclose

def test_electron_update():
    """
    move every electron, update with single particle updates
    then check that update_displacement gives same tables
    """
    test = pbc.electron(20, 30)
    for i in range(20):
        test.up[i] += np.random.rand(3) - 0.5
        test.update_up(i)
    for i in range(30):
        test.down[i] += np.random.rand(3) - 0.5
        test.update_down(i)
    old_up_table = np.copy(test.up_table)
    old_down_table = np.copy(test.down_table)
    old_up_down_table = np.copy(test.up_down_table)
    test.update_displacement()
    assert np.isclose(old_up_table, test.up_table).all(), \
        'update_up and update_displacement are inconsistent for electron'
    assert np.isclose(old_down_table, test.down_table).all(), \
        'update_down and update_displacement are inconsistent for electron'
    assert np.isclose(old_up_down_table, test.up_down_table).all()
    

def test_proton_update():
    """
    move every proton, update object using update_r
    and check that update_displacement at the end gives same result
    """
    test = pbc.proton(50)
    for i in range(50):
        test.r[i] += np.random.rand(3) - 0.5
        test.update_r(i)
    oldtable = np.copy(test.table)
    test.update_displacement()
    assert np.isclose(oldtable, test.table).all(), \
        'update_r and update_displacement are inconsistent for proton class'

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
