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

def test_ewald_lr():
    """
    test ewald_lr, which performs the sum over k with matmul
    against a manual sum over k
    """
    #make supercell
    L = 5*np.random.rand()
    v = L*np.eye(3)
    supercell = pbc.cell(v[0], v[1], v[2])
    volume = supercell.volume
    kvecs = supercell.fermi_sea(3)[1:,:3]
    kvecs *= 2*np.pi/L
    kappa = 5/L
    r = L*np.random.rand(17, 3)
    u = np.zeros(17)
    for n in range(17):
        for k in kvecs:
            ksq = np.dot(k, k)
            kr = np.dot(k, r[n])
            u[n] += np.cos(kr)*np.exp(-ksq / (4*kappa**2)) / ksq
    u *= 4*np.pi/volume
    assert np.isclose(pbc.ewald_lr(r, kappa, kvecs, volume), u).all()

def test_ewald_madelung():
    """
    test ewald sum to see if it can get madelung const for CsCl
    using a 3x3x3 supercell
    the charges are created with the electron class,
    where up and down refer to opposite charges
    """
    alpha = 1.7627 #desired answer
    L = 4.12*3 #4.12 is the lattice const for CsCl
    charge = pbc.electron(27, 27)
    #make structure
    count = 0
    for x in range(3):
        for y in range(3):
            for z in range(3):
                charge.up[count] = [x/3, y/3, z/3]
                count += 1
    charge.down = np.copy(charge.up) + 1/6
    charge.update_displacement()
    #construct k vectors to sum over
    v = L*np.eye(3)
    supercell = pbc.cell(v[0], v[1], v[2])
    k = supercell.fermi_sea(4)[1:,:3]
    k *= 2*np.pi/L
    kappa = 6/L #convergence is around here
    #begin ewald sums
    vol = supercell.volume
    ppterm = pbc.ewald(charge.up_table*L, kappa, k, vol,\
            one_species=True)
    mmterm = pbc.ewald(charge.down_table*L, kappa, k, vol,\
            one_species=True)
    pmterm = pbc.ewald(charge.up_down_table*L, kappa, k, vol,\
            one_species=False)
    self_term = 54*kappa/np.sqrt(np.pi)
    energy = ppterm + mmterm - pmterm - self_term
    #compute madelung constant
    r = L*np.min(np.linalg.norm(charge.up_down_table, axis=-1))
    madelung = -energy*r/27
    assert isclose(madelung, alpha, abs_tol=0.0001), \
            'ewald failed to compute madelung constant for CsCl'
