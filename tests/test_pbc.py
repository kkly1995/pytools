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
        test.dn[i] += np.random.rand(3) - 0.5
        test.update_down(i)
    old_up_table = np.copy(test.uu_table)
    old_down_table = np.copy(test.dd_table)
    old_up_down_table = np.copy(test.ud_table)
    test.update_displacement()
    assert np.isclose(old_up_table, test.uu_table).all(), \
        'update_up and update_displacement are inconsistent for electron'
    assert np.isclose(old_down_table, test.dd_table).all(), \
        'update_down and update_displacement are inconsistent for electron'
    assert np.isclose(old_up_down_table, test.ud_table).all()
    

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

def test_hydrogen_update():
    """
    move every particle in hydrogen, update with single particle updates
    then check that update_displacement gives same tables
    """
    test = pbc.hydrogen(10, 20, 30)
    for i in range(10):
        test.pn[i] += np.random.rand(3) - 0.5
        test.update_proton(i)
    for i in range(20):
        test.up[i] += np.random.rand(3) - 0.5
        test.update_up(i)
    for i in range(30):
        test.dn[i] += np.random.rand(3) - 0.5
        test.update_down(i)
    old_up_table = np.copy(test.uu_table)
    old_down_table = np.copy(test.dd_table)
    old_up_down_table = np.copy(test.ud_table)
    old_pp_table = np.copy(test.pp_table)
    old_pe_table = np.copy(test.pe_table)
    test.update_displacement()
    assert np.isclose(old_up_table, test.uu_table).all(), \
        'update_up and update_displacement are inconsistent for electron'
    assert np.isclose(old_down_table, test.dd_table).all(), \
        'update_down and update_displacement are inconsistent for electron'
    assert np.isclose(old_up_down_table, test.ud_table).all()
    assert np.isclose(old_pp_table, test.pp_table).all()
    assert np.isclose(old_pe_table, test.pe_table).all()

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
    kvecs = supercell.kvecs(3)[1:,:3]
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
    charge.dn = np.copy(charge.up) + 1/6
    charge.update_displacement()
    #construct k vectors to sum over
    v = L*np.eye(3)
    supercell = pbc.cell(v[0], v[1], v[2])
    k = supercell.kvecs(4)[1:,:3]
    k *= 2*np.pi/L
    kappa = 6/L #convergence is around here
    #begin ewald sums
    vol = supercell.volume
    ppterm = pbc.ewald(charge.uu_table*L, kappa, k, vol,\
            one_species=True)
    mmterm = pbc.ewald(charge.dd_table*L, kappa, k, vol,\
            one_species=True)
    pmterm = pbc.ewald(charge.ud_table*L, kappa, k, vol,\
            one_species=False)
    self_term = 54*kappa/np.sqrt(np.pi)
    energy = ppterm + mmterm - pmterm - self_term
    #compute madelung constant
    r = L*np.min(np.linalg.norm(charge.ud_table, axis=-1))
    madelung = -energy*r/27
    assert isclose(madelung, alpha, abs_tol=0.0001), \
            'ewald failed to compute madelung constant for CsCl'

def test_grad_ewald_lr():
    L = 5*np.random.rand()
    v = L*np.eye(3)
    supercell = pbc.cell(v[0], v[1], v[2])
    volume = supercell.volume
    kvecs = supercell.kvecs(3)[1:,:3]
    kvecs *= 2*np.pi/L
    kappa = 5/L
    r = L*np.random.rand(3)
    #compute derivatives
    f = pbc.ewald_lr(r, kappa, kvecs, volume) #h = 0
    h = 0.00001 #finite difference step
    r[0] += h
    ddx = (pbc.ewald_lr(r, kappa, kvecs, volume) - f)/h
    r[0] -= h
    r[1] += h
    ddy = (pbc.ewald_lr(r, kappa, kvecs, volume) - f)/h
    r[1] -= h
    r[2] += h
    ddz = (pbc.ewald_lr(r, kappa, kvecs, volume) - f)/h
    grad = np.array([ddx, ddy, ddz])
    assert np.isclose(grad, pbc.grad_ewald_lr(r, kappa, kvecs, volume),\
            rtol=1e-2).all()

def test_ewald_sr_prime():
    L = 5
    kappa = 1
    r = np.linspace(1, 2)
    f = pbc.ewald_sr(kappa, r)
    df = np.gradient(f, r[1] - r[0])
    assert np.isclose(pbc.ewald_sr_prime(kappa, r)[1:-1], df[1:-1],\
            rtol = 1e-2).all()

def test_laplacian_ewald_sr():
    kappa = 1
    r = 0.1 + np.random.rand()
    h = 0.0001
    ddr = pbc.ewald_sr(kappa, r+h) - pbc.ewald_sr(kappa, r-h)
    ddr /= 2*h
    ddr2 = pbc.ewald_sr(kappa, r+h) + pbc.ewald_sr(kappa, r-h) -\
            2*pbc.ewald_sr(kappa, r)
    ddr2 /= h**2
    lap = 2*ddr/r + ddr2
    assert isclose(pbc.laplacian_ewald_sr(kappa, r), lap, rel_tol=h)

def test_laplacian_ewald_lr():
    L = 5*np.random.rand()
    v = L*np.eye(3)
    supercell = pbc.cell(v[0], v[1], v[2])
    volume = supercell.volume
    kvecs = supercell.kvecs(3)[1:,:3]
    kvecs *= 2*np.pi/L
    kappa = 5/L
    r = L*np.random.rand(3)
    #compute derivatives
    f = pbc.ewald_lr(r, kappa, kvecs, volume) #h = 0
    h = 0.00001 #finite difference step
    lap = 0
    for i in range(3):
        r[i] += h
        f_fwd = pbc.ewald_lr(r, kappa, kvecs, volume)
        r[i] -= 2*h
        f_bwd = pbc.ewald_lr(r, kappa, kvecs, volume)
        r[i] += h #restore original r
        lap += (f_fwd + f_bwd - 2*f) / h**2
    assert np.isclose(lap, pbc.laplacian_ewald_lr(r, kappa, kvecs, volume),\
            rtol=1e-4)

