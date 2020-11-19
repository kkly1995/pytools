import numpy as np
from numba import jit
"""
collection of functions utilizing plane wave bases
e.g. list of vectors from pbc.cell.kvecs()
"""

def delta_k(basis):
    """
    constructs table k[i,j] = basis[i] - basis[j]
    which is used in all potentials
    """
    N = len(basis)
    table = np.zeros((N, N, 3))
    for i in range(N):
        table[i] = basis[i] - basis
    return table

def bare_nuclear_potential(k, supercell, ion_coords):
    """
    k should be an table such as one returned by delta_k
    and should be in reciprocal coordinates
    supercell is a pbc.cell object
    used to convert k to cartesian
    (both crystal and cartesian are used here,
    which is why the conversion occurs internally)
    and also to access the volume of the cell
    ion_coords are the coordinates of the ions in crystal units
    """
    density = len(ion_coords) / supercell.volume
    denominator = np.einsum('ijx,ijx->ij', k, k)
    #set diagonal to inf to avoid division by zero
    denominator[np.diag_indices(len(k))] = np.inf
    exponent = 2j*np.pi*np.einsum('ijx,ax->ija', k, ion_coords)
    lattice_factor = np.sum(np.exp(exponent), axis=-1)
    return 4*np.pi*density*lattice_factor / denominator

def kinetic(basis, supercell):
    """
    kinetic energy (1/2)k^2
    basis is in reciprocal coordinates
    and converted to cartesian internally

    this is so that externally, basis is always kept in reciprocal units

    returns a diagonal matrix
    where T[i,i] is the kinetic energy of plane wave i
    """
    kcart = np.matmul(basis, supercell.reciprocal)
    ksq = np.einsum('ix,ix->i', kcart, kcart)
    N = len(basis)
    val = np.zeros((N, N))
    val[np.diag_indices(N)] = ksq
    return 0.5*val

def two_electron_term(k, supercell, charge_density):
    """
    two-electron integral for the Fock matrix
    args k, supercell are identical to those in nuclear_potential
    charge_density is the P matrix (Szabo & Ostlund)

    i think it is possible to construct the two electron integrals
    (ij | kl) with numpy for all but the trivially small basis sets
    this would require a ton of memory
    hence this is done with for loops
    """
    N = len(k)
    kcart = np.matmul(k, supercell.reciprocal)
    ksq = np.einsum('ijx,ijx->ij', k, k)
    ksq[np.diag_indices(N)] = np.inf #prevent division by zero
    val = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            for a in range(N):
                for b in range(N):
                    net_momentum = k[j,i] + k[b,a]
                    if np.allclose(net_momentum, 0):
                        val[i,j] += (1./ksq[j,i] - 0.5/ksq[a,i]) *\
                                charge_density[a,b]
    val *= 4*np.pi/supercell.volume
    return val

def ashcroft_empty_core(charge, cutoff, screening_length,\
        k, supercell, ion_coords):
    """
    effective potential credited to ashcroft
    see e.g. Marder (2010) s10.2.1
    has three free parameters:
    effective charge of the ion,
    cutoff inside which the core is empty,
    screening length that is the scale for the exponential
    the args k, supercell, ion_coords are same as in
    bare_nuclear_potential i.e. k is a table of integer differences,
    supercell provides the geometry,
    and ion_coords gives the position of ions in reduced coordinates
    """
    latsum = lattice_sum(k, ion_coords)
    diagonal = screening_length*(cutoff + screening_length)*\
            np.exp(-cutoff / screening_length)
    #begin offdiagonal
    kappa = np.einsum('ijl,lx->ijx', k, supercell.reciprocal)
    #set diagonal to nonsense to avoid division by 0
    kappa[np.diag_indices(len(k))] = np.array([1,0,0])
    kappa_norm = np.linalg.norm(kappa, axis=-1)
    denominator = (screening_length**2) * (kappa_norm**3)
    denominator += kappa_norm
    numerator = kappa_norm*screening_length*np.cos(kappa_norm*cutoff)
    numerator += np.sin(kappa_norm*cutoff)
    numerator *= screening_length*np.exp(-cutoff/screening_length)
    #assemble
    potential = numerator/denominator
    potential[np.diag_indices(len(k))] = diagonal
    potential = potential.astype(complex)*latsum / supercell.volume
    potential *= 4*np.pi*charge
    return potential

@jit(nopython=True)
def lattice_sum(k, r):
    """
    perform the lattice sum
    \sum_{r_i} e^{i k_n \cdot r_i}
    for every k_n in input k

    args:
        k (array): "table" of integer triplets,
            which has three axes, and k[i,j] stores each triplet
            i.e. k.shape[2] == 3 must be true
        r (array): array of coordinates to do the lattice sum over
            has two axes: r[i] gives the coordinates of particle i
            i.e. r.shape[1] == 3 must be true
            should be in fractional coordinates if k is integer triplets
    returns:
        array of the same shape as k without the last axis
            i.e. element (i,j) is the lattice sum for k[i,j]
    """
    N = k.shape[0]
    M = k.shape[1]
    L = r.shape[0]
    val = np.zeros((N, M), dtype=np.complex64)
    for i in range(N):
        for j in range(M):
            for a in range(L):
                kdotr = k[i,j,0]*r[a,0] + k[i,j,1]*r[a,1] + k[i,j,2]*r[a,2]
                val[i,j] += np.exp(2j*np.pi*kdotr)
    return val

def gaussian_potential(params, k, supercell, coords):
    """
    one-body effective potential V(k)
    which is just a gaussian centered at 0
    with 2 parameters: (in order)
        overall factor multiplying the whole gaussian
        width of the gaussian
    
    args:
        params (array-like): a 1D array or list with 2 numbers,
            containing the parameters of the potential
        k (array): table of integer triplets to use for k-vectors
            has 3 axes, the last of which has length 3
            see lattice_sum() for more info
        supercell (pbc.cell type): required to convert coordinates
            and also has the volume of the system
        coords (array): coordinates of the atoms in the cell
            has 2 axes, the second of which has length 3
            ideally should be fractional coordinates,
            since k is assumed to be in integer coordinates
    returns:
        array of the same shape as k but without the last axis
            the (i,j)th element is V(k[i,j])
    """
    lattice_factor = lattice_sum(k, coords) / supercell.volume
    kappa = np.einsum('ijl,lx->ijx', k, supercell.reciprocal)
    #set diagonal to nonsense to avoid division by 0
    kappa[np.diag_indices(len(k))] = np.array([1,0,0])
    kappa = np.linalg.norm(kappa, axis=-1)
    v = np.zeros_like(kappa, dtype=complex)
    v += params[0]*np.exp(-kappa**2 / (2*params[1]**2))
    v[np.diag_indices(len(k))] = 0 # diagonal is irrelevant
    return lattice_factor*v
