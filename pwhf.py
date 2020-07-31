import numpy as np
"""
collection of tools to do hartree-fock
using plane-wave basis
e.g. list of vectors from pbc.cell.kvecs()
"""

def delta_k(basis):
    """
    constructs table k[i,j] = basis[i] - basis[j]
    which is used all potentials
    """
    N = len(basis)
    table = np.zeros((N, N, 3))
    for i in range(N):
        table[i] = basis[i] - basis
    return table

def nuclear_potential(k, supercell, ion_coords):
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
    lattice_sum = np.sum(np.exp(exponent), axis=-1)
    return 4*np.pi*density*lattice_sum / denominator

def kinetic_term(basis, supercell):
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
