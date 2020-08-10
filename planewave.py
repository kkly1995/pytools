import numpy as np
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
    lattice_sum = np.sum(np.exp(exponent), axis=-1)
    return 4*np.pi*density*lattice_sum / denominator

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
    exponent = 2j*np.pi*np.einsum('ijx,ax->ija', k, ion_coords)
    lattice_sum = np.sum(np.exp(exponent), axis=-1)
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
    potential = potential.astype(complex)*lattice_sum / supercell.volume
    potential *= 4*np.pi*charge
    return potential
