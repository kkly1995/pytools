"""
collection of functions and objects for simulations
in periodic boundary conditions
"""
import numpy as np
import scipy.special as sp
from pytools.data import sort_rows

def minimum_image(r):
    """
    returns the minimum image of r
    ideally: r is in crystal coordinates,
    and this reduces every component to a number between
    -0.5 and 0.5
    """
    return r - np.round(r)

def ewald_sr(kappa, r):
    """
    shortrange part of ewald sum for 1/r potential

    not summed over, so this is suitable for tabulation
    
    r is a scalar or array of scalars
    """
    return sp.erfc(kappa*r) / r

def ewald_sr_prime(kappa, r):
    """
    first derivative of ewald_sr wrt r
    """
    first_term = 2*kappa*np.exp(-(kappa*r)**2)
    first_term /= np.sqrt(np.pi)*r
    second_term = sp.erfc(kappa*r) / r**2
    return -(first_term + second_term)

def laplacian_ewald_sr(kappa, r):
    """
    laplacian of ewald_sr wrt r
    """
    return 4*(kappa**3)*np.exp(-(kappa*r)**2) / np.sqrt(np.pi)

def ewald_lr(r, kappa, kvecs, volume):
    """
    longrange part of ewald sum for 1/r potential

    not summed over, so this is suitable for tabulation

    unlike in ewald_sr, r is an array of vectors
    i.e. r.shape[-1] must be 3
    where this last axis accesses the components of the vector
    """
    ksquared = np.matmul(kvecs, kvecs.transpose())
    ksquared = np.diagonal(ksquared)
    kr = np.matmul(r, kvecs.transpose())
    longrange = np.matmul(np.cos(kr),\
            np.exp(-ksquared / (4*kappa**2)) / ksquared)
    longrange *= 4*np.pi/volume
    return longrange

def grad_ewald_lr(r, kappa, kvecs, volume):
    """
    gradient of ewald_lr wrt r
    note the difference between this and ewald_sr_prime:
    since ewald_sr depends only on the magnitude of r,
    it makes more sense in that case to compute the magnitude of r
    externally, and then provide the direction afterwards
    since many other functions will also require that magnitude
    and so one avoids computing it multiple times

    here, the direction is less trivial
    and is essentially a weighted sum over kvecs

    note that if r = r_i - r_j,
    this computes the gradient wrt r_i
    and does not have the correct sign if
    the gradient wrt r_j is desired
    """
    ksquared = np.matmul(kvecs, kvecs.transpose())
    ksquared = np.diagonal(ksquared)
    kr = np.matmul(r, kvecs.transpose())
    #have to get a little tricky here
    exppart = np.exp(-ksquared / (4*kappa**2)) / ksquared
    exppart = exppart*kvecs.transpose() #now includes direction of kvecs
    exppart = exppart.transpose() #ensure the second axis has the components
    longrange = -np.matmul(np.sin(kr), exppart)
    longrange *= 4*np.pi/volume
    return longrange

def laplacian_ewald_lr(r, kappa, kvecs, volume):
    """
    computes laplacian of ewald_lr
    not as sophisticated as grad_ewald_lr since this,
    just like ewald_lr, returns a scalar
    for every vector r
    
    in fact the expression is nearly identical to ewald_lr
    but with a minus sign outside
    and no factor of k^2 in denominator
    """
    ksquared = np.matmul(kvecs, kvecs.transpose())
    ksquared = np.diagonal(ksquared)
    kr = np.matmul(r, kvecs.transpose())
    longrange = np.matmul(np.cos(kr),\
            np.exp(-ksquared / (4*kappa**2)))
    longrange *= -4*np.pi/volume
    return longrange

def ewald(displacement, kappa, kvecs, volume, \
        one_species=False):
    """
    perform ewald sum according to the first two terms of eq 6.4
    in allen and tildesley (2017), with q_i*q_j = 1

    NOTE: the self term (third term in aforementioned equation)
    must be subtracted off manually
    the reason is that, in practice, one has multiple displacement tables
    requiring this function to be called separately each time;
    since the self term really only requires the number of charges
    i think it is more reliable to do it externally
    same goes for any neutralizing background

    displacement and one_species are
    same args as in coulomb_potential

    kappa is the convergence parameter
    also written as G in natoli and ceperley (1995)

    kvecs is list of kvectors over which to perform the long range sum
    it is assumed that: k = 0 is excluded
    and the list is symmetric i.e. if k in the list, then -k is too
    so that this potential is real (replace exp with cos)
    KVECS SHOULD BE IN CARTESIAN COORDINATES
    and must have shape (N,3), where first axis enumerates the vectors
    and second axis accesses a vector's components

    volume is the size of the supercell, must have units
    consistent with kvecs and displacement table
    """
    #perform real space sum in cell
    r = np.linalg.norm(displacement, axis=-1)
    if one_species:
        r = r[np.triu_indices_from(r, k=1)]
    shortrange = np.sum(sp.erfc(kappa*r) / r)
    #begin long range term
    longrange = np.sum(ewald_lr(displacement, kappa, kvecs, volume))
    if one_species:
        longrange /= 2
    return shortrange + longrange

class electron:
    """
    class for electrons
    or any particle with two types, e.g. spin up and spin down
    the main purpose is to keep track of coordinates
    and relative displacements
    IN CRYSTAL COORDINATES
    """
    def __init__(self, N_up, N_down):
        """
        N_up and N_down set number of up and down electrons

        up contains crystal coordinates of up electrons,
        down contains crystal coordinates for down electrons,
        up_table contains relative displacements of up electrons
        IN MINIMUM IMAGE CONVENTION
        i.e. up_table[a,b] is the minimum image of up[a] - up[b]
        and similarly for down_table, up_down_table
        """
        self.up = np.zeros((N_up, 3))
        self.down = np.zeros((N_down, 3))
        self.up_table = np.zeros((N_up, N_up, 3))
        self.down_table = np.zeros((N_down, N_down, 3))
        self.up_down_table = np.zeros((N_up, N_down, 3))
        self.N_up = N_up
        self.N_down = N_down

    def update_displacement(self):
        """
        update all tables
        """
        for i in range(self.N_up):
            self.up_table[i,:,:] = minimum_image(\
                    self.up[i] - self.up)
            self.up_down_table[i,:,:] = minimum_image(\
                    self.up[i] - self.down)
        for i in range(self.N_down):
            self.down_table[i,:,:] = minimum_image(\
                    self.down[i] - self.down)

    def update_up(self, i):
        """
        update any rows/columns in up_table
        and up_down_table associated with up electron i
        also move it back into cell if outside
        """
        self.up[i] = minimum_image(self.up[i])
        up_up = self.up[i] - self.up
        up_down = self.up[i] - self.down
        self.up_table[i,:,:] = minimum_image(up_up)
        self.up_table[:,i,:] = -minimum_image(up_up)
        self.up_down_table[i,:,:] = minimum_image(up_down)

    def update_down(self, i):
        """
        update row/column in down_table
        and up_down_table associated with down electron i
        also move it back into cell if outside
        """
        self.down[i] = minimum_image(self.down[i])
        down_down = self.down[i] - self.down
        up_down = self.up - self.down[i]
        self.down_table[i,:,:] = minimum_image(down_down)
        self.down_table[:,i,:] = -minimum_image(down_down)
        self.up_down_table[:,i,:] = minimum_image(up_down)

    def start_random(self):
        self.up = minimum_image(np.random.rand(self.N_up, 3))
        self.down = minimum_image(np.random.rand(self.N_down, 3))
        self.update_displacement()

    def nearest_neighbors(self):
        """
        returns the distance between the closest pair of particles
        made this mostly just for start_semirandom
        """
        up_r = np.linalg.norm(self.up_table, axis=-1)
        up_min = np.min(up_r[np.triu_indices(self.N_up, k=1)])
        down_r = np.linalg.norm(self.down_table, axis=-1)
        down_min = np.min(down_r[np.triu_indices(self.N_down, k=1)])
        updown_r = np.linalg.norm(self.up_down_table, axis=-1)
        updown_min = np.min(updown_r)
        return min([up_min, down_min, updown_min])

    def start_semirandom(self, tol):
        """
        keeps doing start_random() until the no two particles are
        closer than tol

        returns number of random attempts made

        recommendation: start with small tol
        and slowly increase until it starts taking too long
        """
        self.start_random()
        count = 1
        while self.nearest_neighbors() < tol:
            self.start_random()
            count += 1
        return count

class proton:
    """
    class for protons
    or just a particle set with only one species
    (no spin considered)
    the main purpose is to keep track of coordinates
    and relative displacements
    IN CRYSTAL COORDINATES
    """
    def __init__(self, N):
        """
        N sets the number of particles

        r[i] gives the location of particle i
        and table[i,j] gives the r[i] - r[j]
        IN MINIMUM IMAGE CONVENTION
        """
        self.r = np.zeros((N, 3))
        self.table = np.zeros((N, N, 3))
        self.N = N

    def update_displacement(self):
        """
        update all tables
        """
        for i in range(self.N):
            self.table[i,:,:] = minimum_image(self.r[i] - self.r)

    def update_r(self, i):
        """
        update any rows/columns in table
        associated with up proton i
        also move it back into cell if outside
        """
        self.r[i] = minimum_image(self.r[i])
        newrow = minimum_image(self.r[i] - self.r)
        self.table[i,:,:] = np.copy(newrow)
        self.table[:,i,:] = -np.copy(newrow)

    def start_random(self):
        self.r = minimum_image(np.random.rand(self.N, 3))
        self.update_displacement()

    def nearest_neighbors(self):
        """
        returns the distance between the closest pair of particles
        made this mostly just for start_semirandom
        """
        r = np.linalg.norm(self.up_table, axis=-1)
        return np.min(r[np.triu_indices(self.N_up, k=1)])

    def start_semirandom(self, tol):
        """
        keeps doing start_random() until the no two particles are
        closer than tol

        returns number of random attempts made

        recommendation: start with small tol
        and slowly increase until it starts taking too long
        """
        self.start_random()
        count = 1
        while self.nearest_neighbors() < tol:
            self.start_random()
            count += 1
        return count

class cell:
    """
    simulation cell, in PBC, of shape given by three lattice vectors
    """
    def __init__(self, v1, v2, v3):
        #v1, v2, v3 are lattice vectors
        self.cell_params = np.zeros((3,3)) #notation follows QE
        self.cell_params[0] = np.copy(v1)
        self.cell_params[1] = np.copy(v2)
        self.cell_params[2] = np.copy(v3)
        self.cell_params_inverse = np.linalg.inv(self.cell_params)
        #for coordinate transformations
        self.volume = np.dot(v1, np.cross(v2, v3))
        self.reciprocal = np.cross(np.roll(self.cell_params, \
                -1, axis=0), np.roll(self.cell_params, \
                -2, axis=0))
        self.reciprocal *= 2*np.pi / self.volume

    def crystal_to_cart(self, r):
        """
        takes an array r, which is written in crystal coordinates
        and transforms it to cartesian coordinates

        the array is assumed to have its components in its last axis,
        i.e. its shape is (...,3)
        """
        return np.matmul(r, self.cell_params)

    def cart_to_crystal(self, r):
        """
        similar to crystal_to_cart() but does the opposite
        so r is in cartesian coordinates
        and this returns it in crystal coordinates
        """
        return np.matmul(r, self.cell_params_inverse)

    def minimum_image(self, r):
        """
        minimum image of r
        WHICH MUST BE IN CRYSTAL COORDINATES
        """
        return r - np.round(r)

    def fermi_sea(self, N):
        """
        generates all integers triplets [i, j, k]
        such that i, j, k are between -N and N
        which are supposed to serve as wavevectors
        in reciprocal coordinates
        and hence are commensurate with this cell

        returns an array of shape (2N+1, 4)
        the extra column is the norm of the wavevector
        where the norm is computed in cartesian coordinates
        
        the array will be sorted by magnitude of wavevector
        """
        sea = []
        for i in range(-N, N+1):
            for j in range(-N, N+1):
                for k in range(-N, N+1):
                    sea.append([i, j, k, 0])
        sea = np.array(sea, dtype=float)
        sea_cart = np.matmul(sea[:,:-1], self.reciprocal)
        sea[:,-1] = np.linalg.norm(sea_cart, axis=1)
        return sort_rows(sea, -1)

    def closed_shell(self, sea, k_fermi):
        """
        take a fermi sea, of the kind returned by self.fermi_sea,
        cut off at specified k_fermi
        and remove half the wavevectors in the following way:
        if wavevector k is in the list, do not include -k,

        make sure that the sea is large enough to begin with
        """
        n = sum(sea[:,-1] < k_fermi)
        full_shell = np.copy(sea[:n])
        half_shell = []
        for k in full_shell[:,:-1]:
            included = False
            for i in range(len(half_shell)):
                #check k - (-k)
                if np.linalg.norm(k + half_shell[i]) < 0.9:
                    included = True #wavevector already in half_shell
            if not included:
                half_shell.append(k)
        return np.array(half_shell)
