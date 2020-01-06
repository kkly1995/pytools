import numpy as np
import math

def second_deriv(func, x, i, k, h, val=None):
    """
    takes second derivative of func
    which is a function ONLY of the configuration x
    CHECK IF X IS FLOAT
    this is a derivative with respect to one particle coordinate
    so the particle and coordinate has to be specified
    by i and k respectively
    for 3D problems k is 0, 1, or 2
    i should be anywhere from 0 to # of particles minus 1
    h is the finite size
    
    NEW: since all the functions that call this use it multiple times,
    thus computing func(x) everytime, the option val allows
    a computation of func(x) outside to be fed in
    (may 5, 2019) for agp-free this results in no gain
    which i currently dont understand (????)
    """
    xph = np.copy(x)
    xmh = np.copy(x)
    xph[i,k] += h
    xmh[i,k] -= h
    s = func(xph) + func(xmh)
    if type(val) == float:
        #use val as func(x)
        s -= 2*val
    else:
        s -= 2*func(x)
    s /= h*h
    return s

def total_laplacian(func, x, h):
    """
    same arguments as second_deriv
    but will sum over all particles and coordinates
    """
    n_particles = len(x)
    n_dimensions = len(x[0])
    val = func(x)
    s = 0
    for i in range(0, n_particles):
        for j in range(0, n_dimensions):
            s += second_deriv(func, x, i, j, h, val)
    return s

def gradient(func, x, h):
    """
    computes all particle gradients according to func
    using forward difference
    x must be float

    ideally func returns one number (like a wavefunction)
    in which case this will output something with the same shape as x
    however, it does work if func returns an array,
    in which case the output can be more sophisticated
    """
    n_particles = len(x)
    n_dimensions = len(x[0])
    gradients = []
    for particle in range(0, n_particles):
        gradient = []
        for coordinate in range(0, n_dimensions):
            #the derivatives must be done one at a time
            #since only one coordinate can be moved
            xph = np.copy(x) #reset coordinates
            xph[particle, coordinate] += h
            gradient.append(func(xph) - func(x))
        gradients.append(gradient)
    gradients = np.array(gradients) / h
    return gradients

def laplacian_all(func, x, h):
    """
    same args as total_laplacian and gradient_all
    differs from total laplacian in that
    this returns a laplacian for each particle
    ie returns a list with length len(x)
    total_laplacian is equal to the sum of this list
    """
    n_particles = len(x)
    n_dimensions = len(x[0])
    laplacians = []
    for i in range(0, n_particles):
        laplacian = 0 #reset
        for j in range(0, n_dimensions):
            laplacian += second_deriv(func, x, i, j, h)
        laplacians.append(laplacian)
    return laplacians

def displacement(r1, r2, size):
    """
    computes displacement r2 - r1 (mind the sign)
    in a cube with side length size
    in periodic boundary conditions
    according to the nearest image convention
    
    r1 and r2 should be 1d arrays of any length
    so long as their lengths are the same
    """
    r = r1 - r2
    for i in range(len(r)):
        r[i] -= round(r[i] / size)*size
    return r

def transition_probability(start, finish, drift, timestep):
    """
    for a gaussian distribution with standard deviation timestep
    and an extra bias given by drift,
    this gives the relative probability of starting at start
    and ending at finish
    
    start, finish, and drft are assumed to be vectors of len 3
    """
    val = -timestep*drift
    val += finish - start
    return math.exp(-np.dot(val, val)/(2*timestep));

def half_fermi_sea(k_fermi):
    """
    generates a fermi sea with 'radius' k_fermi
    WHERE THE LENGTH OF THE SYSTEM IS 2*PI
    ie must rescale these for actual use
    
    the 'half' refers to the fact that this only produces half the sea
    this should be equivalent to:
        generating the entire fermi sea,
        then going through the entire set and for every k, remove -k 
        from the set
    """
    N = int(math.ceil(k_fermi))
    sea = []
    for kx in range(-N, N+1):
        for ky in range(-N, N+1):
            for kz in range(-N, N+1):
                if kx**2 + ky**2 + kz**2 < 0.1:
                    return sea
                if kx**2 + ky**2 + kz**2 < k_fermi**2:
                    #stay inside sphere of radius k_fermi
                    sea.append([kx, ky, kz])
    return sea

def angle(v1, v2):
    """
    returns the angle between vectors v1 and v2
    in radians
    """
    val = np.dot(v1, v2)
    val /= np.linalg.norm(v1)*np.linalg.norm(v2)
    return math.acos(val)

def latticeVectorInfo(v1, v2, v3):
    """
    takes three lattice vectors v1, v2, v3,
    and prints their norms and relative angles
    e.g. for fcc primitive lattice vectors
    the norms would all be the same and the angles would all be 60 degrees
    and for bcc primitive lattice vectors two of the angles would be
    70.53, and the last angle would be 109.47 degrees
    """
    a1 = np.array(v1)
    a2 = np.array(v2)
    a3 = np.array(v3)
    a = np.linalg.norm(a1)
    b = np.linalg.norm(a2)
    c = np.linalg.norm(a3)
    alpha = math.acos(np.dot(a2, a3)/(b*c))
    beta = math.acos(np.dot(a1, a3)/(a*c))
    gamma = math.acos(np.dot(a1, a2)/(a*b))
    print('a = ' + str(a))
    print('b = ' + str(b))
    print('c = ' + str(c))
    print('alpha = ' + str(math.degrees(alpha)))
    print('beta = ' + str(math.degrees(beta)))
    print('gamma = ' + str(math.degrees(gamma)))
