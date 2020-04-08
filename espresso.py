"""
tools related to inputs / ouputs from Quantum Espresso
"""

import numpy as np

def readmodes(fname, units='cm'):
    """
    read q points, frequencies, and polarizations from a matdyn.modes type file
    where fname is the name of said file

    returns four arrays: array of q vectors, array of frequencies, 
        array of real polarization vectors, array of im polarization vectors

    freq[i, j] gives the jth frequency (in the order listed by the modes file)
        of the ith q point
        by default it reads the number in units of cm-1,
        but changing kwarg units to 'THz' reads the other number
    pol_real[i, j, k] gives the kth polarization vector (real part) of the
        jth frequency of the ith q point
    similarly for pol_im, the imaginary part
    
    NOTE:
    in the file itself the frequencies are numbered starting from 1;
        of course, these arrays are indexed starting from 0
    """
    with open(fname, "r") as f:
        lines = f.readlines()

    q = []
    freq_raw = []
    pol_real = []
    pol_im = []

    for line in lines:
        if len(line) > 1:
            #line is nonempty
            words = line.split()
            if words[0] == 'q':
                q.append([float(words[-3]), float(words[-2]), float(words[-1])])
            if words[0] == 'freq':
                if units == 'THz':
                    freq_raw.append(float(words[-5]))
                else:
                    freq_raw.append(float(words[-2]))
            if words[0] == '(':
                pol_real.append([float(words[1]), float(words[3]), float(words[5])])
                pol_im.append([float(words[2]), float(words[4]), float(words[6])])

    #numbers have been read, but freq and polarizations have to be reshaped
    natoms = len(pol_real) // len(freq_raw)
    q = np.array(q)
    freq_raw = np.array(freq_raw)
    pol_real = np.array(pol_real)
    pol_im = np.array(pol_im)
    freq = np.reshape(freq_raw, (len(q), natoms*3))
    pol_real = np.reshape(pol_real, (len(q), natoms*3, natoms, 3))
    pol_im = np.reshape(pol_im, (len(q), natoms*3, natoms, 3))
    return q, freq, pol_real, pol_im

def read_force(fname):
    """
    read the forces on atoms from an scf output file
    where fname is the name of the file
    """
    with open(fname, "r") as f:
        lines = f.readlines()
    forces = []
    for line in lines:
        words = line.split()
        if ('force' in words) and ('atom' in words):
            forces.append([float(words[-3]),float(words[-2]),float(words[-1])])
    return np.array(forces)

def read_atomic_positions(fname, n_atoms):
    """
    read the atomic positions of an input file whose name is fname
    it assumed that the positions begin after the line beginning with
    'ATOMIC_POSITIONS'
    and that each line, upon applying split(), has only four elements:
    [name, #, #, #]
    
    this will return an array of shape (n_atoms, 3)
    and should look exactly like the numbers under ATOMIC_POSITIONS
    just without the names
    """
    with open(fname, "r") as f:
        lines = f.readlines()
    found = False
    count = 0
    atomic_positions = np.zeros((n_atoms, 3))
    for line in lines:
        if found:
            words = line.split()
            atomic_positions[count,0] = float(words[-3])
            atomic_positions[count,1] = float(words[-2])
            atomic_positions[count,2] = float(words[-1])
            count += 1
        if 'ATOMIC_POSITIONS' in line.split():
            found = True
        if count == n_atoms:
            return atomic_positions
    if count != n_atoms:
        print('too many atoms were specified')
    if found == False:
        print('atomic positions were not found')

def read_energy(fname):
    """
    read energy from pw.x output file fname
    just finds the line that starts with !
    and returns the second to last word
    """
    with open(fname, 'r') as f:
        lines = f.readlines()
    for line in lines:
        words = line.split()
        try:
            if words[0] == "!":
                return float(words[-2])
        except:
            pass
    #if it makes it out of the loop, energy hasnt been found
    print('energy not found')
