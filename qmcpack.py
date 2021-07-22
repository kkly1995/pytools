"""
tools related to inputs / outputs from QMCPACK
"""

import numpy as np

def read_energy(fname):
    """
    read energy and error from a qmca output in fname
    assumes only one series
    so only two numbers are returned
    """
    with open(fname, 'r') as f:
        lines = f.readlines()
    for line in lines:
        words = line.split()
        try:
            if words[0] == 'LocalEnergy':
                energy = float(words[2])
                error = float(words[4])
                return energy, error
        except:
            pass
    #if it makes it out the loop, energy was not found
    print('LocalEnergy not found')

def read_force(fname, name='force'):
    """
    read the forces on atoms from a qmca output in fname
    assumes only one series

    name is the same as the name tag in qmcpack input
    when specifying the estimator
    """
    with open(fname, 'r') as f:
        lines = f.readlines()
    force_list = []
    error_list = []
    indices = []
    for line in lines:
        words = line.split()
        try:
            if words[0][:6] == '%s_' % name:
                label = words[0].split('_')
                atom = int(label[-2])
                component = int(label[-1])
                force_list.append(float(words[2]))
                error_list.append(float(words[4]))
                indices.append((atom, component))
        except:
            pass
    #now make array
    N = len(force_list)//3 #number of atoms
    force = np.zeros((N, 3))
    error = np.zeros((N, 3))
    for i in range(3*N):
        force[indices[i]] = force_list[i]
        error[indices[i]] = error_list[i]
    return force, error

def fix_href(fname, hname):
    """
    when using nexus to generate QMC input
    using generate_only = 1,
    href in sposet_builder is missing (for good reason)

    this manually replaces the missing href, assumed to be named
    MISSING.h5

    args:
        fname (string): path/name of QMCPACK input containing incorrect href
        hname (string): path/name of actual href to use
    returns:
        None, will print complaint if href is not found
    """
    with open(fname, 'r') as f:
        lines = f.readlines()
    found = False
    linenum = -1
    for i in range(len(lines)):
        if lines[i].find('href') != -1:
            found = True
            break
    if found:
        lines[i] = lines[i].replace('MISSING.h5', hname)
        with open(fname, 'w') as f:
            f.writelines(lines)
    else:
        print('could not find missing href in %s' % fname)
