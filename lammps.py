# tools related to lammps
import numpy as np

def read_dump(fname, number_of_samples):
    """
    read dump from lammps

    args:
        fname (string): name of dump file
        number_of_samples (int): number of samples to read from the file;
            can be equal to or less than the number of actual samples
            in the file
    returns:
        tuple of arrays, the first having shape (number_of_samples, N, M)
        where N is the number of atoms
        and M is the number of columns of the dump
        and the second having shape (2, 3) giving the box bounds
    """
    box = []
    data = []
    with open(fname, 'r') as f:
        for i in range(number_of_samples):
            for j in range(3):
                # skip first three lines
                f.readline()
            N = int(f.readline()) # number of atoms
            # this is read for each sample, since it can potentially vary (?)
            box.append(np.loadtxt(f, skiprows=1, max_rows=3))
            data.append(np.loadtxt(f, skiprows=1, max_rows=N))
    return np.array(data), np.array(box)

def read_rdf(fname, number_of_bins, number_of_samples):
    """
    read radial distribution function (rdf) output from lammps

    args:
        fname (string): name of lammps rdf file
        number_of_bins (int): number of bins for rdf,
            must match number of bins that were actually used in lammps
        number_of_samples (int): number of rdf samples,
            can be equal to or less than the number of samples
            that were actually written out
    returns:
        array of shape (number_of_samples, number_of_bins, y),
            where y is the number of columns in each sample;
            see lammps documentation for how many columns one can expect
    """
    data = []
    with open(fname, 'r') as f:
        # skip the first three lines, which is just a header
        for j in range(3):
            f.readline()
        # begin reading samples
        for i in range(number_of_samples):
            data.append(np.loadtxt(f, skiprows=1, max_rows=number_of_bins))
    return np.array(data)
