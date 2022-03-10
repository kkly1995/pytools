"""
collection of data analysis tools
typically geared towards statistical / monte carlo type data
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import h5py
from sklearn.utils import resample

def autocorrelation(data):
    """
    compute autocorrelation for data
    which is 1D array or similar

    computed via FFT, see e.g. Newman and Barkema s3.3.1
    """
    data_fft = np.fft.fft(data) #pad with zeros
    data_fft[0] = 0 #remove 0 frequency component
    func = np.fft.ifft(np.abs(data_fft)**2)
    func /= func[0] #normalize
    return np.real(func)

def correlation_time(data, cutoff, show=False):
    """
    cutoff must always be specified, since ideally dataset is large
    """
    correlations = autocorrelation(data)[:cutoff]
    correlationtime = 1 + 2*sum(correlations[1:])
    if show:
        plt.plot(correlations, 'r-')
        plt.title('correlation time = ' + str(correlationtime))
        plt.axhline()
        plt.show()
    return correlationtime

def plot_estimator(data, start=0, ylims='var'):
    """
    data is a 1d array, a collection of (hopefully independent)  measurements
    plots a trace of the cumulative mean of data
    as opposed to just plotting data itself
    it will not start plotting until the specified start
    this is different from cutting off the beginning of the data
    since the trace is cumulative
    
    this is a way to guess whether or not we have equilibrium
    """
    cumulative = []
    for i in range(1, len(data) - 1):
        cumulative.append(np.mean(data[0:i]))
    plt.plot(cumulative)
    plt.axhline(cumulative[-1], color='r')
    plt.title('mean = ' + str(cumulative[-1]))
    plt.xlim((start, len(cumulative)))
    if ylims=='var':
        plt.ylim((cumulative[-1] - np.var(data), cumulative[-1] + np.var(data)))
    elif ylims=='std':
        plt.ylim((cumulative[-1] - np.std(data), cumulative[-1] + np.std(data)))
    elif ylims=='err':
        #size of 2 errorbars with no correlation time
        #this is more for longer calculations
        scale = np.sqrt(len(data))/2
        plt.ylim((cumulative[-1] - np.std(data)/scale, cumulative[-1] + np.std(data)/scale))
    else:
        #autoset limits
        pass
    plt.show()
    
def pair_correlation(dists, volume, bins='auto', range=None):
    """
    pair correlation function given pair distances dist
    assuming a 3D, isotropic system with specified volume
    dist is assumed to be a 1D array

    bins and range are the same args as in np.histogram()
    """
    hist = np.histogram(dists, bins='auto', range=range)
    dr = hist[1][1] - hist[1][0]
    r = (hist[1] + dr/2)[:-1] #centers of bins
    shell_volume = (r + dr/2)**3 - (r - dr/2)**3
    shell_volume *= 4*np.pi/3
    gr = hist[0] / shell_volume
    normalization = volume / len(dists)
    return normalization*gr, r

def structure_factor(k, r):
    """
    3D structure factor
    WARNING: can be very memory intensive

    k: array of k vectors, shape (#vectors, 3)
    r: array of coordinates, shape (#samples, #particles, 3)
    returns: array of shape (#vectors,)
        element i corresponds to structure factor
        evaluated at k[i]
    """
    kdotr = np.einsum('ij,slj->sil', k, r)
    addend = np.exp(-1j*kdotr)
    addend_minus = np.exp(1j*kdotr)
    rho_k = np.sum(addend, axis=-1)
    rho_minus_k = np.sum(addend_minus, axis=-1)
    N = r.shape[1] # number of particles
    return np.mean(rho_k*rho_minus_k, axis=0) / N

def structure_factor_sample(k, r):
    """
    3D structure factor for a single sample
    so it's less memory intensive

    k: array of k vectors, shape (#vectors, 3)
    r: array of coordinates, shape (#particles, 3)
    returns: array of shape (#vectors,)
        element i corresponds to structure factor
        evaluated at k[i]
    """
    kdotr = np.einsum('ij,nj->in', k, r)
    addend = np.exp(-1j*kdotr)
    addend_minus = np.exp(1j*kdotr)
    rho_k = np.sum(addend, axis=-1)
    rho_minus_k = np.sum(addend_minus, axis=-1)
    return rho_k * rho_minus_k / len(r)

def structure_factor_binary(k, r1, r2):
    """
    similar to structure_factor_sample() but assumes two species

    k: array of k vectors, shape (#vectors, 3)
    r1: array of coordinates for species 1, shape (#particles of type 1, 3)
    r2: array of coordinates for species 2, shape (#particles of type 2, 3)
    returns: array of shape (#vectors,)
        element i corresponds to the structure factor
        evaluated at k[i]
    """
    kdotr1 = np.einsum('ij,nj->in', k, r1)
    kdotr2 = np.einsum('ij,nj->in', k, r2)
    addend1 = np.exp(-1j*kdotr1)
    addend2 = np.exp(1j*kdotr2)
    rho_k = np.sum(addend1, axis=-1)
    rho_minus_k = np.sum(addend2, axis=-1)
    return rho_k *rho_minus_k / (len(r1)*len(r2)) # so that max is 1?

def sort_rows(arr, column):
    """
    sorts the rows of an array according to column
    e.g. consider the data

    3 1.1
    4 1.7
    1 0
    2 -9

    sorting by column 0 would yield

    1 0
    2 -9
    3 1.1
    4 1.7

    and sorting by column 1 would yield

    2 -9
    1 0
    3 1.1
    4 1.7

    this is a one-liner that i found on stackoverflow long ago
    but i cannot find it again
    sorry to whoever you are
    """
    return arr[arr[:,column].argsort()]

def bootstrap_mean_error(data, num_bootstraps):
    """
    given a dataset, this estimates the error
    for the mean of the data via bootstrap
    resamples the data num_bootstraps times
    not recommended for small datasets,
    see e.g. Newman and Barkema ch3
    also can be memory intensive for large datasets

    works on means of arrays as well
    so long as the first index enumerates different data points
    i.e. data[0] is the first array, data[1] is the second, etc
    sklearn.utils.resample appears to correctly resample along this axis
    this will result in an array of errors
    """
    means = []
    for i in range(num_bootstraps):
        means.append(np.mean(resample(data), axis=0))
    return np.std(means, axis=0)

def cube_to_hdf5(cubename, hdfname):
    """
    converts Gaussian cube file to hdf5
    format of the cube file can be found on
    http://paulbourke.net/dataformats/cube/

    args:
        cubename (str): name of cube file to convert
        hdfname (str): name of hdf5 file to write to
    returns:
        nothing
    """
    with open(cubename, 'r') as f:
        lines = f.readlines()

    # third line has number of atoms, as well as the origin
    line = lines[2].split()
    number_of_atoms = int(line[0])
    origin = [float(line[1]), float(line[2]), float(line[3])]

    # next three lines give number of voxels along each axis
    # and corresponding axis vectors
    line = lines[3].split()
    N1 = int(line[0])
    v1 = [float(line[1]), float(line[2]), float(line[3])]
    line = lines[4].split()
    N2 = int(line[0])
    v2 = [float(line[1]), float(line[2]), float(line[3])]
    line = lines[5].split()
    N3 = int(line[0])
    v3 = [float(line[1]), float(line[2]), float(line[3])]

    # next section gives types, charges and positions
    positions = np.zeros((number_of_atoms, 3))
    types = np.zeros(number_of_atoms, dtype=int)
    charges = np.zeros(number_of_atoms)
    for i in range(number_of_atoms):
        line = lines[6+i].split()
        types[i] = int(line[0])
        charges[i] = float(line[1])
        positions[i,0] = float(line[2])
        positions[i,1] = float(line[3])
        positions[i,2] = float(line[4])

    # begin reading the grid, assuming 6 numbers per line
    grid = []
    line_number = 6 + number_of_atoms # starts here
    for line in lines[line_number:]:
        for word in line.split():
            grid.append(float(word))
    grid = np.array(grid)
    grid = grid.reshape((N1, N2, N3))

    # write
    with h5py.File(hdfname, 'w') as f:
        f["/number_of_atoms"] = number_of_atoms
        f["/origin"] = origin
        f["/grid/N1"] = N1
        f["/grid/N2"] = N2
        f["/grid/N3"] = N3
        f["/grid/v1"] = v1
        f["/grid/v2"] = v2
        f["/grid/v3"] = v3
        f["/grid/values"] = grid
        f["/system/types"] = types
        f["/system/charges"] = charges
        f["/system/positions"] = positions

def isotropic_average(x, y, decimals=8):
    """
    args:
        x (array): has shape (N, D), where N is the number of inputs
        and D is the dimension of the inputs
        y (array): has shape (N,), value of function at corresponding x
        decimals (int): when x is float, some machines will separate
        two magnitudes that appear to be the same. this can be handled by
        rounding all the inputs, and this specifies the number of decimal
        places for rounding
    returns:
        array of shape (n, 3), where n is the number of unique magnitudes
        of x, the first column are the magnitudes, and the second column
        are the averages of y over the magnitudes, and the third column
        counts the number of vectors per magnitude
    """
    v, indices, counts = np.unique(\
            np.linalg.norm(x, axis=1).round(decimals=decimals),\
            return_inverse=True, return_counts=True)
    f = []
    for i in range(len(v)):
        f.append(np.mean(y[indices==i]))
    # put into one array
    a = np.zeros((len(v), 3))
    a[:,0] = v
    a[:,1] = f
    a[:,2] = counts
    return a

def velocity_autocorrelation(v):
    """
    args:
        v (array): has shape (samples, particles, 3)
        i.e. v[3,2,1] is the y-component of the velocity
        of particle 2, in sample 3

    returns:
        list  of length T = len(v), the velocity autocorrelation
        as a function of t, the time between samples
    """
    total_time = v.shape[0]
    N = v.shape[1]
    vv = np.einsum('tij,yij->ty', v, v) / N
    # average over times vv(t, t') = vv(t - t')
    avg = []
    for dt in range(total_time):
        avg.append(0)
        for t in range(total_time - dt):
            avg[dt] += vv[t, t+dt]
        avg[dt] /= total_time - dt
    return avg

def fitting_scatterplot(predicted, actual, color='g', plot_error=False):
    """
    convenience function
    to produce a empty-circled scatterplot of predicted vs actual data
    as well as the line y=x for reference

    if plot_error is True, then plot (predicted - actual)
    vs actual instead

    args:
        predicted (1D array): predicted values
        actual (1D array): actual values
        color (str): matplotlib color
        plot_error (bool): whether or not to plot the error

    returns:
        nothing
    """
    if plot_error:
        error = predicted - actual
        plt.scatter(actual, error, facecolors='none', edgecolors=color)
        plt.grid()
    else:
        plt.scatter(actual, predicted, facecolors='none', edgecolors=color)
        x = np.linspace(np.min(actual), np.max(actual))
        plt.plot(x, x)
