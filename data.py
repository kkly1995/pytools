"""
collection of data analysis tools
typically geared towards statistical / monte carlo type data
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import trapz
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
