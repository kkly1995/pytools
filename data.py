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
    correlationtime = 1 + 2*sum(correlations)
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
    
def makegofr(hist):
    """
    hist is a tuple returned by numpy.histogram
    where hist[0] gives the number in each bin
    and hist[1] gives the locations of the bin edges
    this converts the histogram to a normalized g(r) assuming
    a 3D system, no weights, same bin sizes, etc
    """
    y = hist[0].astype(float) #hist values are output as int
    x = np.copy(hist[1][:-1]) #dump the last value
    dx = x[1] - x[0]
    x += dx/2 #shift so that x now gives the middle of each bin
    #normalize
    y /= trapz(y, x)
    y /= 4*np.pi*x**2
    return x, y

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
