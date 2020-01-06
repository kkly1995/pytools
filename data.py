"""
collection of data analysis tools
typically geared towards statistical / monte carlo type data
"""

import numpy as np
import matplotlib.pyplot as plt
import math

def autocorrelation(dataset, offset):
    #data set is a 1D list
    mean = np.mean(dataset)
    var = np.var(dataset)
    autocorr = 0
    for i in range(offset, len(dataset)):
        term = (dataset[i - offset] - mean)*(dataset[i] - mean)
        term /= var
        autocorr += term
    return autocorr / len(dataset)

def correlation_time(dataset, cutoff, show=False):
    """
    cutoff must always be specified, since ideally dataset is large
    """
    correlations = []
    times = np.arange(1, cutoff)
    for time in times:
        correlations.append(autocorrelation(dataset, time))
    correlationtime = 1 + 2*sum(correlations)
    if show:
        plt.plot(times, np.array(correlations), 'r-')
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
    N = len(y)
    y /= x**2
    y /= 4*math.pi*dx*N
    return x, y
