import pytools.data as pd
import numpy as np

def test_sort_rows():
    arr = np.array([[3, 1.1], [4, 1.7], [1, 0], [2, -9]])
    ans0 = np.array([[1, 0], [2, -9], [3, 1.1], [4, 1.7]])
    ans1 = np.array([[2, -9], [1, 0], [3, 1.1], [4, 1.7]])
    msg0 = 'failed to sort by column 0'
    msg1 = 'failed to sort by column 1'
    assert np.isclose(pd.sort_rows(arr, 0), ans0).all(), msg0
    assert np.isclose(pd.sort_rows(arr, 1), ans1).all(), msg1

def test_bootstrap_mean_error():
    #test on 1D array
    data = np.random.normal(size=6900)
    expected_err = 1./np.sqrt(6900)
    bootstrap_err = pd.bootstrap_mean_error(data, 2000)
    assert np.isclose(expected_err, bootstrap_err, rtol=0.1)
    #test on 2D array
    data = np.random.normal(size=(6900, 6, 9))
    expected_err = np.full((6, 9), expected_err)
    bootstrap_err = pd.bootstrap_mean_error(data, 2000)
    assert np.allclose(expected_err, bootstrap_err, rtol=0.1)
