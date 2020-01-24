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