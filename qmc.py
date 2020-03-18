"""
collection of functions specifically for QMC
"""
import numpy as np

def SMW(oldinverse, difference, detratio):
    """
    suppose A and A' are square matrices which differ only by the outer product
    of two vectors (differing by only one row or one column fulfills this
    criteria), and let oldinverse be the inverse of A. Given this outerproduct,
    which is just the difference = A' - A, and the ratio of the determinants
    detratio = det(A') / det(A), this computes the inverse of A'
    a la sherman-morrison-woodbury

    comment: i find that, at least on my current machine, np.linalg.inv
    is competitive with this function, speedwise; it is probably due to
    some aggressive optimization for that function. given that np.linalg.inv
    is also more accurate (i find), this function should not be used
    unless it is found to be faster, which may occur on another machine
    or large enough systems. however, if used, it np.linalg.inv should still be
    used every so often, say after a complete pass, to prevent errors from
    accumulating
    """
    newinverse = -np.matmul(oldinverse, np.matmul(difference, oldinverse))
    newinverse /= detratio
    newinverse += oldinverse
    return newinverse
