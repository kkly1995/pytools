import numpy as np
import pytools.qmc as pq

def test_SMW():
    #not sure what the probability of getting a bad mat is
    oldmat = np.random.rand(7,7)
    newmat = np.copy(oldmat)
    #change row 4
    newmat[4] += np.random.rand(7)
    detratio = np.linalg.det(newmat) / np.linalg.det(oldmat)
    oldinverse = np.linalg.inv(oldmat)
    difference = newmat - oldmat
    newinverse = np.linalg.inv(newmat)
    newinverse_estimate = pq.SMW(oldinverse, difference, detratio)
    msg = 'sherman-morrison-woodbury failed on random 7x7'
    assert np.isclose(newinverse, newinverse_estimate).all(), msg

