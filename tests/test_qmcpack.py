import numpy as np
import pytools.qmcpack as pq

def test_read_energy():
    energy, err = pq.read_energy('data/dmc.dat')
    with open('data/energy.dmc', 'r') as f:
        data = np.loadtxt(f)
    assert np.isclose(energy, data[0]), \
            'read_energy() failed to get energy from test data'
    assert np.isclose(err, data[1]), \
            'read_energy() failed to get error from test data'

def test_read_force():
    force1, err1 = pq.read_force('data/dmc.dat')
    with open('data/force.dmc', 'r') as f:
        data = np.loadtxt(f)
    force2 = np.zeros_like(force1)
    err2 = np.zeros_like(err1)
    for point in data:
        i = int(point[0])
        j = int(point[1])
        force2[i,j] = point[2]
        err2[i,j] = point[3]
    assert np.allclose(force1, force2), \
            'read_force() failed to get force from test data'
    assert np.allclose(err1, err2), \
            'read_force() failed to get error from test data'
