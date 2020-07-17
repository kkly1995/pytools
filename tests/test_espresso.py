import numpy as np
import pytools.espresso as pe

def test_read_force():
    force1 = pe.read_force('data/scf.out')
    with open('data/force.scf', 'r') as f:
        force2 = np.loadtxt(f)
    msg = 'read_force() failed on test set'
    assert np.allclose(force1, force2), msg

def test_read_atomic_positions():
    with open('data/atomic_pos.scf', 'r') as f:
        pos1 = np.loadtxt(f)
    pos2 = pe.read_atomic_positions('data/scf.in', len(pos1))
    msg = 'read_atomic_positions() failed on test set'
    assert np.allclose(pos1, pos2)

def test_read_energy():
    energy1 = pe.read_energy('data/scf.out')
    with open('data/energy.scf', 'r') as f:
        energy2 = np.loadtxt(f)
    msg = 'read_energy() failed on test set'
    assert np.isclose(energy1, energy2)
