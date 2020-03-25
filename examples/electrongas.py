import pytools.pbc as pbc
from random import random
from pytools.qmc import SMW
import numpy as np

class wavefunction:
    def __init__(self, r_s, N):
        """
        jellium in a cubic cell, unpolarized
        currently testing just slater determinant,
        which should resemble hartree-fock (?)
        
        r_s is the usual usual length scale
        N is the number of up electrons
        i.e. the total number of electrons is 2N
        and N should be chosen for a closed shell

        everything begins uninitialized
        """
        self.L = 1.612*r_s*((2*N)**(1./3))
        self.electron = pbc.electron(N, N)
        v = self.L*np.eye(3)
        self.supercell = pbc.cell(v[0], v[1], v[2])
        self.slater_up = np.zeros((N, N))
        self.slater_down = np.zeros((N, N))
        self.inverse_up = np.zeros_like(self.slater_up)
        self.inverse_up = np.zeros_like(self.slater_down)
        self.N = N
        self.k = self.supercell.fermi_sea(3)[:,:3]
        
    def update_all(self):
        """
        update all relevant attributes
        given that electron has been updated
        currently just have the slater matrices and their inverses
        """
        #k is in reciprocal coord and r is in crystal coord
        #but since this is cubic you just need a factor of 2pi
        kr_up = np.matmul(self.k[:self.N], self.electron.up.transpose())
        kr_down = np.matmul(self.k[:self.N], self.electron.down.transpose())
        kr_up *= 2*np.pi
        kr_down *= 2*np.pi
        self.slater_up = np.exp(1j*kr_up)
        self.slater_down = np.exp(1j*kr_down)
        #update inverses
        self.inverse_up = np.linalg.inv(self.slater_up)
        self.inverse_down = np.linalg.inv(self.slater_down)

    def update_slater_up(self, i):
        """
        update column of slater matrix
        associated with up electron i
        """
        kr = np.matmul(self.k[:self.N], self.electron.up[i])
        kr *= 2*np.pi
        self.slater_up[:,i] = np.exp(1j*kr)

    def update_slater_down(self, i):
        """
        update column of slater matrix
        associated with down electron i
        """
        kr = np.matmul(self.k[:self.N], self.electron.down[i])
        kr *= 2*np.pi
        self.slater_up[:,i] = np.exp(1j*kr)

    def psi(self):
        """
        for testing
        """
        return np.linalg.det(self.slater_up) * \
                np.linalg.det(self.slater_down)

    def local_kinetic(self):
        """
        for this slater determinant this is trivial
        i just have all this here in anticipation of a jastrow
        in which case this calculation is not trivial
        """
        ksq = np.matmul(self.k[:self.N], self.k[:self.N].transpose())
        #only want diagonal
        diag = np.diagonal(ksq).copy()
        diag *= (2*np.pi/self.L)**2
        ksq = np.zeros((self.N, self.N))
        ksq[np.diag_indices(self.N)] = diag
        #multiply on left
        laplacian_up = -np.matmul(ksq, self.slater_up)
        laplacian_down = -np.matmul(ksq, self.slater_down)
        kinetic_up = np.matmul(self.inverse_up, laplacian_up).trace()
        kinetic_down = np.matmul(self.inverse_down, laplacian_down).trace()
        return -0.5*np.real(kinetic_up + kinetic_down)

    def local_potential(self, kappa):
        """
        kappa is the parameter which controls the charge smearing
        """
        k = np.copy(self.k[1:]) #remove k = 0
        k *= 2*np.pi/self.L
        upupterm = pbc.ewald(self.electron.up_table*self.L, kappa, k,\
                self.L, one_species=True)
        downdownterm = pbc.ewald(self.electron.down_table*self.L, kappa, k,\
                self.L, one_species=True)
        updownterm = pbc.ewald(self.electron.up_down_table*self.L, kappa, k,\
                self.L, one_species=False)
        self_term = 2*self.N*kappa/np.sqrt(np.pi)
        neutralizer = np.pi*(2*self.N)**2
        neutralizer /= 2*(kappa**2)*(self.L**3)
        return upupterm + downdownterm + updownterm - \
                self_term - neutralizer

    def move_up(self, i, scale):
        """
        uniform MC move of up electron i
        each component of the move will be a random number between
        0 and scale
        note that we are using crystal coordinates
        e.g. scale = 0.1 corresponds to a move of ~10% of L
        """
        step = (np.random.rand(3) - 0.5)*scale
        #copy in case of rejection
        oldslater = np.copy(self.slater_up)
        #begin move
        self.electron.up[i] += step
        self.electron.update_up(i)
        self.update_slater_up(i)
        ratio = np.dot(self.inverse_up[i,:], self.slater_up[:,i])
        #decide whether to accept or reject
        probability = ratio**2
        if random() < probability:
            self.inverse_up = SMW(self.inverse_up, \
                    self.slater_up - oldslater,
                    ratio)
            return True
        else:
            #revert
            self.slater_up = np.copy(oldslater)
            self.electron.up[i] -= step
            self.electron.update_up(i)
            return False

    def move_down(self, i, scale):
        """
        uniform MC move of down electron i
        each component of the move will be a random number between
        0 and scale
        note that we are using crystal coordinates
        e.g. scale = 0.1 corresponds to a move of ~10% of L
        """
        step = (np.random.rand(3) - 0.5)*scale
        #copy in case of rejection
        oldslater = np.copy(self.slater_down)
        #begin move
        self.electron.down[i] += step
        self.electron.update_down(i)
        self.update_slater_down(i)
        ratio = np.dot(self.inverse_down[i,:], self.slater_down[:,i])
        #decide whether to accept or reject
        probability = ratio**2
        if random() < probability:
            self.inverse_down = SMW(self.inverse_down, \
                    self.slater_down - oldslater,
                    ratio)
            return True
        else:
            #revert
            self.slater_down = np.copy(oldslater)
            self.electron.down[i] -= step
            self.electron.update_down(i)
            return False

    def move_all(self, scale):
        """
        moves all up and down electrons
        and returns the TOTAL number of acceptances
        (not the average)
        """
        accepted = 0
        for i in range(self.N):
            accepted += self.move_up(i, scale)
            accepted += self.move_down(i, scale)
        #recompute inverse, else SMW errors accumulate
        self.inverse_up = np.linalg.inv(self.slater_up)
        self.inverse_down = np.linalg.inv(self.slater_down)
        return accepted

if __name__=="__main__":
    test = wavefunction(1, 81)
    test.electron.start_random()
    #with open('27.up', 'r') as f:
    #    test.electron.up = np.loadtxt(f)
    #with open('27.down', 'r') as f:
    #    test.electron.down = np.loadtxt(f)
    #test.electron.update_displacement()
    test.update_all()
    """
    #begin measurement
    kappa = 0.34
    stepsize = 0.4
    #warmup
    for i in range(100):
        test.move_all(stepsize)
    #measure
    energy = []
    for i in range(10000):
        test.move_all(stepsize)
        energy.append(test.local_kinetic() + test.local_potential(0.34))
    """
