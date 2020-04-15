import pytools.pbc as pbc
from random import random
from pytools.qmc import SMW
import numpy as np
import copy
import math

def u_exp(r, F):
    """
    second term in electron gas RPA jastrow
    note that the minus sign here IS included
    and the factor of A outside is NOT
    see e.g. pytools.jastrow.egas
    """
    return -np.exp(-r / F) / r

class wavefunction:
    def __init__(self, r_s, N, N_k, kappa_scale):
        """
        jellium in a cubic cell, unpolarized
        
        r_s is the usual usual length scale
        N is the number of up electrons
        i.e. the total number of electrons is 2N
        and N should be chosen for a closed shell

        N_k (indirectly) sets the number of kvectors to be kept
        and used for ewald sums
        kappa_scale (indirectly) sets the smearing parameter
        for ewald sums: kappa = kappa_scale / L
        should always be tested for convergence

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
        #ewald tables, called u
        self.u_sr_up = np.zeros_like(self.electron.up_table)
        self.u_sr_down = np.zeros_like(self.electron.down_table)
        self.u_sr_up_down = np.zeros_like(self.electron.up_down_table)
        self.u_lr_up = np.zeros_like(self.u_sr_up)
        self.u_lr_down = np.zeros_like(self.u_sr_down)
        self.u_lr_up_down = np.zeros_like(self.u_sr_up_down)
        #parameters and additional tables for jastrow
        n = 2*N/self.supercell.volume #number density, for RPA
        self.A = 1/np.sqrt(4*np.pi*n)
        self.F_uu = np.sqrt(2*self.A)
        self.F_ud = np.sqrt(self.A)
        self.u_exp_up = np.zeros_like(self.electron.up_table)
        self.u_exp_down = np.zeros_like(self.electron.down_table)
        self.u_exp_up_down = np.zeros_like(self.electron.up_down_table)
        self.N = N
        self.k = self.supercell.fermi_sea(N_k)[:,:3]
        self.kappa = kappa_scale / self.L
        
    def update_all(self):
        """
        update all relevant attributes
        given that electron has been updated
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
        #update u
        kvecs = (2*np.pi/self.L)*self.k[1:]
        self.u_lr_up = pbc.ewald_lr(self.L*self.electron.up_table,\
                self.kappa, kvecs, self.supercell.volume)
        self.u_lr_down = pbc.ewald_lr(self.L*self.electron.down_table,\
                self.kappa, kvecs, self.supercell.volume)
        self.u_lr_up_down = pbc.ewald_lr(self.L*self.electron.up_down_table,\
                self.kappa, kvecs, self.supercell.volume)
        r_up = np.linalg.norm(self.electron.up_table, axis=-1)
        r_down = np.linalg.norm(self.electron.down_table, axis=-1)
        r_up_down = np.linalg.norm(self.electron.up_down_table, axis=-1)
        r_up[np.diag_indices(self.N)] = np.inf
        r_down[np.diag_indices(self.N)] = np.inf
        self.u_sr_up = pbc.ewald_sr(self.kappa, r_up*self.L)
        self.u_sr_down = pbc.ewald_sr(self.kappa, r_down*self.L)
        self.u_sr_up_down = pbc.ewald_sr(self.kappa, r_up_down*self.L)
        self.u_exp_up = u_exp(r_up*self.L, self.F_uu)
        self.u_exp_down = u_exp(r_down*self.L, self.F_uu)
        self.u_exp_up_down = u_exp(r_up_down*self.L, self.F_ud)

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
        self.slater_down[:,i] = np.exp(1j*kr)

    def update_u_up(self, i):
        """
        update u tables associated with up electron i
        """
        kvecs = (2*np.pi/self.L)*self.k[1:]
        vol = self.supercell.volume
        lr_up = pbc.ewald_lr(self.electron.up_table[i]*self.L,\
                self.kappa, kvecs, vol)
        lr_up_down = pbc.ewald_lr(self.electron.up_down_table[i]*self.L,\
                self.kappa, kvecs, vol)
        self.u_lr_up[i,:] = np.copy(lr_up)
        #ewald_lr is an even function of r, so no minus sign
        self.u_lr_up[:,i] = np.copy(lr_up)
        self.u_lr_up_down[i,:] = np.copy(lr_up_down)
        #second index is for down electron, so not updated here
        r_uu = np.linalg.norm(self.electron.up_table[i], axis=-1)
        r_ud = np.linalg.norm(self.electron.up_down_table[i], axis=-1)
        r_uu[i] = np.inf
        sr_up = pbc.ewald_sr(self.kappa, r_uu*self.L)
        sr_up_down = pbc.ewald_sr(self.kappa, r_ud*self.L)
        self.u_sr_up[i,:] = np.copy(sr_up)
        self.u_sr_up[:,i] = np.copy(sr_up)
        self.u_sr_up_down[i,:] = np.copy(sr_up_down)
        exp_up = u_exp(r_uu*self.L, self.F_uu)
        exp_up_down = u_exp(r_ud*self.L, self.F_ud)
        self.u_exp_up[i,:] = np.copy(exp_up)
        self.u_exp_up[:,i] = np.copy(exp_up)
        self.u_exp_up_down[i,:] = np.copy(exp_up_down)

    def update_u_down(self, i):
        """
        update u tables associated with down electron i
        """
        kvecs = (2*np.pi/self.L)*self.k[1:]
        vol = self.supercell.volume
        lr_down = pbc.ewald_lr(self.electron.down_table[i]*self.L,\
                self.kappa, kvecs, vol)
        lr_up_down = pbc.ewald_lr(self.electron.up_down_table[:,i]*self.L,\
                self.kappa, kvecs, vol)
        self.u_lr_down[i,:] = np.copy(lr_down)
        #ewald_lr is an even function of r, so no minus sign
        self.u_lr_down[:,i] = np.copy(lr_down)
        self.u_lr_up_down[:,i] = np.copy(lr_up_down)
        #first index is for up electron, not updated here
        r_dd = np.linalg.norm(self.electron.down_table[i], axis=-1)
        r_ud = np.linalg.norm(self.electron.up_down_table[:,i], axis=-1)
        r_dd[i] = np.inf
        sr_down = pbc.ewald_sr(self.kappa, r_dd*self.L)
        sr_up_down = pbc.ewald_sr(self.kappa, r_ud*self.L)
        self.u_sr_down[i,:] = np.copy(sr_down)
        self.u_sr_down[:,i] = np.copy(sr_down)
        self.u_sr_up_down[:,i] = np.copy(sr_up_down)
        exp_down = u_exp(r_dd*self.L, self.F_uu)
        exp_up_down = u_exp(r_ud*self.L, self.F_ud)
        self.u_exp_down[i,:] = np.copy(exp_down)
        self.u_exp_down[:,i] = np.copy(exp_down)
        self.u_exp_up_down[:,i] = np.copy(exp_up_down)

    def psi(self):
        """
        for testing
        """
        jastrow = np.exp(-self.u())
        return jastrow*np.linalg.det(self.slater_up) * \
                np.linalg.det(self.slater_down)

    def u(self):
        """
        return total u in jastrow
        INCLUDING THE FACTOR OF A OUTSIDE
        as well as the constant terms
        (self term, neutralizing background)

        for testing
        """
        uu = self.A*(np.sum(self.u_sr_up) + np.sum(self.u_lr_up) +\
                np.sum(self.u_exp_up))/2
        dd = self.A*(np.sum(self.u_sr_down) + np.sum(self.u_lr_down) +\
                np.sum(self.u_exp_down))/2
        ud = self.A*(np.sum(self.u_sr_up_down) + np.sum(self.u_lr_up_down) +\
                np.sum(self.u_exp_up_down))
        self_term = 2*self.N*self.kappa/np.sqrt(np.pi)
        neutralizer = np.pi*(2*self.N)**2
        neutralizer /= 2*(self.kappa**2)*(self.L**3)
        return uu + dd + ud - self.A*(self_term + neutralizer)

    def laplacian(self):
        #for testing
        r_uu = np.linalg.norm(self.electron.up_table, axis=-1)*self.L
        r_dd = np.linalg.norm(self.electron.down_table, axis=-1)*self.L
        r_ud = np.linalg.norm(self.electron.up_down_table, axis=-1)*self.L
        r_uu[np.diag_indices(self.N)] = np.inf
        r_dd[np.diag_indices(self.N)] = np.inf
        #directions
        r_uu_hat = self.electron.up_table*self.L / r_uu[:,:,np.newaxis]
        r_dd_hat = self.electron.down_table*self.L / r_dd[:,:,np.newaxis]
        r_ud_hat = self.electron.up_down_table*self.L / r_ud[:,:,np.newaxis]
        #for lr
        kvecs = (2*np.pi/self.L)*self.k[1:]
        vol = self.supercell.volume

        #laplacian of sr
        l_u_sr_up = pbc.laplacian_ewald_sr(self.kappa, r_uu)
        l_u_sr_down = pbc.laplacian_ewald_sr(self.kappa, r_dd)
        l_u_sr_up_down = pbc.laplacian_ewald_sr(self.kappa, r_ud)

        #laplacian of lr
        l_u_lr_up = pbc.laplacian_ewald_lr(self.electron.up_table*self.L,\
                self.kappa, kvecs, vol)
        l_u_lr_down = pbc.laplacian_ewald_lr(self.electron.down_table*self.L,\
                self.kappa, kvecs, vol)
        #the weird lr diagonal appears here
        l_u_lr_up[np.diag_indices(self.N)] = 0
        l_u_lr_down[np.diag_indices(self.N)] = 0
        l_u_lr_up_down = pbc.laplacian_ewald_lr(self.electron.up_down_table*self.L,\
                self.kappa, kvecs, vol)

        #laplacian of exp
        l_u_exp_up = self.u_exp_up / self.F_uu**2
        l_u_exp_down = self.u_exp_down / self.F_uu**2
        l_u_exp_up_down = self.u_exp_up_down / self.F_ud**2

        #similar to g_u_up and g_u_down
        #except now 1D array since laplacian is scalar
        l_u_up = self.A*np.sum(l_u_sr_up + l_u_lr_up + l_u_exp_up + \
                l_u_sr_up_down + l_u_lr_up_down + l_u_exp_up_down, axis=1)
        #since theres no minus sign this is less tricky than g_u
        l_u_down = self.A*np.sum(l_u_sr_down + l_u_lr_down + l_u_exp_down + \
                l_u_sr_up_down + l_u_lr_up_down + l_u_exp_up_down, axis=0)
        return l_u_up

    def local_kinetic(self):
        """
        local kinetic energy according to (23.15) in MRC
        """
        k = np.copy(self.k[:self.N])*2*np.pi/self.L
        ksq = np.matmul(k, k.transpose())
        #only want diagonal of ksq
        diag = np.diagonal(ksq).copy()
        ksq = np.zeros((self.N, self.N))
        ksq[np.diag_indices(self.N)] = diag
        g_slater_up = 1j*k[:,np.newaxis,:] * \
                self.slater_up[:,:,np.newaxis]
        g_slater_down = 1j*k[:,np.newaxis,:] * \
                self.slater_down[:,:,np.newaxis]
        #multiply on left
        l_slater_up = -np.matmul(ksq, self.slater_up)
        l_slater_down = -np.matmul(ksq, self.slater_down)

        #begin computing (lots of) derivatives for jastrow
        r_uu = np.linalg.norm(self.electron.up_table, axis=-1)*self.L
        r_dd = np.linalg.norm(self.electron.down_table, axis=-1)*self.L
        r_ud = np.linalg.norm(self.electron.up_down_table, axis=-1)*self.L
        r_uu[np.diag_indices(self.N)] = np.inf
        r_dd[np.diag_indices(self.N)] = np.inf
        #directions
        r_uu_hat = self.electron.up_table*self.L / r_uu[:,:,np.newaxis]
        r_dd_hat = self.electron.down_table*self.L / r_dd[:,:,np.newaxis]
        r_ud_hat = self.electron.up_down_table*self.L / r_ud[:,:,np.newaxis]

        #gradients of u_sr
        g_u_sr_up = pbc.ewald_sr_prime(self.kappa, r_uu)
        g_u_sr_up = g_u_sr_up[:,:,np.newaxis]*r_uu_hat
        g_u_sr_down = pbc.ewald_sr_prime(self.kappa, r_dd)
        g_u_sr_down = g_u_sr_down[:,:,np.newaxis]*r_dd_hat
        g_u_sr_up_down = pbc.ewald_sr_prime(self.kappa, r_ud)
        g_u_sr_up_down = g_u_sr_up_down[:,:,np.newaxis]*r_ud_hat

        #gradients of u_lr
        kvecs = (2*np.pi/self.L)*self.k[1:] #not to be confused with k
        vol = self.supercell.volume
        g_u_lr_up = pbc.grad_ewald_lr(self.electron.up_table*self.L,\
                self.kappa, kvecs, vol)
        g_u_lr_down = pbc.grad_ewald_lr(self.electron.down_table*self.L,\
                self.kappa, kvecs, vol)
        g_u_lr_up_down = pbc.grad_ewald_lr(self.electron.up_down_table*self.L,\
                self.kappa, kvecs, vol)

        #gradients of u_exp
        g_u_exp_up = (1/r_uu) + (1/self.F_uu)
        g_u_exp_up *= -self.u_exp_up
        g_u_exp_up = g_u_exp_up[:,:,np.newaxis]*r_uu_hat
        g_u_exp_down = (1/r_dd) + (1/self.F_uu)
        g_u_exp_down *= -self.u_exp_down
        g_u_exp_down = g_u_exp_down[:,:,np.newaxis]*r_dd_hat
        g_u_exp_up_down = (1/r_ud) + (1/self.F_ud)
        g_u_exp_up_down *= -self.u_exp_up_down
        g_u_exp_up_down = g_u_exp_up_down[:,:,np.newaxis]*r_ud_hat

        #g_u_up has shape (N, 3), where the first index indicates
        #which particle the gradient is w.r.t.
        #similarly for g_u_down
        g_u_up = self.A*(np.sum(g_u_sr_up + g_u_lr_up + g_u_exp_up + \
                g_u_sr_up_down + g_u_lr_up_down + g_u_exp_up_down, axis=1))
        #watch the indices and sign on up_down
        g_u_down = self.A*(np.sum(g_u_sr_down + g_u_lr_down + g_u_exp_down,\
                axis=1) - np.sum(g_u_sr_up_down + g_u_lr_up_down + \
                g_u_exp_up_down, axis=0))

        #laplacian of sr
        l_u_sr_up = pbc.laplacian_ewald_sr(self.kappa, r_uu)
        l_u_sr_down = pbc.laplacian_ewald_sr(self.kappa, r_dd)
        l_u_sr_up_down = pbc.laplacian_ewald_sr(self.kappa, r_ud)

        #laplacian of lr
        l_u_lr_up = pbc.laplacian_ewald_lr(self.electron.up_table*self.L,\
                self.kappa, kvecs, vol)
        l_u_lr_down = pbc.laplacian_ewald_lr(self.electron.down_table*self.L,\
                self.kappa, kvecs, vol)
        #the weird lr diagonal appears here
        l_u_lr_up[np.diag_indices(self.N)] = 0
        l_u_lr_down[np.diag_indices(self.N)] = 0
        l_u_lr_up_down = pbc.laplacian_ewald_lr(self.electron.up_down_table*self.L,\
                self.kappa, kvecs, vol)

        #laplacian of exp
        l_u_exp_up = self.u_exp_up / self.F_uu**2
        l_u_exp_down = self.u_exp_down / self.F_uu**2
        l_u_exp_up_down = self.u_exp_up_down / self.F_ud**2

        #since we only need the sum over all laplacians,
        #these are just single numbers
        l_u_up = self.A*np.sum(l_u_sr_up + l_u_lr_up + l_u_exp_up + \
                l_u_sr_up_down + l_u_lr_up_down + l_u_exp_up_down)
        #since theres no minus sign this is less tricky than g_u
        l_u_down = self.A*np.sum(l_u_sr_down + l_u_lr_down + l_u_exp_down + \
                l_u_sr_up_down + l_u_lr_up_down + l_u_exp_up_down)

        term1 = l_u_up + l_u_down
        term2 = np.trace(np.matmul(g_u_up, g_u_up.transpose())) + \
                np.trace(np.matmul(g_u_down, g_u_down.transpose()))
        term3 = np.matmul(self.inverse_up, l_slater_up).trace() + \
                np.matmul(self.inverse_down, l_slater_down).trace()
        term4 = np.matmul(self.inverse_up,\
                np.einsum('il,kil->ki', g_u_up, g_slater_up)).trace()
        term4 += np.matmul(self.inverse_down,\
                np.einsum('il,kil->ki', g_u_down, g_slater_down)).trace()
        term4 *= 2
        return 0.5*np.real(term1 - term2 - term3 + term4)

    def local_potential(self):
        """
        local potential energy according to ewald sum
        using u tables
        """
        uuterm = (np.sum(self.u_sr_up) + np.sum(self.u_lr_up))/2
        ddterm = (np.sum(self.u_sr_down) + np.sum(self.u_lr_down))/2
        udterm = np.sum(self.u_sr_up_down) + np.sum(self.u_lr_up_down)
        self_term = 2*self.N*self.kappa/np.sqrt(np.pi)
        neutralizer = np.pi*(2*self.N)**2
        neutralizer /= 2*(self.kappa**2)*(self.L**3)
        return uuterm + ddterm + udterm - self_term - neutralizer

    def local_potential_test(self, kappa):
        """
        version of potential that uses pbc.ewald instead of u tables
        kept for testing
        """
        k = np.copy(self.k[1:]) #remove k = 0
        k *= 2*np.pi/self.L
        vol = self.supercell.volume
        upupterm = pbc.ewald(self.electron.up_table*self.L, kappa, k,\
                vol, one_species=True)
        downdownterm = pbc.ewald(self.electron.down_table*self.L, kappa, k,\
                vol, one_species=True)
        updownterm = pbc.ewald(self.electron.up_down_table*self.L, kappa, k,\
                vol, one_species=False)
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
        oldu_sr_up = np.copy(self.u_sr_up)
        oldu_lr_up = np.copy(self.u_lr_up)
        oldu_exp_up = np.copy(self.u_exp_up)
        oldu_sr_up_down = np.copy(self.u_sr_up_down)
        oldu_lr_up_down = np.copy(self.u_lr_up_down)
        oldu_exp_up_down = np.copy(self.u_exp_up_down)
        #begin move
        self.electron.up[i] += step
        self.electron.update_up(i)
        self.update_slater_up(i)
        self.update_u_up(i)
        ratio = np.dot(self.inverse_up[i,:], self.slater_up[:,i])
        #change in log of jastrow
        du = np.sum(self.u_sr_up[i] - oldu_sr_up[i])
        du += np.sum(self.u_lr_up[i] - oldu_lr_up[i])
        du += np.sum(self.u_exp_up[i] - oldu_exp_up[i])
        du += np.sum(self.u_sr_up_down[i] - oldu_sr_up_down[i])
        du += np.sum(self.u_lr_up_down[i] - oldu_lr_up_down[i])
        du += np.sum(self.u_exp_up_down[i] - oldu_exp_up_down[i])
        du *= self.A
        #prevent under/overflow
        if du > 100:
            du = 100
        if du < -100:
            du = -100
        #decide whether to accept or reject
        probability = ratio.real**2 + ratio.imag**2
        probability *= math.exp(-2*du)
        if random() < probability:
            self.inverse_up = SMW(self.inverse_up, \
                    self.slater_up - oldslater,
                    ratio)
            return True
        else:
            #revert
            self.slater_up = np.copy(oldslater)
            self.u_sr_up = np.copy(oldu_sr_up)
            self.u_lr_up = np.copy(oldu_lr_up)
            self.u_exp_up = np.copy(oldu_exp_up)
            self.u_sr_up_down = np.copy(oldu_sr_up_down)
            self.u_lr_up_down = np.copy(oldu_lr_up_down)
            self.u_exp_up_down = np.copy(oldu_exp_up_down)
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
        oldu_sr_down = np.copy(self.u_sr_down)
        oldu_lr_down = np.copy(self.u_lr_down)
        oldu_exp_down = np.copy(self.u_exp_down)
        oldu_sr_up_down = np.copy(self.u_sr_up_down)
        oldu_lr_up_down = np.copy(self.u_lr_up_down)
        oldu_exp_up_down = np.copy(self.u_exp_up_down)
        #begin move
        self.electron.down[i] += step
        self.electron.update_down(i)
        self.update_slater_down(i)
        self.update_u_down(i)
        ratio = np.dot(self.inverse_down[i,:], self.slater_down[:,i])
        #change in log of jastrow
        du = np.sum(self.u_sr_down[i] - oldu_sr_down[i])
        du += np.sum(self.u_lr_down[i] - oldu_lr_down[i])
        du += np.sum(self.u_exp_down[i] - oldu_exp_down[i])
        du += np.sum(self.u_sr_up_down[:,i] - oldu_sr_up_down[:,i])
        du += np.sum(self.u_lr_up_down[:,i] - oldu_lr_up_down[:,i])
        du += np.sum(self.u_exp_up_down[:,i] - oldu_exp_up_down[:,i])
        du *= self.A
        #prevent under/overflow
        if du > 100:
            du = 100
        if du < -100:
            du = -100
        #decide whether to accept or reject
        probability = ratio.real**2 + ratio.imag**2
        probability *= math.exp(-2*du)
        if random() < probability:
            self.inverse_down = SMW(self.inverse_down, \
                    self.slater_down - oldslater,
                    ratio)
            return True
        else:
            #revert
            self.slater_down = np.copy(oldslater)
            self.u_sr_down = np.copy(oldu_sr_down)
            self.u_lr_down = np.copy(oldu_lr_down)
            self.u_exp_down = np.copy(oldu_exp_down)
            self.u_sr_up_down = np.copy(oldu_sr_up_down)
            self.u_lr_up_down = np.copy(oldu_lr_up_down)
            self.u_exp_up_down = np.copy(oldu_exp_up_down)
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

def test_laplacian():
    #test ran on 4/14/20
    test = wavefunction(3, 7, 3, 5)
    test.electron.start_semirandom(0.05)
    #put the electrons in a line
    #test.electron.up[:,0] = np.array([-0.06, -0.04, -0.02, \
    #        0.00, 0.02, 0.04, 0.06])
    #test.electron.down[:,0] = np.array([-0.07, -0.05, -0.03, \
    #        -0.01, 0.01, 0.03, 0.05])
    test.electron.update_displacement()
    test.update_all()

    #check a single finite difference
    h_crys = 0.0001 #finite difference in units of L
    u_mid = test.u()
    estimate = 0
    for i in range(3):
        test.electron.down[4,i] += h_crys
        test.electron.update_displacement()
        test.update_all()
        u_fwd = test.u()
        test.electron.down[4,i] -= 2*h_crys
        test.electron.update_displacement()
        test.update_all()
        u_bwd = test.u()
        estimate += (u_fwd + u_bwd - 2*u_mid) / (h_crys*test.L)**2
        #return electrons back to original spot
        test.electron.down[4,i] += h_crys
        test.electron.update_displacement()
        test.update_all()

def test_grad():
    #test ran on 4/14/20
    test = wavefunction(3, 7, 3, 5)
    test.electron.start_semirandom(0.05)
    #with open('81.up', 'r') as f:
    #    test.electron.up = np.loadtxt(f)
    #with open('81.down', 'r') as f:
    #    test.electron.down = np.loadtxt(f)
    #test.electron.update_displacement()
    test.update_all()

    #check a single finite difference
    h_crys = 0.0001 #finite difference in units of L
    estimate = []
    for i in range(3):
        test.electron.up[4,i] += h_crys
        test.electron.update_displacement()
        test.update_all()
        u_fwd = test.u()
        test.electron.up[4,i] -= 2*h_crys
        test.electron.update_displacement()
        test.update_all()
        u_bwd = test.u()
        estimate.append((u_fwd - u_bwd) / (2*h_crys*test.L))
        #return electrons back to original spot
        test.electron.up[4,i] += h_crys
        test.electron.update_displacement()
        test.update_all()

if __name__=="__main__":
    test = wavefunction(3, 81, 3, 5)
    #test.electron.start_semirandom(0.05)
    with open('81.up', 'r') as f:
        test.electron.up = np.loadtxt(f)
    with open('81.down', 'r') as f:
        test.electron.down = np.loadtxt(f)
    test.electron.update_displacement()
    test.update_all()
