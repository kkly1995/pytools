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
        self.slater_dn = np.zeros((N, N))
        self.inverse_up = np.zeros_like(self.slater_up)
        self.inverse_dn = np.zeros_like(self.slater_dn)
        #ewald tables, called u
        self.u_sr_uu = np.zeros_like(self.electron.uu_table)
        self.u_sr_dd = np.zeros_like(self.electron.dd_table)
        self.u_sr_ud = np.zeros_like(self.electron.ud_table)
        self.u_lr_uu = np.zeros_like(self.u_sr_uu)
        self.u_lr_dd = np.zeros_like(self.u_sr_dd)
        self.u_lr_ud = np.zeros_like(self.u_sr_ud)
        #parameters and additional tables for jastrow
        n = 2*N/self.supercell.volume #number density, for RPA
        self.A = 1/np.sqrt(4*np.pi*n)
        self.F_uu = np.sqrt(2*self.A)
        self.F_ud = np.sqrt(self.A)
        self.u_exp_uu = np.zeros_like(self.electron.uu_table)
        self.u_exp_dd = np.zeros_like(self.electron.dd_table)
        self.u_exp_ud = np.zeros_like(self.electron.ud_table)
        self.N = N
        self.k = self.supercell.kvecs(N_k)[:,:3]
        self.kappa = kappa_scale / self.L
        
    def update_all(self):
        """
        update all relevant attributes
        given that electron has been updated
        """
        #k is in reciprocal coord and r is in crystal coord
        #but since this is cubic you just need a factor of 2pi
        kr_up = np.matmul(self.k[:self.N], self.electron.up.transpose())
        kr_dn = np.matmul(self.k[:self.N], self.electron.dn.transpose())
        kr_up *= 2*np.pi
        kr_dn *= 2*np.pi
        self.slater_up = np.exp(1j*kr_up)
        self.slater_dn = np.exp(1j*kr_dn)
        #update inverses
        self.inverse_up = np.linalg.inv(self.slater_up)
        self.inverse_dn = np.linalg.inv(self.slater_dn)
        #update u
        kvecs = (2*np.pi/self.L)*self.k[1:]
        self.u_lr_uu = pbc.ewald_lr(self.L*self.electron.uu_table,\
                self.kappa, kvecs, self.supercell.volume)
        self.u_lr_dd = pbc.ewald_lr(self.L*self.electron.dd_table,\
                self.kappa, kvecs, self.supercell.volume)
        self.u_lr_ud = pbc.ewald_lr(self.L*self.electron.ud_table,\
                self.kappa, kvecs, self.supercell.volume)
        r_uu = np.linalg.norm(self.electron.uu_table, axis=-1)
        r_dd = np.linalg.norm(self.electron.dd_table, axis=-1)
        r_ud = np.linalg.norm(self.electron.ud_table, axis=-1)
        r_uu[np.diag_indices(self.N)] = np.inf
        r_dd[np.diag_indices(self.N)] = np.inf
        self.u_sr_uu = pbc.ewald_sr(self.kappa, r_uu*self.L)
        self.u_sr_dd = pbc.ewald_sr(self.kappa, r_dd*self.L)
        self.u_sr_ud = pbc.ewald_sr(self.kappa, r_ud*self.L)
        self.u_exp_uu = u_exp(r_uu*self.L, self.F_uu)
        self.u_exp_dd = u_exp(r_dd*self.L, self.F_uu)
        self.u_exp_ud = u_exp(r_ud*self.L, self.F_ud)

    def update_slater_up(self, i):
        """
        update column of slater matrix
        associated with up electron i
        """
        kr = np.matmul(self.k[:self.N], self.electron.up[i])
        kr *= 2*np.pi
        self.slater_up[:,i] = np.exp(1j*kr)

    def update_slater_dn(self, i):
        """
        update column of slater matrix
        associated with down electron i
        """
        kr = np.matmul(self.k[:self.N], self.electron.dn[i])
        kr *= 2*np.pi
        self.slater_dn[:,i] = np.exp(1j*kr)

    def update_u_up(self, i):
        """
        update u tables associated with up electron i
        """
        kvecs = (2*np.pi/self.L)*self.k[1:]
        vol = self.supercell.volume
        lr_uu = pbc.ewald_lr(self.electron.uu_table[i]*self.L,\
                self.kappa, kvecs, vol)
        lr_ud = pbc.ewald_lr(self.electron.ud_table[i]*self.L,\
                self.kappa, kvecs, vol)
        self.u_lr_uu[i,:] = np.copy(lr_uu)
        #ewald_lr is an even function of r, so no minus sign
        self.u_lr_uu[:,i] = np.copy(lr_uu)
        self.u_lr_ud[i,:] = np.copy(lr_ud)
        #second index is for down electron, so not updated here
        r_uu = np.linalg.norm(self.electron.uu_table[i], axis=-1)
        r_ud = np.linalg.norm(self.electron.ud_table[i], axis=-1)
        r_uu[i] = np.inf
        sr_uu = pbc.ewald_sr(self.kappa, r_uu*self.L)
        sr_ud = pbc.ewald_sr(self.kappa, r_ud*self.L)
        self.u_sr_uu[i,:] = np.copy(sr_uu)
        self.u_sr_uu[:,i] = np.copy(sr_uu)
        self.u_sr_ud[i,:] = np.copy(sr_ud)
        exp_uu = u_exp(r_uu*self.L, self.F_uu)
        exp_ud = u_exp(r_ud*self.L, self.F_ud)
        self.u_exp_uu[i,:] = np.copy(exp_uu)
        self.u_exp_uu[:,i] = np.copy(exp_uu)
        self.u_exp_ud[i,:] = np.copy(exp_ud)

    def update_u_dn(self, i):
        """
        update u tables associated with down electron i
        """
        kvecs = (2*np.pi/self.L)*self.k[1:]
        vol = self.supercell.volume
        lr_dd = pbc.ewald_lr(self.electron.dd_table[i]*self.L,\
                self.kappa, kvecs, vol)
        lr_ud = pbc.ewald_lr(self.electron.ud_table[:,i]*self.L,\
                self.kappa, kvecs, vol)
        self.u_lr_dd[i,:] = np.copy(lr_dd)
        #ewald_lr is an even function of r, so no minus sign
        self.u_lr_dd[:,i] = np.copy(lr_dd)
        self.u_lr_ud[:,i] = np.copy(lr_ud)
        #first index is for up electron, not updated here
        r_dd = np.linalg.norm(self.electron.dd_table[i], axis=-1)
        r_ud = np.linalg.norm(self.electron.ud_table[:,i], axis=-1)
        r_dd[i] = np.inf
        sr_dd = pbc.ewald_sr(self.kappa, r_dd*self.L)
        sr_ud = pbc.ewald_sr(self.kappa, r_ud*self.L)
        self.u_sr_dd[i,:] = np.copy(sr_dd)
        self.u_sr_dd[:,i] = np.copy(sr_dd)
        self.u_sr_ud[:,i] = np.copy(sr_ud)
        exp_dd = u_exp(r_dd*self.L, self.F_uu)
        exp_ud = u_exp(r_ud*self.L, self.F_ud)
        self.u_exp_dd[i,:] = np.copy(exp_dd)
        self.u_exp_dd[:,i] = np.copy(exp_dd)
        self.u_exp_ud[:,i] = np.copy(exp_ud)

    def psi(self):
        """
        for testing
        """
        jastrow = np.exp(-self.u())
        return jastrow*np.linalg.det(self.slater_up) * \
                np.linalg.det(self.slater_dn)

    def logpsi(self):
        #for testing, just have determinant for now
        slaterproduct = np.linalg.det(self.slater_up) * \
                np.linalg.det(self.slater_dn)
        return np.log(np.abs(slaterproduct))

    def u(self):
        """
        return total u in jastrow
        INCLUDING THE FACTOR OF A OUTSIDE
        as well as the constant terms
        (self term, neutralizing background)

        for testing
        """
        uu = self.A*(np.sum(self.u_sr_uu) + np.sum(self.u_lr_uu) +\
                np.sum(self.u_exp_uu))/2
        dd = self.A*(np.sum(self.u_sr_dd) + np.sum(self.u_lr_dd) +\
                np.sum(self.u_exp_dd))/2
        ud = self.A*(np.sum(self.u_sr_ud) + np.sum(self.u_lr_ud) +\
                np.sum(self.u_exp_ud))
        self_term = 2*self.N*self.kappa/np.sqrt(np.pi)
        neutralizer = np.pi*(2*self.N)**2
        neutralizer /= 2*(self.kappa**2)*(self.L**3)
        return uu + dd + ud - self.A*(self_term + neutralizer)

    def alt_kinetic(self):
        k = np.copy(self.k[:self.N])*2*np.pi/self.L
        ksq = np.matmul(k, k.transpose())
        #only want diagonal of ksq
        diag = np.diagonal(ksq).copy()
        ksq = np.zeros((self.N, self.N))
        ksq[np.diag_indices(self.N)] = diag
        g_slater_up = 1j*k[:,np.newaxis,:] * \
                self.slater_up[:,:,np.newaxis]
        g_slater_dn = 1j*k[:,np.newaxis,:] * \
                self.slater_dn[:,:,np.newaxis]
        #multiply on left
        l_slater_up = -np.matmul(ksq, self.slater_up)
        l_slater_dn = -np.matmul(ksq, self.slater_dn)
        g_logdet_up = np.einsum('ij,jil->il', self.inverse_up, g_slater_up)
        g_logdet_dn = np.einsum('ij,jil->il', self.inverse_dn, \
                g_slater_dn)
        l_det_over_det_up = np.matmul(self.inverse_up, l_slater_up)
        l_det_over_det_dn = np.matmul(self.inverse_dn, l_slater_dn)
        l_logdet_up = l_det_over_det_up - \
                np.matmul(g_logdet_up, g_logdet_up.transpose())
        l_logdet_dn = l_det_over_det_dn - \
                np.matmul(g_logdet_dn, g_logdet_dn.transpose())

        r_uu = np.linalg.norm(self.electron.uu_table, axis=-1)*self.L
        r_dd = np.linalg.norm(self.electron.dd_table, axis=-1)*self.L
        r_ud = np.linalg.norm(self.electron.ud_table, axis=-1)*self.L
        r_uu[np.diag_indices(self.N)] = np.inf
        r_dd[np.diag_indices(self.N)] = np.inf

        #laplacian of sr
        l_u_sr_uu = pbc.laplacian_ewald_sr(self.kappa, r_uu)
        l_u_sr_dd = pbc.laplacian_ewald_sr(self.kappa, r_dd)
        l_u_sr_ud = pbc.laplacian_ewald_sr(self.kappa, r_ud)

        kvecs = (2*np.pi/self.L)*self.k[1:] #not to be confused with k
        vol = self.supercell.volume

        #laplacian of lr
        l_u_lr_uu = pbc.laplacian_ewald_lr(self.electron.uu_table*self.L,\
                self.kappa, kvecs, vol)
        l_u_lr_dd = pbc.laplacian_ewald_lr(self.electron.dd_table*self.L,\
                self.kappa, kvecs, vol)
        #the weird lr diagonal appears here
        l_u_lr_uu[np.diag_indices(self.N)] = 0
        l_u_lr_dd[np.diag_indices(self.N)] = 0
        l_u_lr_ud = pbc.laplacian_ewald_lr(self.electron.ud_table*self.L,\
                self.kappa, kvecs, vol)

        #laplacian of exp
        l_u_exp_uu = self.u_exp_uu / self.F_uu**2
        l_u_exp_dd = self.u_exp_dd / self.F_uu**2
        l_u_exp_ud = self.u_exp_ud / self.F_ud**2

        #since we only need the sum over all laplacians,
        #these are just single numbers
        l_u_up = self.A*np.sum(l_u_sr_uu + l_u_lr_uu + l_u_exp_uu + \
                l_u_sr_ud + l_u_lr_ud + l_u_exp_ud)
        #since theres no minus sign this is less tricky than g_u
        l_u_dn = self.A*np.sum(l_u_sr_dd + l_u_lr_dd + l_u_exp_dd + \
                l_u_sr_ud + l_u_lr_ud + l_u_exp_ud)
        return 0.25*np.real(l_logdet_up.trace() + l_logdet_dn.trace() -\
                l_u_up - l_u_dn)

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
        g_slater_dn = 1j*k[:,np.newaxis,:] * \
                self.slater_dn[:,:,np.newaxis]
        #multiply on left
        l_slater_up = -np.matmul(ksq, self.slater_up)
        l_slater_dn = -np.matmul(ksq, self.slater_dn)

        #begin computing (lots of) derivatives for jastrow
        r_uu = np.linalg.norm(self.electron.uu_table, axis=-1)*self.L
        r_dd = np.linalg.norm(self.electron.dd_table, axis=-1)*self.L
        r_ud = np.linalg.norm(self.electron.ud_table, axis=-1)*self.L
        r_uu[np.diag_indices(self.N)] = np.inf
        r_dd[np.diag_indices(self.N)] = np.inf
        #directions
        r_uu_hat = self.electron.uu_table*self.L / r_uu[:,:,np.newaxis]
        r_dd_hat = self.electron.dd_table*self.L / r_dd[:,:,np.newaxis]
        r_ud_hat = self.electron.ud_table*self.L / r_ud[:,:,np.newaxis]

        #gradients of u_sr
        g_u_sr_uu = pbc.ewald_sr_prime(self.kappa, r_uu)
        g_u_sr_uu = g_u_sr_uu[:,:,np.newaxis]*r_uu_hat
        g_u_sr_dd = pbc.ewald_sr_prime(self.kappa, r_dd)
        g_u_sr_dd = g_u_sr_dd[:,:,np.newaxis]*r_dd_hat
        g_u_sr_ud = pbc.ewald_sr_prime(self.kappa, r_ud)
        g_u_sr_ud = g_u_sr_ud[:,:,np.newaxis]*r_ud_hat

        #gradients of u_lr
        kvecs = (2*np.pi/self.L)*self.k[1:] #not to be confused with k
        vol = self.supercell.volume
        g_u_lr_uu = pbc.grad_ewald_lr(self.electron.uu_table*self.L,\
                self.kappa, kvecs, vol)
        g_u_lr_dd = pbc.grad_ewald_lr(self.electron.dd_table*self.L,\
                self.kappa, kvecs, vol)
        g_u_lr_ud = pbc.grad_ewald_lr(self.electron.ud_table*self.L,\
                self.kappa, kvecs, vol)

        #gradients of u_exp
        g_u_exp_uu = (1/r_uu) + (1/self.F_uu)
        g_u_exp_uu *= -self.u_exp_uu
        g_u_exp_uu = g_u_exp_uu[:,:,np.newaxis]*r_uu_hat
        g_u_exp_dd = (1/r_dd) + (1/self.F_uu)
        g_u_exp_dd *= -self.u_exp_dd
        g_u_exp_dd = g_u_exp_dd[:,:,np.newaxis]*r_dd_hat
        g_u_exp_ud = (1/r_ud) + (1/self.F_ud)
        g_u_exp_ud *= -self.u_exp_ud
        g_u_exp_ud = g_u_exp_ud[:,:,np.newaxis]*r_ud_hat

        #g_u_up has shape (N, 3), where the first index indicates
        #which particle the gradient is w.r.t.
        #similarly for g_u_dn
        g_u_up = self.A*(np.sum(g_u_sr_uu + g_u_lr_uu + g_u_exp_uu + \
                g_u_sr_ud + g_u_lr_ud + g_u_exp_ud, axis=1))
        #watch the indices and sign on ud
        g_u_dn = self.A*(np.sum(g_u_sr_dd + g_u_lr_dd + g_u_exp_dd,\
                axis=1) - np.sum(g_u_sr_ud + g_u_lr_ud + \
                g_u_exp_ud, axis=0))

        #laplacian of sr
        l_u_sr_uu = pbc.laplacian_ewald_sr(self.kappa, r_uu)
        l_u_sr_dd = pbc.laplacian_ewald_sr(self.kappa, r_dd)
        l_u_sr_ud = pbc.laplacian_ewald_sr(self.kappa, r_ud)

        #laplacian of lr
        l_u_lr_uu = pbc.laplacian_ewald_lr(self.electron.uu_table*self.L,\
                self.kappa, kvecs, vol)
        l_u_lr_dd = pbc.laplacian_ewald_lr(self.electron.dd_table*self.L,\
                self.kappa, kvecs, vol)
        #the weird lr diagonal appears here
        l_u_lr_uu[np.diag_indices(self.N)] = 0
        l_u_lr_dd[np.diag_indices(self.N)] = 0
        l_u_lr_ud = pbc.laplacian_ewald_lr(self.electron.ud_table*self.L,\
                self.kappa, kvecs, vol)

        #laplacian of exp
        l_u_exp_uu = self.u_exp_uu / self.F_uu**2
        l_u_exp_dd = self.u_exp_dd / self.F_uu**2
        l_u_exp_ud = self.u_exp_ud / self.F_ud**2

        #since we only need the sum over all laplacians,
        #these are just single numbers
        l_u_up = self.A*np.sum(l_u_sr_uu + l_u_lr_uu + l_u_exp_uu + \
                l_u_sr_ud + l_u_lr_ud + l_u_exp_ud)
        #since theres no minus sign this is less tricky than g_u
        l_u_dn = self.A*np.sum(l_u_sr_dd + l_u_lr_dd + l_u_exp_dd + \
                l_u_sr_ud + l_u_lr_ud + l_u_exp_ud)

        term1 = l_u_up + l_u_dn
        term2 = np.trace(np.matmul(g_u_up, g_u_up.transpose())) + \
                np.trace(np.matmul(g_u_dn, g_u_dn.transpose()))
        term3 = np.matmul(self.inverse_up, l_slater_up).trace() + \
                np.matmul(self.inverse_dn, l_slater_dn).trace()
        term4 = np.matmul(self.inverse_up,\
                np.einsum('il,kil->ki', g_u_up, g_slater_up)).trace()
        term4 += np.matmul(self.inverse_dn,\
                np.einsum('il,kil->ki', g_u_dn, g_slater_dn)).trace()
        term4 *= 2
        return 0.5*np.real(term1 - term2 - term3 + term4)

    def local_potential(self):
        """
        local potential energy according to ewald sum
        using u tables
        """
        uuterm = (np.sum(self.u_sr_uu) + np.sum(self.u_lr_uu))/2
        ddterm = (np.sum(self.u_sr_dd) + np.sum(self.u_lr_dd))/2
        udterm = np.sum(self.u_sr_ud) + np.sum(self.u_lr_ud)
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
        uuterm = pbc.ewald(self.electron.uu_table*self.L, kappa, k,\
                vol, one_species=True)
        ddterm = pbc.ewald(self.electron.dd_table*self.L, kappa, k,\
                vol, one_species=True)
        udterm = pbc.ewald(self.electron.ud_table*self.L, kappa, k,\
                vol, one_species=False)
        self_term = 2*self.N*kappa/np.sqrt(np.pi)
        neutralizer = np.pi*(2*self.N)**2
        neutralizer /= 2*(kappa**2)*(self.L**3)
        return uuterm + ddterm + udterm - \
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
        oldu_sr_uu = np.copy(self.u_sr_uu)
        oldu_lr_uu = np.copy(self.u_lr_uu)
        oldu_exp_uu = np.copy(self.u_exp_uu)
        oldu_sr_ud = np.copy(self.u_sr_ud)
        oldu_lr_ud = np.copy(self.u_lr_ud)
        oldu_exp_ud = np.copy(self.u_exp_ud)
        #begin move
        self.electron.up[i] += step
        self.electron.update_up(i)
        self.update_slater_up(i)
        self.update_u_up(i)
        ratio = np.dot(self.inverse_up[i,:], self.slater_up[:,i])
        #change in log of jastrow
        du = np.sum(self.u_sr_uu[i] - oldu_sr_uu[i])
        du += np.sum(self.u_lr_uu[i] - oldu_lr_uu[i])
        du += np.sum(self.u_exp_uu[i] - oldu_exp_uu[i])
        du += np.sum(self.u_sr_ud[i] - oldu_sr_ud[i])
        du += np.sum(self.u_lr_ud[i] - oldu_lr_ud[i])
        du += np.sum(self.u_exp_ud[i] - oldu_exp_ud[i])
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
            self.u_sr_uu = np.copy(oldu_sr_uu)
            self.u_lr_uu = np.copy(oldu_lr_uu)
            self.u_exp_uu = np.copy(oldu_exp_uu)
            self.u_sr_ud = np.copy(oldu_sr_ud)
            self.u_lr_ud = np.copy(oldu_lr_ud)
            self.u_exp_ud = np.copy(oldu_exp_ud)
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
        oldslater = np.copy(self.slater_dn)
        oldu_sr_dd = np.copy(self.u_sr_dd)
        oldu_lr_dd = np.copy(self.u_lr_dd)
        oldu_exp_dd = np.copy(self.u_exp_dd)
        oldu_sr_ud = np.copy(self.u_sr_ud)
        oldu_lr_ud = np.copy(self.u_lr_ud)
        oldu_exp_ud = np.copy(self.u_exp_ud)
        #begin move
        self.electron.dn[i] += step
        self.electron.update_down(i)
        self.update_slater_dn(i)
        self.update_u_dn(i)
        ratio = np.dot(self.inverse_dn[i,:], self.slater_dn[:,i])
        #change in log of jastrow
        du = np.sum(self.u_sr_dd[i] - oldu_sr_dd[i])
        du += np.sum(self.u_lr_dd[i] - oldu_lr_dd[i])
        du += np.sum(self.u_exp_dd[i] - oldu_exp_dd[i])
        du += np.sum(self.u_sr_ud[:,i] - oldu_sr_ud[:,i])
        du += np.sum(self.u_lr_ud[:,i] - oldu_lr_ud[:,i])
        du += np.sum(self.u_exp_ud[:,i] - oldu_exp_ud[:,i])
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
            self.inverse_dn = SMW(self.inverse_dn, \
                    self.slater_dn - oldslater,
                    ratio)
            return True
        else:
            #revert
            self.slater_dn = np.copy(oldslater)
            self.u_sr_dd = np.copy(oldu_sr_dd)
            self.u_lr_dd = np.copy(oldu_lr_dd)
            self.u_exp_dd = np.copy(oldu_exp_dd)
            self.u_sr_ud = np.copy(oldu_sr_ud)
            self.u_lr_ud = np.copy(oldu_lr_ud)
            self.u_exp_ud = np.copy(oldu_exp_ud)
            self.electron.dn[i] -= step
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
        self.inverse_dn = np.linalg.inv(self.slater_dn)
        return accepted

def test_laplacian():
    #test ran on 4/14/20
    test = wavefunction(3, 7, 3, 5)
    test.electron.start_semirandom(0.05)
    test.electron.update_displacement()
    test.update_all()

    #check a single finite difference
    h_crys = 0.0001 #finite difference in units of L
    u_mid = test.u()
    estimate = 0
    for i in range(3):
        test.electron.dn[4,i] += h_crys
        test.electron.update_displacement()
        test.update_all()
        u_fwd = test.u()
        test.electron.dn[4,i] -= 2*h_crys
        test.electron.update_displacement()
        test.update_all()
        u_bwd = test.u()
        estimate += (u_fwd + u_bwd - 2*u_mid) / (h_crys*test.L)**2
        #return electrons back to original spot
        test.electron.dn[4,i] += h_crys
        test.electron.update_displacement()
        test.update_all()

def test_grad():
    #test ran on 4/14/20
    test = wavefunction(3, 7, 3, 5)
    test.electron.start_semirandom(0.05)
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
    test = wavefunction(4, 27, 3, 5)
    #test.electron.start_semirandom(0.05)
    with open('27.up', 'r') as f:
        test.electron.up = np.loadtxt(f)
    with open('27.down', 'r') as f:
        test.electron.dn = np.loadtxt(f)
    test.electron.update_displacement()
    test.update_all()

#    #check a single finite difference
#    h_crys = 0.0001 #finite difference in units of L
#    u_mid = test.logpsi()
#    estimate = 0
#    for i in range(3):
#        test.electron.up[0,i] += h_crys
#        test.electron.update_displacement()
#        test.update_all()
#        u_fwd = test.logpsi()
#        test.electron.up[0,i] -= 2*h_crys
#        test.electron.update_displacement()
#        test.update_all()
#        u_bwd = test.logpsi()
#        estimate += (u_fwd + u_bwd - 2*u_mid) / (h_crys*test.L)**2
#        #return electrons back to original spot
#        test.electron.up[0,i] += h_crys
#        test.electron.update_displacement()
#        test.update_all()

