# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 16:17:26 2020

@author: Kevin
"""
import numpy as np
import pytools.pbc as pbc
import pytools.jastrow as jas
import math
from random import random
from copy import deepcopy
from pytools.qmc import log_transition_ratio

def potential(r):
    epsilon = 10.22 #kelvin
    sigma = 2.556 #angstrom
    return 4*epsilon*((sigma / r)**12 - (sigma / r)**6)

class wavefunction:
    def __init__(self, N, a1, a2):
        """
        this will be at fixed density, so the number of atoms N
        completely specifies the physical system
        
        a1, a2 are the parameters of the wavefunction
        """
        density = 0.022 #atoms per angstroms cubed
        self.L = (N/density)**(1/3) #length of cubic cell in angstroms
        self.He = pbc.proton(N) #N spinless particles
        self.u = np.zeros((N, N)) #-log(f), where f defined in eq 5
        self.g_u = np.zeros((N, N, 3)) #gradient of u
        self.a1 = a1
        self.a2 = a2
        
    def update_all(self):
        """
        update all entries in u
        the trick is to avoid the diagonal of table, since it's zero
        """
        r = self.L*np.linalg.norm(self.He.table, axis=-1)
        r[np.diag_indices(self.He.N)] = np.inf #prevent division by zero
        self.u = jas.mcmillan(r, self.a1, self.a2)
        #gradient
        rhat = self.L*self.He.table / r[:,:,np.newaxis]
        uprime = -(self.a2 / r)*self.u
        self.g_u = uprime[:,:,np.newaxis]*rhat
       
    def update_single(self, i):
        """
        update entries in u that are related to atom i
        """
        r = self.L*np.linalg.norm(self.He.table[i], axis=-1)
        r[i] = np.inf
        u_row = jas.mcmillan(r, self.a1, self.a2)
        self.u[i,:] = np.copy(u_row)
        self.u[:,i] = np.copy(u_row)
        #gradient
        rhat = self.L*self.He.table[i] / r[:,np.newaxis]
        uprime = -(self.a2 / r)*u_row
        g_u_row = uprime[:,np.newaxis]*rhat
        self.g_u[i] = np.copy(g_u_row)
        self.g_u[:,i] = np.copy(-g_u_row)
        
    def logpsi(self):
        """
        returns log of wavefunction
        """
        return -np.sum(np.triu(self.u, k=1))
        
    def g_logpsi(self):
        """
        returns ALL gradients of logpsi
        has shape (N, 3)
        [i, j] is the gradient w.r.t.
        component j of atom i

        to instead just have one gradient, see drift()
        """
        return -np.sum(self.g_u, axis=1)

    def d_logpsi_d_a1(self):
        """
        derivative of logpsi w.r.t parameter a1
        somewhat trivial
        """
        return (self.a2/self.a1)*self.logpsi()

    def d_logpsi_d_a2(self):
        """
        derivative of logpsi w.r.t. parameter a2
        less trivial

        the diagonal of r is set to a1 rather than inf
        so that the diagonal of 1/r is 1, rather than 0
        so we can take the log of it without numpy complaining
        """
        r = self.L*np.linalg.norm(self.He.table, axis=-1)
        r[np.diag_indices(self.He.N)] = self.a1 #prevent division by zero
        logpart = np.log(self.a1 / r)
        return -np.sum(np.triu(logpart*self.u, k=1))

    def drift(self, i):
        """
        similar to g_logpsi but is just the gradient w.r.t atom i
        ie returns vector of length 3

        separate function since it is a sum over a smaller array
        """
        return -np.sum(self.g_u[i], axis=0)

    def local_kinetic(self):
        """
        laplacian of psi over psi,
        summed over all particles,
        multiplied by minus hbar^2 over 2*mass

        in kelvin, to match potential()
        """
        #laplacian of u
        r = self.L*np.linalg.norm(self.He.table, axis=-1)
        r[np.diag_indices(self.He.N)] = np.inf #prevent division by zero
        l_u = self.u*(self.a2 - 1)*self.a2 / r**2
        #sum over l_u gives laplacian of logpsi summed over all particles

        g_logpsi = self.g_logpsi()
        l_psi_over_psi = -np.sum(l_u) + np.einsum('ix,ix', g_logpsi, g_logpsi)
        return -6.0599*l_psi_over_psi

    def local_potential(self):
        r = self.L*np.linalg.norm(self.He.table, axis=-1)
        r[np.diag_indices(self.He.N)] = np.inf #prevent division by zero
        return np.sum(potential(r))

    def move(self, i, scale):
        """
        attempt to move atom i
        with a uniform random displacement (in units of L)
        scaled by scale arg
        note that scale is also in units of L
        """
        step = scale*(np.random.rand(3) - 0.5)
        #save old u, g_u, in case of rejection
        old_u = np.copy(self.u)
        old_g_u = np.copy(self.g_u)
        old_logpsi = self.logpsi()
        self.He.r[i] += step
        self.He.update_r(i)
        self.update_single(i)
        #begin computing acceptance ratio
        logprobability = 2*(self.logpsi() - old_logpsi)
        if logprobability > 0:
            #just set to 0 to prevent potential overflow
            logprobability = 0
        probability = math.exp(logprobability)
        if random() < probability:
            #move has been accepted
            return True
        else:
            #move has been rejected
            self.He.r[i] -= step
            self.He.update_r(i)
            self.u = np.copy(old_u)
            self.g_u = np.copy(old_g_u)
            return False

    def move_drift(self, i, timestep):
        """
        attempt to move atom i
        with a drift
        timestep is NOT rescaled by L
        i.e. the generated displacement is meant to be in angstroms
        """
        step = np.random.normal(scale=math.sqrt(timestep), size=3)
        #save old u, g_u, r, in case of rejection
        old_u = np.copy(self.u)
        old_g_u = np.copy(self.g_u)
        #to compute probability
        old_logpsi = self.logpsi()
        old_drift = self.drift(i)
        old_position = np.copy(self.He.r[i])
        #begin move
        step += timestep*old_drift
        self.He.r[i] += step/self.L
        self.He.update_r(i)
        self.update_single(i)
        #begin computing acceptance ratio
        logprobability = 2*(self.logpsi() - old_logpsi)
        logprobability += log_transition_ratio(timestep=timestep,\
                position_new=self.He.r[i]*self.L,\
                position_old=old_position*self.L,\
                drift_new=self.drift(i), drift_old=old_drift)
        if logprobability > 0:
            #just set to 0 to prevent potential overflow
            logprobability = 0
        probability = math.exp(logprobability)
        if random() < probability:
            #move has been accepted
            return True
        else:
            #move has been rejected
            self.He.r[i] = np.copy(old_position)
            self.He.update_r(i)
            self.u = np.copy(old_u)
            self.g_u = np.copy(old_g_u)
            return False

    def move_all(self, scale):
        """
        move all atoms according to scale
        returns total number of acceptances, not the average
        """
        accepted = 0
        for i in range(self.He.N):
            accepted += self.move(i, scale)
        return accepted

    def move_all_drift(self, timestep):
        """
        move all atoms with drift using timestep
        returns total number of acceptances, not the average
        """
        accepted = 0
        for i in range(self.He.N):
            accepted += self.move_drift(i, timestep)
        return accepted
       
if __name__=="__main__":
    np.random.seed(69)
    #initialize
    N = 32
    wf = wavefunction(N, 2.72, 5.018)
    #wf.He.start_semirandom(0.01)
    with open('configs/He4/%s.dat' % N) as f:
        wf.He.r = np.loadtxt(f)
    wf.He.update_displacement()
    wf.update_all()

#    #test update_all against update_single
#    wf2 = deepcopy(wf)
#    for i in range(N):
#        wf2.update_single(i)
    
#    #test derivative
#    wf.move_all(0.1)
#    step = 0.0001
#    wf.a1 += step
#    wf.update_all()
#    fwd = wf.logpsi()
#    wf.a1 -= 2*step
#    wf.update_all()
#    bwd = wf.logpsi()
#    estimate = (fwd - bwd) / (2*step)
#    #return
#    wf.a1 += step
#    wf.update_all()

#    #test kinetic energy after sampling once
#    wf.move_all(0.3)
#    print('exact: %f' % (wf.local_kinetic()))
#    step = 0.000001 #in units of L
#    mid = wf.logpsi()
#    estimate = 0
#    for n in range(N):
#        for i in range(3):
#            wf.He.r[n,i] += step
#            wf.He.update_displacement()
#            wf.update_all()
#            fwd = wf.logpsi()
#            wf.He.r[n,i] -= 2*step
#            wf.He.update_displacement()
#            wf.update_all()
#            bwd = wf.logpsi()
#            estimate += (math.exp(fwd - mid) +\
#                    math.exp(bwd - mid) - 2) /\
#                    (step*wf.L)**2
#            #return
#            wf.He.r[n,i] += step
#            wf.He.update_displacement()
#            wf.update_all()
