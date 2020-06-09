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
        self.atom = pbc.proton(N) #N spinless particles
        self.u = np.zeros((N, N)) #-log(f), where f defined in eq 5
        self.a1 = a1
        self.a2 = a2
        
    def update_u_all(self):
        """
        update all entries in u
        the trick is to avoid the diagonal of atom.table, since it's zero
        """
        r = self.L*np.linalg.norm(self.atom.table, axis=-1)
        r[r == 0] = np.inf #set zeros to inf to prevent division by 0
        self.u = jas.mcmillan(r, self.a1, self.a2)
        #self.u = np.where(r > 0, jas.mcmillan(r, self.a1, self.a2), 0)
        
    def update_u(self, i):
        """
        update entries in u that are related to atom i
        """
        r = self.L*np.linalg.norm(self.atom.table[i], axis=-1)
        r[i] = np.inf
        #newrow = np.where(r > 0, jas.mcmillan(r, self.a1, self.a2), 0)
        newrow = jas.mcmillan(r, self.a1, self.a2)
        self.u[i,:] = np.copy(newrow)
        self.u[:,i] = np.copy(newrow)
        
    def log_psi(self):
        """
        returns log of wavefunction
        """
        return -np.sum(np.triu(self.u, k=1))
    
    def grad_u(self):
        """
        returns gradient of u, same shape as displacement table
        also antisymmetric, just like aforementioned table
        
        to be done: save this gradient as an attribute of object
        """
        r = self.L*np.linalg.norm(self.atom.table, axis=-1)
        r[r == 0] = np.inf
        uprime = -(self.a2 / r)*self.u
        ugrad = uprime[:,:,np.newaxis]*self.atom.table
        ugrad /= r[:,:,np.newaxis]
        return ugrad
    
    def sample(self, i, scale):
        """
        generate sample by moving atom i
        with a uniform random move, scaled by scale
        note that scale is in units of L
        """
        displacement = scale*(np.random.rand(3) - 0.5)
        #save old u, in case of rejection
        old_u = np.copy(self.u)
        self.atom.r[i] += displacement
        self.atom.update_r(i)
        self.update_u(i)
        #begin computing acceptance ratio
        ratio = np.sum(self.u[i] - old_u[i])
        if ratio < 0:
            #just set to 0 to prevent potential overflow
            ratio = 0
        ratio = math.exp(-2*ratio)
        if random() < ratio:
            #move has been accepted
            return True
        else:
            #move has been rejected
            self.atom.r[i] -= displacement
            self.atom.update_r(i)
            self.u = np.copy(old_u)
            return False
        
if __name__=="__main__":
    #initialize
    N = 11
    test = wavefunction(N, 2.6, 5)
    #test.atom.start_random()
    with open('11.cfg', 'r') as f:
        test.atom.r = np.loadtxt(f)
    test.atom.update_displacement()
    test.update_u_all()