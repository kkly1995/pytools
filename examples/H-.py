#simple solveable problem, in order to fix drift
import numpy as np
import math
from random import random
from pytools.qmc import log_transition_ratio
from pytools.jastrow import pade, pade_prime, laplacian_pade, pade_params

class wavefunction:
    def __init__(self, Z, a, b, c):
        """
        slater orbitals with 'charge' Z
        and pade jastrow with parameters a, b, c
        """
        self.r = np.random.normal(size=(2, 3)) #begin random spots
        self.Z = Z
        self.a = a
        self.b = b
        self.c = c

    def logpsi(self):
        r = sum(np.linalg.norm(self.r, axis=1))
        r_ee = np.linalg.norm(self.r[1] - self.r[0])
        u = pade(r_ee, self.a, self.b, self.c)
        return -self.Z*r + u

    def param_derivs_logpsi(self):
        """
        derivatives of logpsi w.r.t
        Z, a, b, c
        respectively
        """
        r = sum(np.linalg.norm(self.r, axis=1))
        r_ee = np.linalg.norm(self.r[1] - self.r[0])
        d_da, d_db, d_dc = pade_params(r_ee, self.a, self.b, self.c)
        d_dZ = -r
        return d_dZ, d_da, d_db, d_dc

    def g_logpsi(self):
        rhat = self.r / np.linalg.norm(self.r, axis=1)[:,np.newaxis]
        r_ee = np.linalg.norm(self.r[1] - self.r[0])
        r_eehat = (self.r[1] - self.r[0]) / r_ee
        uprime = pade_prime(r_ee, self.a, self.b, self.c)
        val = -self.Z*rhat
        val[0] -= uprime*r_eehat
        val[1] += uprime*r_eehat
        return val

    def drift(self, i):
        rhat = self.r[i] / np.linalg.norm(self.r[i])
        r_ee = np.linalg.norm(self.r[1] - self.r[0])
        r_eehat = (self.r[i] - self.r[i-1]) / r_ee
        uprime = pade_prime(r_ee, self.a, self.b, self.c)
        val = -self.Z*rhat
        val += uprime*r_eehat
        return val

    def local_kinetic(self):
        r = np.linalg.norm(self.r, axis=1)
        r_ee = np.linalg.norm(self.r[1] - self.r[0])
        l_logpsi = -2*self.Z*(1/r[0] + 1/r[1])
        l_logpsi += 2*laplacian_pade(r_ee, self.a, self.b, self.c)
        g_logpsi = self.g_logpsi()
        l_psi_over_psi = l_logpsi + np.dot(g_logpsi[0], g_logpsi[0]) + \
                np.dot(g_logpsi[1], g_logpsi[1])
        return -0.5*l_psi_over_psi

    def local_potential(self):
        r_ee = np.linalg.norm(self.r[1] - self.r[0])
        r = np.linalg.norm(self.r, axis=1)
        return 1./r_ee - 1./r[0] - 1./r[1]

    def move(self, scale):
        #move both electrons simultaneously
        step = scale*(np.random.rand(2,3) - 0.5)
        old_logpsi = self.logpsi()
        self.r += step
        logprobability = 2*(self.logpsi() - old_logpsi)
        if logprobability > 0:
            logprobability = 0
        probability = math.exp(logprobability)
        if random() < probability:
            return True
        else:
            self.r -= step
            return False

    def move_drift(self, i, timestep):
        step = np.random.normal(scale=math.sqrt(timestep), size=3)
        old_logpsi = self.logpsi()
        old_drift = self.drift(i)
        old_r = np.copy(self.r[i])
        self.r[i] += step + timestep*old_drift
        logprobability = 2*(self.logpsi() - old_logpsi)
        logprobability += log_transition_ratio(\
                timestep=timestep, position_new = self.r[i],\
                position_old = old_r, drift_new = self.drift(i),\
                drift_old = old_drift)
        if logprobability > 0:
            logprobability = 0
        probability = math.exp(logprobability)
        if random() < probability:
            return True
        else:
            self.r[i] = np.copy(old_r)
            return False

    def move_multiple(self, scale, N):
        #call move() N times and record number of acceptances
        a = 0
        for i in range(N):
            a += self.move(scale)
        return a

    def move_drift_all(self, timestep):
        accepted = 0
        for i in range(2):
            accepted += self.move_drift(i, timestep)
        return accepted

if __name__=="__main__":
    #np.random.seed(69)
    Z = 0.8812
    a = 0.048
    b = 0.9886
    c = 0.241
    wf = wavefunction(Z, a, b, c)
