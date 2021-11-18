"""
diffract_RS1.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 11-18-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: diffraction calculation of input quadrature to target grid via
    direct integration of Rayleigh-Sommerfeld equation of the first kind.

"""

import numpy as np
from mpi4py import MPI
rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

def diffract_RS1(xq, yq, wq, wave, zz, gx_2D, gy_2D=None, is_babinet=False, z0=1e19):
    #Target x,y are the same
    if gy_2D is None:
        gy_2D = gx_2D.T

    #Source-Occulter
    rr = np.sqrt(xq**2 + yq**2 + z0**2)

    #Pre-calcs
    kk = 2*np.pi/wave
    U0wq = np.exp(1j*kk*rr) * (z0/rr) * wq

    #Build solution
    uu_tmp = np.zeros_like(gx_2D) + 0j

    #Loop over grid and calculate
    for i in range(gx_2D.shape[0]):
        for j in range(gx_2D.shape[1]):

            if (i*gx_2D.shape[0] + j) % size != rank:
                continue

            #Occulter-Target
            ss = np.sqrt((xq - gx_2D[i,j])**2 + (yq - gy_2D[i,j])**2 + zz**2)

            #Integrate
            uu_tmp[i,j] = np.sum(U0wq * (np.exp(1j*kk*ss)/ss) * (zz/ss) * (1j*kk + 1/ss))

    #Collect processors
    MPI.COMM_WORLD.Barrier()
    uu = np.zeros_like(uu_tmp)
    nothing = MPI.COMM_WORLD.Allreduce(uu_tmp, uu)

    #Add constant prefactors
    uu *= -1/(2*np.pi)

    #Subtract from Babinet field
    if is_babinet:
        #Incident field (distance in denominator of quadratic phase = z0 + zz)
        u0 = z0/(zz + z0) * np.exp(1j*2*np.pi/wave*z0) * \
            np.exp((1j*np.pi/(wave*(zz + z0)))*(xi**2 + eta**2))

        #Subtract from incident field
        uu = u0 - uu

        #Cleanup
        del u0

    #Cleanup
    del rr, U0wq, ss, uu_tmp

    return uu
