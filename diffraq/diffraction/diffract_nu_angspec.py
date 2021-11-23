"""
diffract_nu_angspec.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 11-12-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: diffraction calculation of input quadrature to target grid using
    the non-uniform angular spectrum method.

"""

import numpy as np
import finufft

def diffract_nu_angspec(xq, yq, wq, u0, fx, fy, fw, wave, zz, tx, ty, tol, is_babinet=False, z0=1e19):
    """
    uu = nu_diffract_angspec(xq, yq, wq, u0, fx, fy, fw, wave, zz, tx, ty, tol, is_babinet=False)

    calculates diffraction propagation from input quadrature to target grid
        using the angular spectrum method and nufft

    Inputs:
        xq, yq = x,y coordinates of area quadrature nodes [meters]
        wq = area quadrature weights
        u0 = incident field
        fx, fy = x,y coordinates of frequency-space area quadrature nodes
        fw = frequency-space area quadrature weights
        wave = wavelength [meters]
        zz = propagation distance
        tx, ty = target grid coordinats (flatten)
        tol = tolerance to which to calculate FFT (via finufft)
        is_babinet = calculation implies Babinet's Principle and need to subtract 1?
        z0 = source distance

    Outputs:
        uu = complex field over target grid
    """

    ###################################
    ### Angular Spectrum ###
    ###################################

    #Number of points
    num_pts = int(np.sqrt(tx.size))

    #Critical distance
    dx = xq.max()/np.sqrt(xq.shape)     #TODO: what about non-uniform?
    zcrit = 2*num_pts*dx**2/wave

    #Calculate bandwidth
    if zz < zcrit:
        bf = 1/dx
    elif zz >= 3*zcrit:
        bf = np.sqrt(2*num_pts/(wave*zz))
    else:
        bf = 2*num_pts*dx/(wave*zz)

    #Take half-bandwidth
    hbf = bf/2

    #Scale factor to become FT
    scl = 2.*np.pi * hbf

    #Compute strengths
    cq = u0 * wq

    #Do FINUFFT
    aspec = finufft.nufft2d3(xq, yq, cq, scl*fx, scl*fy, isign=-1, eps=tol)

    ###################################
    ### Diffraction ###
    ###################################

    #Get propagation constant
    kz = 1 - (fx*wave*hbf)**2 - (fy*wave*hbf)**2
    k0inds = kz < 0
    kz = np.exp(1j * 2*np.pi/wave * np.sqrt(np.abs(kz)) * zz)
    #Zero out evanescent waves
    kz[k0inds] *= 0

    #Propagation kernel * aspec as strengths
    cq = aspec * kz * fw * hbf**2

    #Cleanup
    del kz, k0inds, aspec

    #Do FINUFFT (inverse direction)
    uu = finufft.nufft2d3(scl*fx, scl*fy, cq, tx, ty, isign=1, eps=tol)

    #Subtract from Babinet field
    if is_babinet:
        #Incident field (distance in denominator of quadratic phase = z0 + zz)
        ub = z0/(zz + z0) * np.exp((1j*np.pi/wave/(zz + z0))*(tx**2 + ty**2)) * \
            np.exp(1j*2*np.pi/wave*zz)

        #Subtract from inciden field
        uu = ub - uu

        #Cleanup
        del ub

    #Reshape
    uu = uu.reshape((num_pts, num_pts))

    #Transpose to match visual representation
    uu = uu.T

    #Cleanup
    del cq

    return uu
