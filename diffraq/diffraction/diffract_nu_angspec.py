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

def diffract_nu_angspec(xq, yq, wq, u0, xf, yf, wf, wave, zz, grid_pts, tol, is_babinet=False):
    """
    uu = nu_diffract_angspec(xq, yq, wq, u0, xf, yf, wf, wave, zz, grid_pts, tol, is_babinet=False)

    calculates diffraction propagation from input quadrature to target grid
        using the angular spectrum method and nufft

    Inputs:
        xq, yq = x,y coordinates of area quadrature nodes [meters]
        wq = area quadrature weights
        u0 = incident field
        xf, yf = x,y coordinates of frequency-space area quadrature nodes
        wf = frequency-space area quadrature weights
        wave = wavelength [meters]
        zz = propagation distance
        grid_pts = target grid
        tol = tolerance to which to calculate FFT (via finufft)
        is_babinet = calculation implies Babinet's Principle and need to subtract 1?

    Outputs:
        uu = complex field over target grid
    """

    ###################################
    ### Angular Spectrum ###
    ###################################

    #Scale factor to become FT
    sc = 2.*np.pi/wave

    #Compute strengths
    cq = u0 * wq

    #Do FINUFFT
    aspec = finufft.nufft2d3(xq, yq, cq, sc*xf, sc*yf, isign=-1, eps=tol)

    ###################################
    ### Diffraction ###
    ###################################

    #Scaled grid spacing
    ngrid = len(grid_pts)
    dgrid = grid_pts[1] - grid_pts[0]
    dkg = sc * dgrid

    #Max NU coord
    maxNU = dkg * max(abs(xf).max(), abs(yf).max())

    #only if needed
    if maxNU > 3*np.pi:
        #Wrap in case grid too coarse
        xf = np.mod(xf, 2*np.pi/dkg)
        yf = np.mod(yf, 2*np.pi/dkg)

    #Get propagation constant
    kz = 1 - xf**2 - yf**2
    k0inds = kz < 0
    kz = np.exp(1j * 2*np.pi/wave * np.sqrt(np.abs(kz)) * zz)
    #Zero out evanescent waves
    kz[k0inds] *= 0

    #Propagation kernel * aspec as strengths
    cq = aspec * kz * wf

    #Cleanup
    del kz, k0inds, aspec

    #Do FINUFFT (inverse direction)
    uu = finufft.nufft2d1(dkg*xf, dkg*yf, cq, (ngrid, ngrid), isign=1, eps=tol)

    #Transpose to match visual representation
    uu = uu.T

    #Normalize
    uu /= wave**2

    #Cleanup
    del cq

    return uu
