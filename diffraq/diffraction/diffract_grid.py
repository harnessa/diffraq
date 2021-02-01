"""
diffract_grid.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-15-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: diffraction calculation of input quadrature to target grid.
    Taken from FRESNAQ's fresnaq_grid.m (Barnett 2021).

"""

import numpy as np
import finufft

def diffract_grid(xq, yq, wq, lambdaz, grid_pts, tol, is_babinet=False):
    """
    uu = diffract_grid(xq, yq, wq, lambdaz, grid_pts, tol, is_babinet)

    calculates diffraction propagation from input quadrature to target grid (Barnet 2021).

    Inputs:
        xq, yq = x,y coordinates of area quadrature nodes [meters]
        wq = area quadrature weights
        lambdaz = wavelength * z [meters^2]
        grid_pts = target grid points (1D) [meters]
        tol = tolerance to which to calculate FFT (via finufft)
        is_babinet = calculation implies Babinet's Principle and need to subtract 1?

    Outputs:
        uu = complex field over target grid
    """

    #Scale factor to become FT
    sc = 2.*np.pi/lambdaz
    #Scaled grid spacing
    dk = sc * (grid_pts[1] - grid_pts[0])
    ngrid = len(grid_pts)
    #Max NU coord
    maxNU = max(abs(dk*xq).max(), abs(dk*yq).max())

    #Premultiply by quadratric kernel
    cq = np.exp(1j*np.pi/lambdaz*(xq**2 + yq**2)) * wq

    #only if needed
    if maxNU > 3*np.pi:
        #Wrap in case grid too coarse
        xq = np.mod(xq, 2*np.pi/dk)
        yq = np.mod(yq, 2*np.pi/dk)

    #Do FINUFFT
    uu = finufft.nufft2d1(dk*xq, dk*yq, cq, (ngrid, ngrid), isign=-1, eps=tol)

    #post multiply by quadratic phase of target and Kirchoff prefactor
    tarq = np.exp((1j*np.pi/lambdaz)*grid_pts**2)
    uu *= 1./(1j*lambdaz) * (tarq * tarq[:,None])

    #Subtract from Babinet field
    if is_babinet:
        uu = 1.+0j - uu

    return uu
