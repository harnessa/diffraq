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

def diffract_grid(xq, yq, wq, lamzz, grid_pts, tol, is_babinet=False, lamz0=1e13):
    """
    uu = diffract_grid(xq, yq, wq, lamzz, grid_pts, tol, is_babinet, lamz0)

    calculates diffraction propagation from input quadrature to target grid (Barnet 2021).

    Inputs:
        xq, yq = x,y coordinates of area quadrature nodes [meters]
        wq = area quadrature weights
        lamzz = wavelength * zz [meters^2] (diffractor - target distance)
        grid_pts = target grid points (1D) [meters]
        tol = tolerance to which to calculate FFT (via finufft)
        is_babinet = calculation implies Babinet's Principle and need to subtract 1?
        lamz0 = wavelength * z0 [meters^2] (source - diffractor distance)

    Outputs:
        uu = complex field over target grid
    """

    #Scale factor to become FT
    sc = 2.*np.pi/lamzz
    #Scaled grid spacing
    dk = sc * (grid_pts[1] - grid_pts[0])
    ngrid = len(grid_pts)
    #Max NU coord
    maxNU = max(abs(dk*xq).max(), abs(dk*yq).max())

    #Premultiply by quadratric kernel
    lamzeff = 1/(1/lamzz + 1/lamz0)
    cq = np.exp(1j*np.pi/lamzeff*(xq**2 + yq**2)) * wq

    #only if needed
    if maxNU > 3*np.pi:
        #Wrap in case grid too coarse
        xq = np.mod(xq, 2*np.pi/dk)
        yq = np.mod(yq, 2*np.pi/dk)

    #Do FINUFFT
    uu = finufft.nufft2d1(dk*xq, dk*yq, cq, (ngrid, ngrid), isign=-1, eps=tol)

    #post multiply by quadratic phase of target and Kirchoff prefactor
    tarq = np.exp((1j*np.pi/lamzz)*grid_pts**2)
    uu *= 1./(1j*lamzz) * (tarq * tarq[:,None])

    #Subtract from Babinet field
    if is_babinet:
        #Incident field (distance in denominator of quadratic phase = z0 + zz)
        tarq0 = np.exp((1j*np.pi/(lamzz + lamz0))*grid_pts**2)
        u0 = lamz0/(lamzz + lamz0) * (tarq0 * tarq0[:,None])

        #Subtract from inciden field
        uu = u0 - uu

    return uu
