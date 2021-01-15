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

def diffract_grid(xq, yq, wq, lambdaz, grid_pts, tol):
    """
    uu = diffract_grid(xq, yq, wq, lambdaz, grid_pts, tol)

    calculates diffraction propagation from input quadrature to target grid (Barnet 2021).

    Inputs:
        xq, yq = x,y coordinates of area quadrature nodes
        wq = area quadrature weights
        lambdaz = wavelength * z [m^2]
        grid_pts = target grid points (1D)
        tol = tolerance to which to calculate FFT (via finufft)

    Outputs:
        uu = complex field over target grid
    """

    #Scale factor to become FT
    sc = 2.*np.pi/lambdaz
    #Scaled grid spacing
    dk = sc * (grid_pts[1] - grid_pts[0])
    ngrid = len(grid_pts)
    #Max NU coord
    maxNU = dk*max(abs(xq).max(), abs(yq).max())

    #Premultiply by quadratric kernel
    cq = np.exp(1j*np.pi/lambdaz*(xq**2 + yq**2)) * wq

    #only if needed
    if maxNU > 3*np.pi:
        #Wrap in case grid too coarse
        xq = np.mod(xq, 2*np.pi/dk)
        yq = np.mod(yq, 2*np.pi/dk)

    #Do FINUFFT
    uu = finufft.nufft2d1(dk*xq, dk*yq, cq, isign=-1, eps=tol, n_modes=(ngrid, ngrid))
    #post multiply by quadratic phase of target
    tarq = np.exp((1j*np.pi/lambdaz)*grid_pts**2)
    uu *= tarq * tarq[:,None]
    #Mulitply by Kirchhoff prefactor
    uu *= 1./(1j*lambdaz)

    return uu

def build_grid(ngrid, grid_rad):
    """
    grid_pts = build_grid(ngrid, grid_rad)

    build uniformly spaced target grid from number of points and size

    Inputs:
        ngrid = length of one size of target grid
        grid_rad = physical half-width of target [m]

    Outputs:
        grid_pts = target grid points (1D)
    """

    #Build target grid
    grid_pts = grid_rad*(2*np.arange(ngrid)/ngrid - 1)
    #Grid spacing
    dx = 2*grid_rad/ngrid
    #Handle the odd case
    if ngrid % 2 == 1:
        grid_pts += dx/2

    return grid_pts
