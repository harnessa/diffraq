"""
diffract_points.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-15-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: diffraction calculation of input quadrature to arbitrary points.
    Taken from FRESNAQ's fresnaq_grid.m (Barnett 2021).

"""

import numpy as np
import finufft

def diffract_points(xq, yq, wq, lambdaz, xi, eta, tol):
    """
    uu = diffract_points(xq, yq, wq, lambdaz, xi, eta, tol)

    calculates diffraction propagation from input quadrature to arbitrary
    points (xi, eta). (Barnet 2021)

    Inputs:
        xq, yq = x,y coordinates of area quadrature nodes
        wq = area quadrature weights
        lambdaz = wavelength * z [m^2]
        xi, eta = target points (each must be 1D numpy array)
        tol = tolerance to which to calculate FFT (via finufft)

    Outputs:
        uu = complex field at target points
    """

    #Premultiply by quadratric kernel (cq are input strengths to NUFFT)
    cq = np.exp(1j*np.pi/lambdaz*(xq**2 + yq**2)) * wq
    #Scale factor to become FT
    sc = 2.*np.pi/lambdaz

    #Do FINUFFT
    uu = finufft.nufft2d3(xq, yq, cq, sc*xi, sc*eta, isign=-1, eps=tol)

    #post multiply bit
    uu *= 1./(1j*lambdaz) * np.exp((1j*np.pi/lambdaz)*(xi**2 + eta**2))

    return uu
