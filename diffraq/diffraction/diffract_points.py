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

def diffract_points(xq, yq, wq, lamzz, xi, eta, tol, is_babinet=False, lamz0=1e13):
    """
    uu = diffract_points(xq, yq, wq, lambdaz, xi, eta, tol)

    calculates diffraction propagation from input quadrature to arbitrary
    points (xi, eta). (Barnett 2021)

    Inputs:
        xq, yq = x,y coordinates of area quadrature nodes [meters]
        wq = area quadrature weights
        lamzz = wavelength * zz [meters^2] (diffractor - target distance)
        xi, eta = target points (each must be 1D numpy array) [meters]
        tol = tolerance to which to calculate FFT (via finufft)
        is_babinet = calculation implies Babinet's Principle and need to subtract 1?
        lamz0 = wavelength * z0 [meters^2] (source - diffractor distance)

    Outputs:
        uu = complex field at target points
    """

    #Premultiply by quadratric kernel (cq are input strengths to NUFFT)
    lamzeff = 1/(1/lamzz + 1/lamz0)
    cq = np.exp(1j*np.pi/lamzeff*(xq**2 + yq**2)) * wq
    #Scale factor to become FT
    sc = 2.*np.pi/lamzz

    #Do FINUFFT
    uu = finufft.nufft2d3(xq, yq, cq, sc*xi, sc*eta, isign=-1, eps=tol)

    #Multiply by quadratic phase at target and Kirchhoff amplitude scaling
    uu *= 1./(1j*lamzz) * np.exp((1j*np.pi/lamzz)*(xi**2 + eta**2))

    #Subtract from Babinet field
    if is_babinet:
        #Incident field (distance in denominator of quadratic phase = z0 + zz)
        u0 = lamz0/(lamzz + lamz0) * np.exp((1j*np.pi/(lamzz + lamz0))*(xi**2 + eta**2))

        #Subtract from incident field
        uu = u0 - uu

    #Cleanup
    del cq

    return uu
