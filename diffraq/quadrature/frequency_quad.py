"""
frequency_quad.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 11-12-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: quadrature in frequency space to use in non-uniform angular spectrum.

"""

from diffraq.quadrature import polar_quad
import numpy as np

def frequency_quad(wave, zz, m, n, Dmax, grid_pts, over_sample=6):
    """
    xf, yf, wf = frequency_quad(wave, zz, m, n, Dmax, grid_pts, over_sample=6)

    calculates quadrature points and weights in frequency space to non-uniformally
        sample angular spectrum.

    Inputs:
        wave = wavelength [meters]
        zz = propagation distance
        m = # of radial nodes
        n = # of theta nodes
        Dmax = max width of occulter
        grid_pts = target grid
        over_sample = over sampling of Nyquist

    Outputs:
        xf, yf = numpy array of x,y coordinates of nodes
        wf = numpy array of weights
    """

    #Get sampling requirements
    dgrid = grid_pts[1] - grid_pts[0]
    D = (Dmax/2 + grid_pts.max())       #Maximum extent

    #Get max alpha
    amax = max(D/np.hypot(zz,D)/wave, Dmax/(wave*zz)) * wave * over_sample
    amax = min(amax, 1/np.sqrt(2))

    #Get frequency quadrature
    xqf, yqf, wqf = polar_quad(lambda t: np.ones_like(t)*amax, m, n)

    return xqf, yqf, wqf
