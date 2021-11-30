"""
square_quad.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 11-29-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: quadrature for integrals over rectangular.
    See Barnett (2021) Eq. 11, 12, 14.

"""

from diffraq.quadrature import lgwt
import numpy as np

def square_quad(wx, wy, m, n):
    """
    xq, yq, wq = square_quad(wx, wy, m, n)

    Inputs:
        wx = width of x side
        wy = width of y side
        m = # radial nodes
        n = # theta nodes

    Outputs:
        xq, yq = numpy array of x,y coordinates of nodes [meters]
        wq = numpy array of weights
    """

    #Theta nodes, constant weights
    pt = 2.*np.pi/n * (np.arange(n)[:,None] + 1)
    wt = 2.*np.pi/n

    #Radial quadrature nodes, weights
    pr, wr = lgwt(m, 0, 1)

    #Get function values at all theta nodes
    ft = fxy(pt)[:,:,None]
    dt = dxy(pt)[:,:,None]

    #Nodes (Eq. 11 Barnett 2021)
    xq = (ft[:,0] * pr).ravel()
    yq = (ft[:,1] * pr).ravel()

    #Weights (Eq. 12 Barnett 2021)
    wq = wt * (pr * wr * (ft[:,0] * dt[:,1] - ft[:,1] * dt[:,0])).ravel()

    #Cleanup
    del pt, pr, wr, ft, dt

    return xq, yq, wq

############################################
############################################

def cartesian_edge(fxy, n):
    """
    xy = square_edge(wx, wy, n)

    Build loci demarking the cartesian occulter edge.

    Inputs:
        wx = width of x side
        wy = width of y side
        n = # theta nodes

    Outputs:
        xy = numpy array (2D) of x,y coordinates of occulter edge [meters]
    """

    #Theta nodes
    pt = 2.*np.pi/n * (np.arange(n) + 1)[:,None]

    #Boundary points
    return fxy(pt)
