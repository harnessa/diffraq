"""
cartesian_quad.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-05-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: quadrature for integrals over area with cartesian parametric functions.
    See Barnett (2021) Eq. 11, 12, 14.

"""

from diffraq.quadrature import lgwt
import numpy as np

def cartesian_quad(fx, fy, dx, dy, m, n):
    """
    xq, yq, wq = cartesian_quad(fx, fy, dx, dy, m, n)

    returns list of nodes and their weights for 2D quadrature, over a cartesian domain,

        sum_{j=1}^N f(xq(j),yq(j)) wq(j)   \approx   int_Omega f(x,y) dx dy

    for all smooth functions f, where Omega is the cartesian domain defined by
    function (x,y) = fxy(theta). (Barnett 2021)

    Inputs:
        fx[y] = function handle for [y]]) = fx[y](theta), theta in [0,2pi)
        dx[y] = function handle for derivative of fx[y], theta in [0,2pi)
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
    ftx = fx(pt)
    fty = fy(pt)

    #Nodes (Eq. 11 Barnett 2021)
    xq = (ftx[:,None] * pr).ravel()
    yq = (fty[:,None] * pr).ravel()

    #Weights (Eq. 12 Barnett 2021)
    wq = wt * (pr * wr * (ftx * dy(pt) - fty * dx(pt))[:,None]).ravel()

    #Cleanup
    del pt, pr, wr, ftx, fty

    return xq, yq, wq

############################################
############################################

def cartesian_edge(fx, fy, n):
    """
    xy = cartesian_edge(fx, fy, n)

    Build loci demarking the cartesian occulter edge.

    Inputs:
        fx[y] = function handle for [y]]) = fx[y](theta), theta in [0,2pi)
        n = # theta nodes

    Outputs:
        xy = numpy array (2D) of x,y coordinates of occulter edge [meters]
    """

    #Theta nodes
    pt = 2.*np.pi/n * (np.arange(n) + 1)

    #Boundary points
    return np.stack((fx(pt), fy(pt)),1)
