"""
polar_quad.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-08-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: quadrature for integrals over area with polar function.
    Taken from FRESNAQ's polarareaquad.m (Barnett 2021).

"""

from diffraq.quadrature import lgwt
import numpy as np

def polar_quad(g, m, n):
    """
    xq, yq, wq = polar_quad(g, m, n)

    returns list of nodes and their weights for 2D quadrature over a polar domain,

        sum_{j=1}^N f(xq(j),yq(j)) wq(j)   \approx   int_Omega f(x,y) dx dy

    for all smooth functions f, where Omega is the polar domain defined by
    radial function r=g(theta). (Barnett 2021)

    Inputs:
        g = function handle for g(theta), theta in [0,2pi)
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
    gt = g(pt)

    #line of nodes
    gtpr = gt * pr
    xq = (np.cos(pt) * gtpr).flatten()
    yq = (np.sin(pt) * gtpr).flatten()

    #Theta weight times rule for r*dr on (0,r)
    wq = wt * (gt**2 * pr * wr).flatten()

    #Cleanup
    del pt, wt, pr, wr, gt, gtpr

    return xq, yq, wq

############################################
############################################

def polar_edge(g, m, n):
    """
    xe, ye = polar_edge(g, m, n)

    Build loci demarking the polar occulter edge.

    Inputs:
        g = function handle for g(theta), theta in [0,2pi)
        m = # radial nodes (unused)
        n = # theta nodes

    Outputs:
        xe, ye = numpy array of x,y coordinates of occulter edge [meters]
    """

    #Theta nodes
    pt = 2.*np.pi/n * (np.arange(n)[:,None] + 1)

    #Boundary points
    bx = np.cos(pt) * g(pt)
    by = np.sin(pt) * g(pt)

    return bx, by
