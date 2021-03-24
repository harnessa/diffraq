"""
seam_polar_quad.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 03-18-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: quadrature over seam of edge with polar function.

"""

from diffraq.quadrature import lgwt
import numpy as np

def seam_polar_quad(g, m, n, seam_width):
    """
    xq, yq, wq = seam_polar_quad(g, m, n)

    returns list of nodes and their weights for 2D quadrature over a polar domain,

        sum_{j=1}^N f(xq(j),yq(j)) wq(j)   \approx   int_Omega f(x,y) dx dy

    for all smooth functions f, where Omega is the polar domain defined by
    radial function r=g(theta). (Barnett 2021)

    Inputs:
        g = function handle for g(theta), theta in [0,2pi)
        m = # radial nodes
        n = # theta nodes
        seam_width = seam half-width [meters]

    Outputs:
        xq, yq = numpy array of x,y coordinates of nodes [meters]
        wq = numpy array of weights
    """

    #Theta nodes, constant weights
    pt = 2.*np.pi/n * (np.arange(n)[:,None] + 1)
    wt = 2.*np.pi/n

    #Radial quadrature nodes, weights
    pr, wr = lgwt(m, -1, 1)

    #Get radial function values at all nodes
    gtpr = g(pt) + pr * seam_width

    #line of nodes
    xq = (np.cos(pt) * gtpr).ravel()
    yq = (np.sin(pt) * gtpr).ravel()

    #Get weights r*dr = rr*seam_width*wr
    wq = wt * seam_width * (gtpr * wr).ravel()

    #Cleanup
    del wr, gtpr

    #Return nodes along primary axis (radius) and values along orthogonal axis (theta)
    return xq, yq, wq, pr, pt

############################################
############################################

def seam_polar_edge(g, n, seam_width):
    """
    xy = seam_polar_edge(g, n, seam_width)

    Build loci demarking the polar occulter edge.

    Inputs:
        g = function handle for g(theta), theta in [0,2pi)
        n = # theta nodes
        seam_width = seam half-width [meters]

    Outputs:
        xy = numpy array (2D) of x,y coordinates of occulter edge [meters]
    """

    #Theta nodes
    pt = 2.*np.pi/n * (np.arange(n) + 1)

    #Radial edges of seam
    gtpr = (g(pt) + seam_width*np.array([1,-1]))[:,None]

    #Boundary points
    xy = gtpr * np.stack((np.cos(pt), np.sin(pt)), 1)

    return xy
