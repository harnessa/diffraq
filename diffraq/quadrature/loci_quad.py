"""
loci_quad.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-05-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: quadrature for integrals over arbitrary shape with given (x,y) coordinates
    of the loci of edge points. Taken from FRESNAQ's curveareaareaquad.m (Barnett 2021).


"""

from diffraq.quadrature import lgwt
import numpy as np

def loci_quad(locix, lociy, m):
    """
    xq, yq, wq = loci_quad(locix, lociy, m)

    returns list of nodes and their weights for 2D quadrature over an arbirary
    domain,

        sum_{j=1}^N f(xq(j),yq(j)) wq(j)   \approx   int_Omega f(x,y) dx dy

    for all smooth functions f, where Omega is the domain defined by the boundary
    points (locix, lociy). (Barnett 2021)

    Inputs:
        locix = x-coordinates of boundary points
        lociy = y-coordinates of boundary points
        m = # radial nodes

    Outputs:
        xq, yq = numpy array of x,y coordinates of nodes [meters]
        wq = numpy array of weights
    """

    #Add axis
    locix = locix[:,None]
    lociy = lociy[:,None]

    #Get edge nodes and weights using midpoint rule
    bx = (np.roll(locix, -1) + locix)/2
    by = (np.roll(lociy, -1) + lociy)/2
    wx = np.roll(locix, -1) - locix         #weights are the displacement
    wy = np.roll(lociy, -1) - lociy

    #Radial quadrature nodes, weights
    pr, wr = lgwt(m, 0, 1)

    #Nodes (Eq. 11 Barnett 2021)
    xq = (bx * pr).ravel()
    yq = (by * pr).ravel()

    #Weights (Eq. 12 Barnett 2021)
    wq = (pr * wr * (bx * wy - by * wx)).ravel()

    #Cleanup
    del pr, wr, bx, by, wx, wy

    return xq, yq, wq

############################################
############################################

def loci_edge(locix, lociy, m):
    """
    xy = loci_edge(locix, lociy, m)

    Build loci demarking the starshade edge.

    Inputs:
        locix = x-coordinates of boundary points
        lociy = y-coordinates of boundary points
        m = # radial nodes

    Outputs:
        xy = numpy array (2D) of x,y coordinates of starshade edge [meters]
    """

    return np.stack((locix, lociy), 1)
