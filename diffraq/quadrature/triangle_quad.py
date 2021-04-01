"""
triangle_quad.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 04-01-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: quadrature for integrals over triangluar area.
    Taken from FRESNAQ's triquad.m (Barnett 2021).

"""

from diffraq.quadrature import lgwt
import numpy as np

def triangle_quad(vx, vy, m):
    """
    xq, yq, wq = triangle_quad(vx, vy, m)

    Compute areal quadrature of triange for input vertices

    Inputs:
        vx = list of x-coordinates of CCW ordered vertices of triangle
        vy = list of y-coordinates of CCW ordered vertices of triangle
        m = # nodes

    Outputs:
        xq, yq = numpy array of x,y coordinates of nodes [meters]
        wq = numpy array of weights
    """

    #Get difference between vertices (1st point is base)
    dx = vx[1:] - vx[0]
    dy = vy[1:] - vy[0]

    #Get nodes
    pt, wt = lgwt(m, 0, 1)

    #Build grids
    aa, tt = np.meshgrid(pt, pt)

    #Node locations
    xq = (vx[0] + aa*(dx[0]*(1 - tt) + dx[1]*tt)).ravel()
    yq = (vy[0] + aa*(dy[0]*(1 - tt) + dy[1]*tt)).ravel()

    #Weights
    wq = (dx[0]*(vy[2] - vy[1]) - dy[0]*(vx[2] - vx[1])) * ((pt*wt) * wt[:,None]).ravel()

    #Cleanup
    del pt, wt, aa, tt, dx, dy

    return xq, yq, wq

############################################
############################################

def triangle_edge(vx, vy, m):
    """
    xy = triangle_edge(vx, vy, m)

    Build loci demarking the triangle edge.

    Inputs:
        vx = list of x-coordinates of CCW ordered vertices of triangle
        vy = list of y-coordinates of CCW ordered vertices of triangle
        m = # nodes

    Outputs:
        xy = numpy array (2D) of x,y coordinates of triangle edge [meters]
    """

    #Get difference between vertices (1st point is base)
    dx = vx[1:] - vx[0]
    dy = vy[1:] - vy[0]

    #Get nodes
    pt, wt = lgwt(m, 0, 1)

    #Add nodes from base
    xe = (vx[0] + pt*dx[:,None]).ravel()
    ye = (vy[0] + pt*dy[:,None]).ravel()

    #Add third leg
    xe = np.concatenate((xe, vx[1] + pt*(vx[2]-vx[1])))
    ye = np.concatenate((ye, vy[1] + pt*(vy[2]-vy[1])))

    #Stack
    edge = np.stack((xe, ye), 1)

    #Cleanup
    del dx, dy, pt, wt, xe, ye

    return edge
