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

def seam_cartesian_quad(fxy, dxy, m, n, seam_width):
    """
    xq, yq, wq = seam_cartesian_quad(fxy, dxy, m, n, seam_width)

    returns list of nodes and their weights for 2D quadrature, over a cartesian domain,

        sum_{j=1}^N f(xq(j),yq(j)) wq(j)   \approx   int_Omega f(x,y) dx dy

    for all smooth functions f, where Omega is the cartesian domain defined by
    function (x,y) = fxy(theta). (Barnett 2021)

    Inputs:
        fxy = function handle for (x,y) = fxy(theta), theta in [0,2pi)
        dxy = function handle for derivative of fxy, theta in [0,2pi)
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

    #Cartesian quadrature nodes, weights
    pr, wr = lgwt(m, -1, 1)

    #Get function values at all theta nodes
    ft = fxy(pt)[:,:,None]
    dt = dxy(pt)[:,:,None]

    #Get normal angle at all theta nodes (flipped negative to point inward to match seam_polar_quad)
    norm = np.hypot(*dt[...,0].T)
    nx =  dt[:,1,0] / norm
    ny = -dt[:,0,0] / norm

    #Nodes (Eq. 11 Barnett 2021)
    xq = ft[:,0] + nx[:,None]*pr*seam_width
    yq = ft[:,1] + ny[:,None]*pr*seam_width

    #Weights (Eq. 12 Barnett 2021)
    wq = wt * seam_width * (wr * (xq * dt[:,1] - yq * dt[:,0])).ravel()

    #Ravel
    xq = xq.ravel()
    yq = yq.ravel()

    #Cleanup
    del wr, ft, dt, norm, nx, ny

    return xq, yq, wq, pr, pt

############################################
############################################

def seam_cartesian_edge(fxy, n, seam_width):
    """
    xy = seam_cartesian_edge(fxy, n, seam_width)

    Build loci demarking the cartesian occulter edge.

    Inputs:
        fxy = function handle for (x,y) = fxy(theta), theta in [0,2pi)
        n = # theta nodes
        seam_width = seam half-width [meters]

    Outputs:
        xy = numpy array (2D) of x,y coordinates of occulter edge [meters]
    """

    #Theta nodes
    pt = 2.*np.pi/n * (np.arange(n) + 1)

    #Get function and derivative values
    edge = fxy(pt)
    diff = dxy(pt)

    #Get normal angle at all theta nodes (flipped negative to point inward to match seam_polar_quad)
    norm = np.hypot(*diff.T)
    nx =  diff[:,1] / norm
    ny = -diff[:,0] / norm

    #Add seam widths with normals
    xx = (edge[:,0][:,None] + seam_width*np.array([1, -1])*nx[:,None]).ravel()
    yy = (edge[:,1][:,None] + seam_width*np.array([1, -1])*ny[:,None]).ravel()
    xy = np.stack((xx, yy), 1)

    #Cleanup
    del edge, diff, norm, nx, ny, xx, yy

    return xy
