"""
petal_quad.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-15-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: quadrature for integrals over area with petalized radial function (starshade).
    Taken from FRESNAQ's starshadequad.m (Barnett 2021).

"""

from diffraq.quadrature import lgwt, polar_quad
import numpy as np

def petal_quad(Afunc, num_pet, r0, r1, m, n, has_center=True):
    """
    xq, yq, wq = petal_quad(Afunc, num_pet, r0, r1, m, n)

    Uses Theta(r) formula of (1)-(2) in Cady '12 to build area quadrature scheme
    over starshade, given apodization function and other geometric parameters. (Barnett 2021)

    Inputs:
        Afunc = apodization profile over domain [r0, r1]
        num_pet = # petals
        r0, r1 = apodization domain of radii [meters]. r<r0: A=1; r>r1: A=0
        m = # nodes over disc and over radial apodization [r0, r1]
        n = # nodes over petal width
        has_center = does the starshade have an opaque central disc? i.e. (common)

    Outputs:
        xq, yq = numpy array of x,y coordinates of nodes [meters]
        wq = numpy array of weights
    """

    #Build nodes
    #Petals width nodes and weights over [-1,1]
    pw, ww = lgwt(n, -1, 1)

    #Petals radius nodes and weights over [0,1]
    pr, wr = lgwt(m, 0, 1)

    ### Build Disk ###

    #Central disk radial nodes and weights
    if has_center:
        nd = int(np.ceil(0.3*n*num_pet))    #Less in theta
        xq, yq, wq = polar_quad(lambda t: r0*np.ones_like(t), m, nd, pr=pr, wr=wr)
    else:
        xq, yq, wq = np.array([]), np.array([]), np.array([])

    ### Build petals ###

    #Add axis
    wr = wr[:,None]
    pr = pr[:,None]

    #Scale radius quadrature nodes to physical size
    pr = r0 + (r1 - r0) * pr

    #Apodization value at nodes and weights
    if not isinstance(Afunc, list):
        Aval = (np.pi/num_pet) * Afunc(pr)
        Apw = np.tile(Aval*pw, (1, num_pet))
        Aww = np.tile(Aval*ww, (1, num_pet))
    else:
        Aval = [(np.pi/num_pet)*af(pr) for af in Afunc]
        Apw = np.hstack(([Av*pw for Av in Aval]))
        Aww = np.hstack(([Av*ww for Av in Aval]))

    #Cleanup
    del Aval, pw, ww

    #r*dr
    wi = (r1 - r0) * wr * pr

    #Add Petal weights (rdr * dtheta)
    wq = np.concatenate(( wq, (wi * Aww).ravel() ))

    #Cleanup
    del Aww, wi

    #thetas
    tt = Apw + np.repeat((2.*np.pi/num_pet) * (np.arange(num_pet) + 1), n)

    #Add Petal nodes
    xq = np.concatenate(( xq, (pr * np.cos(tt)).ravel() ))
    yq = np.concatenate(( yq, (pr * np.sin(tt)).ravel() ))

    #Cleanup
    del Apw, pr, wr, tt

    return xq, yq, wq

############################################
############################################

def petal_edge(Afunc, num_pet, r0, r1, m):
    """
    xy = petal_edge(Afunc, num_pet, r0, r1, m)

    Build loci demarking the starshade edge.

    Inputs:
        Afunc = apodization profile over domain [r0, r1]
        num_pet = # petals
        r0, r1 = apodization domain of radii [meters]. r<r0: A=1; r>r1: A=0
        m = # nodes over disc and over radial apodization [r0, r1]

    Outputs:
        xy = numpy array (2D) of x,y coordinates of starshade edge [meters]
    """

    #Petals radius nodes and weights over [0,1]
    pr, wr = lgwt(m, 0, 1)

    #Scale radius quadrature nodes to physical size
    pr = r0 + (r1 - r0) * pr

    #Add axis
    pr = pr[:,None]

    #Theta nodes (only at edges on each side)
    pw = np.array([1, -1])

    #Apodization value at nodes and weights
    if not isinstance(Afunc, list):
        Aval = (np.pi/num_pet) * Afunc(pr)
        Apw = np.tile(Aval*pw, (1, num_pet))
    else:
        Aval = [(np.pi/num_pet)*af(pr) for af in Afunc]
        Apw = np.hstack(([Av*pw for Av in Aval]))

    #thetas
    tt = Apw + np.repeat((2.*np.pi/num_pet) * (np.arange(num_pet) + 1), 2)

    #Cartesian coords
    xx = (pr * np.cos(tt)).ravel()
    yy = (pr * np.sin(tt)).ravel()

    #Stack
    xy = np.stack((xx, yy),1)

    #Cleanup
    del xx, yy, tt, pr, wr, Aval, Apw

    return xy
