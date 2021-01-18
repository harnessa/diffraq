"""
starshade_quad.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-15-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: quadrature for integrals over area with starshade.
    Taken from FRESNAQ's starshadequad.m (Barnett 2021).

"""

from diffraq.quadrature import lgwt, polar_quad
import numpy as np

def starshade_quad(num_pet, Afunc, r0, r1, m, n):
    """
    xq, yq, wq = starshade_quad(num_pet, Afunc, r0, r1, m, n)

    Uses Theta(r) formula of (1)-(2) in Cady '12 to build area quadrature scheme
    over starshade, given apodization function and other geometric parameters. (Barnett 2021)

    Inputs:
        num_pet = # petals
        Afunc = apodization profile over domain [r0, r1]
        r0, r1 = apodization domain of radii [meters]. r<r0: A=1; r>r1: A=0
        m = # nodes over disc and over radial apodization [r0, r1]
        n = # nodes over petal width

    Outputs:
        xq, yq = numpy array of x,y coordinates of nodes [meters]
        wq = numpy array of weights
    """

    #Central disk radial nodes and weights
    nd = int(np.ceil(0.3*n*num_pet))    #Less in theta, rough guess so dx about same
    xq, yq, wq = polar_quad(lambda t: r0*np.ones_like(t), m, nd)

    ### Build over petals ###

    #Petals width nodes and weights over [-1,1]
    pw, ww = lgwt(n, -1, 1)

    #Petals radius nodes and weights over [0,1]
    pr, wr = lgwt(m, 0, 1)

    #Add axis
    wr = wr[:,None]
    pr = pr[:,None]

    #Scale radius quadrature nodes to physical size
    pr = r0 + (r1 - r0) * pr

    #Apodization value at nodes
    Aval = Afunc(pr)

    #r*dr
    wi = (r1 - r0) * wr * pr

    #thetas
    tt = np.kron(np.ones(num_pet), (np.pi/num_pet) * Aval * pw) + \
         np.kron(2.*np.pi*(np.arange(num_pet) + 1), np.ones(n))

    #Add Petal nodes + weights
    xq = np.concatenate(( xq, (pr * np.cos(tt)).flatten() ))
    yq = np.concatenate(( yq, (pr * np.sin(tt)).flatten() ))
    wq = np.concatenate(( wq, (np.pi/num_pet) * (wi * Aval * ww).flatten() ))

    return xq, yq, wq
