"""
unique_petal_quad.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 06-02-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: quadrature for integrals over area with petalized radial function (starshade).
    Taken from FRESNAQ's starshadequad.m (Barnett 2021).

"""

from diffraq.quadrature import lgwt, polar_quad
import numpy as np

def unique_petal_quad(Afunc, edge_keys, num_pet, r0, r1, m, n, has_center=True):
    """
    xq, yq, wq = unique_petal_quad(Afunc, edge_keys, num_pet, r0, r1, m, n)

    Uses Theta(r) formula of (1)-(2) in Cady '12 to build area quadrature scheme
    over starshade, given apodization function and other geometric parameters. (Barnett 2021)

    Inputs:
        Afunc = apodization profile over domain [r0, r1]
        edge_keys = array of length 2*num_pet matching each edge that holds the index to that edges' Afunc
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
    #Petals width nodes and weights over [0,1]
    pw, ww = lgwt(n//2, 0, 1)

    #Petals radius nodes and weights over [0,1]
    pr0, wr0 = lgwt(m, 0, 1)

    ### Build Disk ###

    #Central disk radial nodes and weights
    if has_center:
        nd = int(np.ceil(0.3*n*num_pet))    #Less in theta
        xq, yq, wq = polar_quad(lambda t: r0*np.ones_like(t), m, nd, pr=pr0, wr=wr0)
    else:
        xq, yq, wq = np.array([]), np.array([]), np.array([])

    ### Build petals ###

    #Add axis
    wr0 = wr0[:,None]
    pr0 = pr0[:,None]

    #Loop through each edge combo and build
    for ic in range(len(Afunc)):

        #Get edge keys that match
        kinds = np.where(edge_keys == ic)[0]

        #Scale radius quadrature nodes to physical size
        pr = r0[ic] + (r1 - r0)[ic] * pr0

        #Apodization value at nodes and weights
        Aval = (np.pi/num_pet)*Afunc[ic](pr)
        Apw = np.tile(Aval*pw, (1, len(kinds)))
        Aww = np.tile(Aval*ww, (1, len(kinds)))

        #Flip odd edges to negative dtheta
        Apw *= np.repeat(np.array([-1,1])[kinds%2], n//2)

        #Theta
        tt = Apw + np.repeat(2.*np.pi/num_pet * (kinds//2), n//2)

        #r*dr
        wi = (r1 - r0)[ic] * wr0 * pr

        #Add Petal weights (rdr * dtheta)
        wq = np.concatenate(( wq, (wi * Aww).ravel() ))

        #Add Petal nodes
        xq = np.concatenate(( xq, (pr * np.cos(tt)).ravel() ))
        yq = np.concatenate(( yq, (pr * np.sin(tt)).ravel() ))

    #Cleanup
    del pw, ww, pr0, wr0, kinds, pr, wi, Aval, Apw, Aww, tt

    return xq, yq, wq

############################################
############################################

def unique_petal_edge(Afunc, edge_keys, num_pet, r0, r1, m):
    """
    xy = petal_edge(Afunc, edge_keys, num_pet, r0, r1, m)

    Build loci demarking the starshade edge.

    Inputs:
        Afunc = apodization profile over domain [r0, r1]
        edge_keys = array of length 2*num_pet matching each edge that holds the index to that edges' Afunc
        num_pet = # petals
        r0, r1 = apodization domain of radii [meters]. r<r0: A=1; r>r1: A=0
        m = # nodes over disc and over radial apodization [r0, r1]

    Outputs:
        xy = numpy array (2D) of x,y coordinates of starshade edge [meters]
    """

    #Petals radius nodes and weights over [0,1]
    pr0, wr0 = lgwt(m, 0, 1)

    #Add axis
    wr0 = wr0[:,None]
    pr0 = pr0[:,None]

    #Loop through each edge combo and build
    xx, yy = np.empty(0), np.empty(0)
    for ic in range(len(Afunc)):

        #Get edge keys that match
        kinds = np.where(edge_keys == ic)[0]

        #Scale radius quadrature nodes to physical size
        pr = r0[ic] + (r1 - r0)[ic] * pr0

        #r*dr
        wi = (r1 - r0)[ic] * wr0 * pr

        #Apodization value at nodes and weights
        Aval = (np.pi/num_pet)*Afunc[ic](pr)
        Apw = np.tile(Aval, (1, len(kinds)))

        #Flip odd edges to negative dtheta
        Apw *= np.array([-1,1])[kinds%2]

        tt = Apw + 2.*np.pi/num_pet * (kinds//2)

        #Cartesian coords
        xx = np.concatenate(( xx, (pr * np.cos(tt)).ravel() ))
        yy = np.concatenate(( yy, (pr * np.sin(tt)).ravel() ))

    #Stack
    xy = np.stack((xx, yy),1)

    #Cleanup
    del xx, yy, tt, pr, Aval, Apw, pr0, wr0, kinds

    return xy
