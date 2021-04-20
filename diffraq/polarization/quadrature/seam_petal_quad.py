"""
seam_petal_quad.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-15-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: quadrature over seam of edge with radial function.

"""

from diffraq.quadrature import lgwt, polar_quad
import numpy as np

def seam_petal_quad(Afunc, num_pet, r0, r1, m, n, seam_width):
    """
    xq, yq, wq = seam_petal_quad(Afunc, num_pet, r0, r1, m, n, seam_width)

    Uses Theta(r) formula of (1)-(2) in Cady '12 to build area quadrature scheme
    over starshade, given apodization function and other geometric parameters. (Barnett 2021)

    Inputs:
        Afunc = apodization profile over domain [r0, r1]
        num_pet = # petals
        r0, r1 = apodization domain of radii [meters]. r<r0: A=1; r>r1: A=0
        m = # nodes over disc and over radial apodization [r0, r1]
        n = # nodes over petal width
        seam_width = seam half-width [meters]

    Outputs:
        xq, yq = numpy array of x,y coordinates of nodes [meters]
        wq = numpy array of weights
    """

    #Petals width nodes and weights over [0,1]
    pw, ww = lgwt(n, 0, 1)

    #Combine nodes for positive and negative sides of edge
    pw = np.concatenate((pw, -pw[::-1]))
    ww = np.concatenate((ww, ww[::-1]))

    #Petals radius nodes and weights over [0,1]
    pr, wr = lgwt(m, 0, 1)

    #Add axis
    wr = wr[:,None]
    pr = pr[:,None]

    #Scale radius quadrature nodes to physical size
    pr = r0 + (r1 - r0) * pr

    #Apodization value at nodes
    Aval = Afunc(pr)

    #Turn seam_width into angle
    seam_width = seam_width / pr

    #r*dr
    wi = (r1 - r0) * wr * pr

    #Get trailing edge theta
    tt0 = np.pi/num_pet*Aval + pw*seam_width

    #Add leading edge (negative A+pw)
    tt0 = np.hstack((tt0, -tt0))
    pw = np.hstack((pw, pw))
    ww = np.hstack((ww, ww))

    #Rotate theta to other petals
    tt = np.tile(tt0, (1, num_pet)) + \
        np.repeat((2.*np.pi/num_pet) * (np.arange(num_pet) + 1), 4*n) #4n = pos/neg edge + trail/lead

    #Build nodes
    xq = (pr * np.cos(tt)).ravel()
    yq = (pr * np.sin(tt)).ravel()

    #Build Petal weights (rdr * dtheta)
    wq = np.tile(wi * ww * seam_width, (1, num_pet)).ravel()

    #Add theta nodes for all petals for edge distances and normal angles
    pw = np.tile(pw, (1, num_pet))[0]

    #Cleanup
    del ww, Aval, wi, wr, tt, tt0

    #Return nodes along primary axis (theta) and values along orthogonal axis (radius)
    return xq, yq, wq, pw, pr

############################################
############################################

def seam_petal_edge(Afunc, num_pet, r0, r1, m, seam_width):
    """
    xy = seam_petal_edge(Afunc, num_pet, r0, r1, m, n, seam_width)

    Build loci demarking the starshade edge.

    Inputs:
        Afunc = apodization profile over domain [r0, r1]
        num_pet = # petals
        r0, r1 = apodization domain of radii [meters]. r<r0: A=1; r>r1: A=0
        m = # nodes over disc and over radial apodization [r0, r1]
        seam_width = seam half-width [meters]

    Outputs:
        xy = numpy array (2D) of x,y coordinates of starshade edge [meters]
    """
    #Petals radius nodes and weights over [0,1]
    pr, wr = lgwt(m, 0, 1)

    #Scale radius quadrature nodes to physical size
    pr = r0 + (r1 - r0) * pr

    #Add axis
    pr = pr[:,None]

    #Apodization value at nodes
    Aval = Afunc(pr)

    #Get trailing edge theta
    tt0 = (np.pi/num_pet) * (Aval + np.array([1,-1])*seam_width)

    #Add leading edge (negative A+pw)
    tt0 = np.hstack((tt0, -tt0))

    #Rotate theta to other petals
    tt = np.tile(tt0, (1, num_pet)) + \
        np.repeat((2.*np.pi/num_pet) * (np.arange(num_pet) + 1), tt0.shape[-1])

    #Cartesian coords
    xx = (pr * np.cos(tt)).ravel()
    yy = (pr * np.sin(tt)).ravel()

    #Stack
    xy = np.stack((xx, yy),1)

    #Cleanup
    del xx, yy, tt, pr, wr, Aval, tt0

    return xy
