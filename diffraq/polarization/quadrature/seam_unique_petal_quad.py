"""
seam_unique_petal_quad.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-15-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: quadrature over seam of edge with radial function.

"""

from diffraq.quadrature import lgwt, polar_quad
import numpy as np

def seam_unique_petal_quad(Afunc, edge_keys, num_pet, r0, r1, m, n, seam_width):
    """
    xq, yq, wq = seam_petal_quad(Afunc, num_pet, r0, r1, m, n, seam_width)

    Uses Theta(r) formula of (1)-(2) in Cady '12 to build area quadrature scheme
    over starshade, given apodization function and other geometric parameters. (Barnett 2021)

    Inputs:
        Afunc = apodization profile over domain [r0, r1]
        edge_keys = array of length 2*num_pet matching each edge that holds the index to that edges' Afunc
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
    ww = np.concatenate((ww,  ww[::-1]))

    #Petals radius nodes and weights over [0,1]
    pr0, wr0 = lgwt(m, 0, 1)

    #Add axis
    wr0 = wr0[:,None]
    pr0 = pr0[:,None]

    #Initiate
    xq, yq, wq = np.array([]), np.array([]), np.array([])

    #Also radii for each function
    all_pr = []

    #Loop through each edge combo and build
    for ic in range(len(Afunc)):

        #Get edge keys that match
        kinds = np.where(edge_keys == ic)[0]

        #Scale radius quadrature nodes to physical size
        pr = r0[ic] + (r1 - r0)[ic] * pr0

        #Turn seam_width into angle
        seam_width_angle = seam_width / pr

        #Apodization value at nodes and weights for all petals
        tt = np.tile((np.pi/num_pet)*Afunc[ic](pr) + pw*seam_width_angle, \
            (1, len(kinds)))

        #Flip odd edges to negative theta
        tt *= np.repeat(np.array([-1,1])[kinds%2], 2*n)

        #Rotate to other petals
        tt += np.repeat((2.*np.pi/num_pet) * (kinds//2), 2*n)

        #r*dr
        wi = (r1 - r0)[ic] * wr0 * pr

        #rdr * dtheta
        wgt = np.tile(wi * ww * seam_width_angle, (1, len(kinds)))

        # breakpoint()
        #Add Petal weights (rdr * dtheta)
        wq = np.concatenate(( wq, wgt.ravel() ))

        #Add Petal nodes
        xq = np.concatenate(( xq, (pr * np.cos(tt)).ravel() ))
        yq = np.concatenate(( yq, (pr * np.sin(tt)).ravel() ))

        #Append
        all_pr.append(pr)

    #Add theta nodes for all petals for edge distances and normal angles
    pw = np.tile(pw, 2*num_pet)

    #Cleanup
    del ww, wr0, wi, tt, kinds, seam_width_angle, wgt, pr

    #Return nodes along primary axis (theta) and values along orthogonal axis (radius)
    return xq, yq, wq, pw, all_pr

############################################
############################################

def seam_unique_petal_edge(Afunc, edge_keys, num_pet, r0, r1, m, seam_width):
    """
    xy = seam_petal_edge(Afunc, num_pet, r0, r1, m, n, seam_width)

    Build loci demarking the starshade edge.

    Inputs:
        Afunc = apodization profile over domain [r0, r1]
        edge_keys = array of length 2*num_pet matching each edge that holds the index to that edges' Afunc
        num_pet = # petals
        r0, r1 = apodization domain of radii [meters]. r<r0: A=1; r>r1: A=0
        m = # nodes over disc and over radial apodization [r0, r1]
        seam_width = seam half-width [meters]

    Outputs:
        xy = numpy array (2D) of x,y coordinates of starshade edge [meters]
    """

    #FIXME: seams not tested
    # #Petals radius nodes and weights over [0,1]
    # pr0, wr0 = lgwt(m, 0, 1)
    #
    # #Add axis
    # wr0 = wr0[:,None]
    # pr0 = pr0[:,None]
    #
    # #Loop through each edge combo and build
    # xx, yy = np.empty(0), np.empty(0)
    # for ic in range(len(Afunc)):
    #
    #     #Get edge keys that match
    #     kinds = np.where(edge_keys == ic)[0]
    #
    #     #Scale radius quadrature nodes to physical size
    #     pr = r0[ic] + (r1 - r0)[ic] * pr0
    #
    #     #Turn seam_width into angle
    #     seam_width_angle = seam_width / pr
    #
    #     #Apodization value at nodes and weights for all petals
    #     # Aval = (np.pi/num_pet)*Afunc[ic](pr) + np.array()
    #     tt = np.tile((np.pi/num_pet)*Afunc[ic](pr) + pw*seam_width_angle, \
    #         (1, len(kinds)))
    #
    #     #Flip odd edges to negative theta
    #     tt *= np.repeat(np.array([-1,1])[kinds%2], 2*n)
    #
    #     #Rotate to other petals
    #     tt += np.repeat((2.*np.pi/num_pet) * (kinds//2), 2*n)
    #
    #     #r*dr
    #     wi = (r1 - r0)[ic] * wr0 * pr
    #
    #     #rdr * dtheta
    #     wgt = np.tile(wi * ww * seam_width_angle, (1, len(kinds)))
    #
    #     # breakpoint()
    #     #Add Petal weights (rdr * dtheta)
    #     wq = np.concatenate(( wq, wgt.ravel() ))
    #
    #     #Add Petal nodes
    #     xq = np.concatenate(( xq, (pr * np.cos(tt)).ravel() ))
    #     yq = np.concatenate(( yq, (pr * np.sin(tt)).ravel() ))
    #
    #     #Append
    #     all_pr.append(pr)
    #
    # #Add theta nodes for all petals for edge distances and normal angles
    # pw = np.tile(pw, 2*num_pet)

    return xy
