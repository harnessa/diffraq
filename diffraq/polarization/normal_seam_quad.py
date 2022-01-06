"""
normal_seam_quad.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 12-09-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Script to calculate seam around petal occulter, where quadrature
    is calculated relative to the normal of the edge.

"""

import numpy as np
from diffraq.quadrature import lgwt, polar_quad
from scipy.interpolate import InterpolatedUnivariateSpline

def build_normal_quadrature(shape, seam_width, radial_nodes, theta_nodes):

    #Copy over parameters
    num_pet = shape.num_petals

    #Petals width nodes and weights over [0,1]
    pw, ww = lgwt(theta_nodes, 0, 1)

    #Combine nodes for positive and negative sides of edge (negative pw is positive side)
    pw = np.concatenate((pw, -pw[::-1]))
    ww = np.concatenate((ww,  ww[::-1]))

    #Petals radius nodes and weights over [0,1]
    pr0, wr0 = lgwt(radial_nodes, 0, 1)

    #Add axis
    wr0 = wr0[:,None]
    pr0 = pr0[:,None]

    #############################

    #Initiate
    xq, yq, wq = np.array([]), np.array([]), np.array([])
    nq, dq, gq = np.array([]), np.array([]), np.array([])

    #Loop over half petals
    for ip in range(2*num_pet):

        #Get current edge properties
        ekey = shape.edge_keys[ip]
        r0 = shape.min_radius[ekey]
        r1 = shape.max_radius[ekey]
        Afunc = shape.outline.func[ekey]
        Adiff = shape.outline.diff[ekey]

        #############################

        #Scale radius quadrature nodes to physical size
        pr = r0 + (r1 - r0) * pr0

        #Get rotation and side (trailing/leading) of petal
        pet_sgn = [-1,1][ip%2]
        pet_mul = np.pi/num_pet * pet_sgn
        pet_add = (2*np.pi/num_pet) * (ip//2)

        #Get theta value of edge
        Aval = Afunc(pr)
        tt = Aval * pet_mul + pet_add

        #Get cartesian coordinates of edge
        exx = pr * np.cos(tt)
        eyy = pr * np.sin(tt)

        #Get cartesian derivative of edge
        pdiff = Adiff(pr) * pet_mul
        edx = np.cos(tt) - eyy*pdiff
        edy = np.sin(tt) + exx*pdiff

        #Build unit normal vectors (remember to switch x/y and make new x negative)
        evx = -edy / np.hypot(edx, edy)
        evy =  edx / np.hypot(edx, edy)

        #Get edge distance
        cd = np.ones_like(evx) * pw*seam_width * -pet_sgn

        #Build coordinates in seam
        cx = exx + evx*pw*seam_width
        cy = eyy + evy*pw*seam_width

        #Build normal angle (all get the same on given line)
        cn = np.ones_like(pw) * np.arctan2(pet_mul*edx, -pet_mul*edy)

        #Calculate cos(angle) between normal and theta vector (orthogonal to position vector) at edge
        pos_angle = -(exx*edx + eyy*edy) / (np.hypot(exx, eyy) * np.hypot(edx, edy))

        #dtheta
        wthe = np.abs(ww * seam_width/pr * pos_angle)

        #r*dr
        wrdr = (r1 - r0) * wr0 * pr

        #FIXME: weights are wrong!! b/c no longer polar integration
        breakpoint()
        #Weights (rdr * dtheta)
        cw = wrdr * wthe

        #Get gap widths
        cg = 2*Aval*abs(pet_mul)*pr

        #Find overlap (we only want to stop overlap on opposing screen, but we want data in gaps to add)
        ovr_inds = cd > cg

        #Zero out weights on overlap
        cw[ovr_inds] = 0

        #Append
        xq = np.concatenate((xq, cx.ravel()))
        yq = np.concatenate((yq, cy.ravel()))
        wq = np.concatenate((wq, cw.ravel()))
        nq = np.concatenate((nq, cn.ravel()))
        dq = np.concatenate((dq, cd.ravel()))
        gq = np.concatenate((gq, cg.ravel()))

    #Cleanup
    del pw, ww, pr0, wr0, pr, Aval, tt, exx, eyy, pdiff, edx, edy, evx, evy, \
        pos_angle, wthe, wrdr, ovr_inds, cx, cy, cw, cn, cd, cg

    return xq, yq, wq, dq, nq, gq
