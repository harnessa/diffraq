"""
polar_seam.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 03-18-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Braunbek seam to go with PolarShape.

"""

import numpy as np
import diffraq.quadrature as quad

class PolarSeam(object):

    kind = 'polar'

    def __init__(self, shape):
        self.shape = shape

        #TODO: should this inherit a base class? how much is similar to others?

############################################
#####  Main Seam Shape #####
############################################

    def build_seam_quadrature(self, seam_width):
        #Calculate quadrature and get radial nodes and theta values
        xq, yq, wq, pr, pt = quad.seam_polar_quad(self.shape.outline.func, \
            self.shape.radial_nodes, self.shape.theta_nodes, seam_width)

        #Calculate angle between normal and position vector
        func = self.shape.cart_func(pt)
        diff = self.shape.cart_diff(pt)
        angle = (-func[:,0]*diff[:,1] + func[:,1]*diff[:,0]) / \
            (np.hypot(*func.T) * np.hypot(*diff.T))

        #Build edge distances
        dq = seam_width * (pr * angle[:,None]).ravel()

        #Build normal angle
        nq = (np.ones_like(pr) * np.arctan2(diff[:,0], -diff[:,1])[:,None]).ravel()

        #Flip sign of distance and rotate normal angle by pi if opaque
        if self.shape.is_opaque:
            dq *= -1
            nq += np.pi

        #Cleanup
        del func, diff, angle, pr, pt

        return xq, yq, wq, dq, nq

    def build_seam_edge(self, npts=None):
        #Theta nodes
        if npts is None:
            npts = self.shape.theta_nodes

        #Get polar edge
        edge = quad.seam_polar_edge(self.shape.outline.func, npts)

        return edge

############################################
############################################
