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

    def __init__(self, parent, shape):
        self.parent = parent    #Seam
        self.shape = shape

        #TODO: should this inherit a base class? how much is similar to others?

############################################
#####  Main Seam Shape #####
############################################

    def build_seam_quadrature(self):
        #Calculate quadrature and get radial nodes and theta values
        xq, yq, wq, pr, pt = quad.seam_polar_quad(self.shape.outline.func, \
            self.shape.radial_nodes, self.shape.theta_nodes, self.parent.seam_width)

        #Calculate angle between normal and position vector
        func = self.shape.cart_func(pt)
        diff = self.shape.cart_diff(pt)
        angle = (-func[:,0]*diff[:,1] + func[:,1]*diff[:,0]) / \
            (np.hypot(*func.T) * np.hypot(*diff.T))

        #Build edge distances
        dq = self.parent.seam_width * pr * angle[:,None]

        #Build normal angle
        nq = np.arctan2(diff[:,0], -diff[:,1])

        #Cleanup
        del func, diff, angle, pr, pt

        return xq, yq, wq, dq, nq

    def build_seam_edge(self, npts=None):
        #Theta nodes
        if npts is None:
            npts = self.theta_nodes

        #Get polar edge
        edge = quad.seam_polar_edge(self.shape.outline.func, npts)

        return edge

############################################
############################################
