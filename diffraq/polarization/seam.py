"""
seam.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 03-18-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Class representing the narrow seam around the edge of an occulter/aperture
    that builds the area quadrature and edge distances allowing the calculation of
    non-scalar diffraction via the Braunbek method.

"""

import numpy as np
import diffraq.quadrature as quad

class Seam(object):

    def __init__(self, shape):
        self.shape = shape

############################################
#####  Build Quadrature and Edge distances #####
############################################

    def build_seam_quadrature(self, seam_width):
        #Get shape specific quadrature and nodes in dominant direction and values in orthogonal direction
        xq, yq, wq, prim_nodes, orth_values = getattr(self, \
            f'get_quad_{self.shape.kind}')(seam_width)

        #Calculate angle between normal and position vector
        func = self.shape.cart_func(orth_values)
        diff = self.shape.cart_diff(orth_values)
        angle = (-func[:,0]*diff[:,1] + func[:,1]*diff[:,0]) / \
            (np.hypot(*func.T) * np.hypot(*diff.T))

        #Build edge distances
        dq = seam_width * (prim_nodes * angle[:,None]).ravel()

        #Build normal angle
        nq = (np.ones_like(prim_nodes) * np.arctan2(diff[:,0], -diff[:,1])[:,None]).ravel()

        #Flip sign of distance and rotate normal angle by pi if opaque
        if self.shape.is_opaque:
            dq *= -1
            nq += np.pi

        #Cleanup
        del func, diff, angle, prim_nodes, orth_values

        return xq, yq, wq, dq, nq

############################################
############################################

############################################
#####  Shape Specific Quadrature #####
############################################

    def get_quad_polar(self, seam_width):
        return quad.seam_polar_quad(self.shape.outline.func, \
            self.shape.radial_nodes, self.shape.theta_nodes, seam_width)

    def get_quad_cartesian(self, seam_width):
        return quad.seam_cartesian_quad(self.shape.outline.func,
            self.shape.outline.diff, self.shape.radial_nodes, \
            self.shape.theta_nodes, seam_width)

    def get_quad_petal(self, seam_width):
        return quad.seam_starshade_quad(self.shape.outline.func, self.shape.num_petals, \
            self.shape.min_radius, self.shape.max_radius, self.shape.radial_nodes, \
            self.shape.theta_nodes, seam_width)

############################################
############################################

############################################
#####  Build Edge Shape #####
############################################

    def build_seam_edge(self, npts=None):
        #TODO: add edge
        breakpoint()
        return edge

############################################
############################################
