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
        #Get shape specific quadrature and nodes in dependent direction (r-polar, theta-petal) and
            #values in independent, i.e. parameter, direction (theta-polar, r-petal)
        xq, yq, wq, dept_nodes, indt_values = getattr(self, \
            f'get_quad_{self.shape.kind}')(seam_width)

        #Get normal and position angles depending on shape
        if self.shape.kind == 'petal':
            pos_angle, nq = self.get_normal_angles_petal(indt_values)
        else:
            pos_angle, nq = self.get_normal_angles_polar(indt_values)

        #Build edge distances
        dq = seam_width * (dept_nodes * pos_angle).ravel()

        #Flip sign of distance and rotate normal angle by pi if opaque
        if self.shape.is_opaque:
            dq *= -1
            nq += np.pi

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
        return quad.seam_petal_quad(self.shape.outline.func, self.shape.num_petals, \
            self.shape.min_radius, self.shape.max_radius, self.shape.radial_nodes, \
            self.shape.theta_nodes, seam_width)

############################################
############################################

############################################
#####  Shape Specific Normals + Distances #####
############################################

    def get_normal_angles_petal(self, indt_values):
        #We need to pack the indt_values into a parameter, to account for petals
        pets = np.roll(np.arange(self.shape.num_petals) + 1, -1)
        sgns = np.stack((np.ones(self.shape.theta_nodes), \
                        -np.ones(self.shape.theta_nodes)), 1)[...,None]
        petals = (pets*sgns).T.ravel()
        indt_values = self.shape.pack_param(indt_values, petals)[...,None]

        #Get function and derivative values at the parameter values
        func = self.shape.cart_func(indt_values)
        diff = self.shape.cart_diff(indt_values)
        diff_2nd = self.shape.cart_diff_2nd(indt_values)

        #Calculate angle between normal and theta vector (orthogonal to position vector)
        pos_angle = -(func[...,0]*diff[...,0] + func[...,1]*diff[...,1]) / \
            (np.hypot(func[...,0], func[...,1]) * np.hypot(diff[...,0], diff[...,1]))

        #Build normal angle (get sign from second derivatives)
        beta = np.sign(diff[...,0]*diff_2nd[...,1] - diff[...,1]*diff_2nd[...,0])
        nq = np.arctan2(beta*diff[...,0], -beta*diff[...,1]).ravel()

        #Cleanup
        del pets, sgns, petals, indt_values, func, diff, diff_2nd, beta

        return pos_angle, nq

    def get_normal_angles_polar(self, indt_values):
        #Get function and derivative values at the parameter values
        func = self.shape.cart_func(indt_values)
        diff = self.shape.cart_diff(indt_values)

        #Calculate angle between normal and radius vector (position vector)
        pos_angle = ((-func[:,0]*diff[:,1] + func[:,1]*diff[:,0]) / \
            (np.hypot(*func.T) * np.hypot(*diff.T)))[:,None]

        #Build normal angles
        nq = (np.ones(self.shape.radial_nodes) * \
            np.arctan2(diff[:,0], -diff[:,1])[:,None]).ravel()

        #Cleanup
        del func, diff

        return pos_angle, nq

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
