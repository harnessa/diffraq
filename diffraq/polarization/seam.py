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
import diffraq.polarization as polar

class Seam(object):

    def __init__(self, shape, radial_nodes, theta_nodes):
        self.shape = shape
        self.radial_nodes = radial_nodes
        self.theta_nodes = theta_nodes

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

        #Cleanup
        del dept_nodes, pos_angle

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
        return polar.seam_polar_quad(self.shape.outline.func, \
            self.radial_nodes, self.theta_nodes, seam_width)

    def get_quad_cartesian(self, seam_width):
        return polar.seam_cartesian_quad(self.shape.outline.func,
            self.shape.outline.diff, self.radial_nodes, \
            self.theta_nodes, seam_width)

    def get_quad_petal(self, seam_width):
        return polar.seam_petal_quad(self.shape.outline.func, self.shape.num_petals, \
            self.shape.min_radius, self.shape.max_radius, self.radial_nodes, \
            self.theta_nodes, seam_width)

############################################
############################################

############################################
#####  Shape Specific Normals + Distances #####
############################################

    def get_normal_angles_petal(self, indt_values):

        #Get petal signs and angle to rotate
        ones = np.ones(self.theta_nodes, dtype=int)
        pet_mul = np.tile(np.concatenate((ones, -ones)), self.shape.num_petals)
        pet_add = 2*(np.repeat(np.roll(np.arange(self.shape.num_petals) + 1, -1), \
            2*self.theta_nodes) - 1)

        #Get function and derivative values at the parameter values
        func = self.shape.outline.func(indt_values)*pet_mul + pet_add
        diff = self.shape.outline.diff(indt_values)*pet_mul

        #Get cartesian function and derivative values at the parameter values
        cart_func, cart_diff = self.shape.cart_func_diffs( \
            indt_values, func=func, diff=diff)

        #Cleanup
        del func, diff, pet_add, ones

        #Calculate angle between normal and theta vector (orthogonal to position vector)
        pos_angle = -(cart_func[...,0]*cart_diff[...,0] + cart_func[...,1]*cart_diff[...,1]) / \
            (np.hypot(cart_func[...,0], cart_func[...,1]) * np.hypot(cart_diff[...,0], cart_diff[...,1]))

        #Build normal angle
        nq = np.arctan2(pet_mul*cart_diff[...,0], -pet_mul*cart_diff[...,1]).ravel()

        #Cleanup
        del indt_values, cart_func, cart_diff, pet_mul

        return pos_angle, nq

    def get_normal_angles_polar(self, indt_values):
        #Get function and derivative values at the parameter values
        func, diff = self.shape.cart_func_diffs(indt_values)

        #Calculate angle between normal and radius vector (position vector)
        pos_angle = ((-func[:,0]*diff[:,1] + func[:,1]*diff[:,0]) / \
            (np.hypot(func[:,0],func[:,1]) * np.hypot(diff[:,0],diff[:,1])))[:,None]

        #Build normal angles
        nq = (np.ones(self.radial_nodes) * \
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
