"""
cartesian_shape.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-08-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Derived class of Shape with outline parameterized in cartesian coordinates.

"""

import numpy as np
import diffraq.quadrature as quad
from diffraq.geometry import Shape

class CartesianShape(Shape):

    kind = 'cartesian'

############################################
#####  Main Shape #####
############################################

    def build_local_shape_quad(self):
        #Calculate quadrature
        xq, yq, wq = quad.cartesian_quad(self.outline.func, self.outline.diff, \
            self.radial_nodes, self.theta_nodes)

        return xq, yq, wq

    def build_local_shape_edge(self, npts=None):
        #Theta nodes
        if npts is None:
            npts = self.theta_nodes

        #Get cartesian edge
        edge = quad.cartesian_edge(self.outline.func, npts)

        return edge

############################################
############################################

############################################
#####  Wrappers for Cartesian coordinate systems #####
############################################

    def cart_func(self, t):
        return self.outline.func(t)

    def cart_diff(self, t):
        return self.outline.diff(t)

    def cart_diff_2nd(self, t):
        return self.outline.diff_2nd(t)

    ############################################

    def cart_func_diffs(self, t, func=None, diff=None, diff_2nd=None, with_2nd=False):
        """Same functions as above, just calculate all at once to save time"""

        func_ans = self.outline.func(t)
        diff_ans = self.outline.diff(t)

        if with_2nd:
            diff_2nd_ans = self.outline.diff_2nd(t)
            return func_ans, diff_ans, diff_2nd_ans
        else:
            return func_ans, diff_ans

############################################
############################################
