"""
cartesian_shape.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-08-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Derived class of occulter with shape parameterized in cartesian coordinates.

"""

import numpy as np
import diffraq.quadrature as quad
from diffraq.geometry import Shape, CartesianOutline

class CartesianShape(Shape):

############################################
#####  Main Shape #####
############################################

    def set_outline(self):
        self.outline = CartesianOutline(self.edge_func, self.edge_diff)

    def build_shape_quadrature(self):
        #Calculate quadrature
        xq, yq, wq = quad.cartesian_quad(self.outline.func, self.outline.diff, \
            self.radial_nodes, self.theta_nodes)

        return xq, yq, wq

    def build_shape_edge(self, npts=None):
        #Theta nodes
        if npts is None:
            npts = self.theta_nodes

        #Get cartesian edge
        edge = quad.cartesian_edge(self.outline.func, npts)

        return edge

############################################
############################################
