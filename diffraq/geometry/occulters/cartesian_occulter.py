"""
cartesian_occulter.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-08-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Derived class of occulter with shape parameterized in cartesian coordinates.

"""

import numpy as np
import diffraq.quadrature as quad
from diffraq.geometry import Occulter, CartesianShapeFunction

class CartesianOcculter(Occulter):

    name = 'cartesian'

############################################
#####  Main Shape #####
############################################

    def set_shape_function(self):
        self.shape_func = CartesianShapeFunction(self.sim.apod_func, self.sim.apod_diff)

    def build_shape_quadrature(self):
        #Calculate quadrature
        xq, yq, wq = quad.cartesian_quad(self.shape_func.func, self.shape_func.diff, \
            self.sim.radial_nodes, self.sim.theta_nodes)

        return xq, yq, wq

    def build_edge(self, npts=None):
        #Theta nodes
        if npts is None:
            npts = self.sim.theta_nodes

        #Get cartesian edge
        edge = quad.cartesian_edge(self.shape_func.func, npts)

        return edge

############################################
############################################
