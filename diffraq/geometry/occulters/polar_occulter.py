"""
polar_occulter.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-08-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Derived class of occulter with shape parameterized in polar coordinates.

"""

import numpy as np
import diffraq.quadrature as quad
from diffraq.geometry import Occulter, PolarShapeFunction

class PolarOcculter(Occulter):

    name = 'polar'

############################################
#####  Main Shape #####
############################################

    def set_shape_function(self):
        self.shape_func = PolarShapeFunction(self.sim.apod_func, self.sim.apod_diff)

    def build_shape_quadrature(self):
        #Calculate quadrature
        xq, yq, wq = quad.polar_quad(self.shape_func.func, \
            self.sim.radial_nodes, self.sim.theta_nodes)

        return xq, yq, wq

    def build_shape_edge(self, npts=None):
        #Theta nodes
        if npts is None:
            npts = self.sim.theta_nodes

        #Get polar edge
        edge = quad.polar_edge(self.shape_func.func, npts)

        return edge

############################################
############################################

############################################
#####  Circle Occulter #####
############################################

class CircleOcculter(PolarOcculter):

    def set_shape_function(self):
        func = lambda t: self.sim.circle_rad * np.ones_like(t)
        diff = lambda t: np.zeros_like(t)
        self.shape_func = PolarShapeFunction(func, diff)

############################################
############################################
