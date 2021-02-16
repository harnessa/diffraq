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
from diffraq.geometry import Shape, PolarOutline

class PolarShape(Shape):

############################################
#####  Main Shape #####
############################################

    def set_outline(self):
        self.outline = PolarOutline(self.edge_func, self.edge_diff)

    def build_local_shape_quad(self):
        #Calculate quadrature
        xq, yq, wq = quad.polar_quad(self.outline.func, \
            self.radial_nodes, self.theta_nodes)

        return xq, yq, wq

    def build_local_shape_edge(self, npts=None):
        #Theta nodes
        if npts is None:
            npts = self.theta_nodes

        #Get polar edge
        edge = quad.polar_edge(self.outline.func, npts)

        return edge

############################################
############################################

############################################
#####  Circle Occulter #####
############################################

class CircleShape(PolarShape):

    def set_outline(self):
        func = lambda t: self.max_radius * np.ones_like(t)
        diff = lambda t: np.zeros_like(t)
        self.outline = PolarOutline(func, diff)

############################################
############################################
