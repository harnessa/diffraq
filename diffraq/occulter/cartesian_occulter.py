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
from diffraq.occulter import Occulter, Shape_Function

class Cartesian_Occulter(Occulter):

    name = 'cartesian'

############################################
#####  Main Shape #####
############################################

    def set_shape_function(self):
        func =  np.atleast_1d(self.sim.apod_func)
        deriv = np.atleast_1d(self.sim.apod_deriv)
        self.shape_func = [Shape_Function(func[i], deriv[i]) for i in range(len(func))]

    def build_shape_quadrature(self):
        #Calculate quadrature
        xq, yq, wq = quad.cartesian_quad(self.shape_func[0].func, self.shape_func[1].func,\
            self.shape_func[0].deriv, self.shape_func[1].deriv, \
            self.sim.radial_nodes, self.sim.theta_nodes)

        return xq, yq, wq

    def build_edge(self, npts=None):
        #Theta nodes
        if npts is None:
            npts = self.sim.theta_nodes

        #Get cartesian edge
        edge = quad.cartesian_edge(*self.shape_func.func, npts)

        return edge

############################################
############################################

############################################
#####  Defects #####
############################################

    def build_defects_quadrature(self):

        breakpoint()

############################################
############################################
