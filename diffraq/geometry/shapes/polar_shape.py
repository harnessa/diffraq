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
from diffraq.geometry import Shape

class PolarShape(Shape):

    kind = 'polar'

############################################
#####  Main Shape #####
############################################

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
#####  Wrappers for Cartesian coordinate systems #####
############################################

    def cart_func(self, t):
        return self.outline.func(t) * np.hstack((np.cos(t), np.sin(t)))

    def cart_diff(self, t):
        func = self.outline.func(t)
        diff = self.outline.diff(t)
        ct = np.cos(t)
        st = np.sin(t)
        ans = np.hstack((diff*ct - func*st, diff*st + func*ct))
        del func, diff, ct, st
        return ans

    def cart_diff_2nd(self, t):
        func = self.outline.func(t)
        diff = self.outline.diff(t)
        dif2 = self.outline.diff_2nd(t)
        ct = np.cos(t)
        st = np.sin(t)
        ans =  np.hstack((dif2*ct - 2.*diff*st - func*ct,
                          dif2*st + 2.*diff*ct - func*st))
        del func, diff, dif2, ct, st
        return ans

############################################
############################################

############################################
#####  Circle Occulter #####
############################################

class CircleShape(PolarShape):

    def __init__(self, parent, **kwargs):
        #Set radius
        if 'max_radius' in kwargs.keys():
            self.max_radius = kwargs['max_radius']
        else:
            self.max_radius = 12            #Default

        #Replace edge function
        kwargs['edge_func'] = lambda t: self.max_radius * np.ones_like(t)
        kwargs['edge_diff'] = lambda t: np.zeros_like(t)
        kwargs['edge_file'] = None
        kwargs['edge_data'] = None

        #Initiate super
        super().__init__(parent, **kwargs)

############################################
############################################
