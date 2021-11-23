"""
polar_shape.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-08-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Derived class of Shape with outline parameterized in polar coordinates.

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

    def cart_func(self, t, func=None):
        #Grab function if not specified (usually by etch_error)
        if func is None:
            func = self.outline.func(t)
        return func * np.hstack((np.cos(t), np.sin(t)))

    def cart_diff(self, t, func=None, diff=None):
        #Grab function and derivative if not specified (usually by etch_error)
        if func is None or diff is None:
            func = self.outline.func(t)
            diff = self.outline.diff(t)
        ct = np.cos(t)
        st = np.sin(t)
        ans = np.hstack((diff*ct - func*st, diff*st + func*ct))
        del func, diff, ct, st
        return ans

    def cart_diff_2nd(self, t, func=None, diff=None, diff_2nd=None):
        if func is None or diff is None:
            func = self.outline.func(t)
            diff = self.outline.diff(t)
            diff_2nd = self.outline.diff_2nd(t)
        ct = np.cos(t)
        st = np.sin(t)
        ans =  np.hstack((diff_2nd*ct - 2.*diff*st - func*ct,
                          diff_2nd*st + 2.*diff*ct - func*st))
        del func, diff, diff_2nd, ct, st
        return ans

    ############################################

    def cart_func_diffs(self, t, func=None, diff=None, diff_2nd=None, with_2nd=False):
        """Same functions as above, just calculate all at once to save time"""
        if func is None:
            func = self.outline.func(t)
            diff = self.outline.diff(t)
            if with_2nd:
                diff_2nd = self.outline.diff_2nd(t)

        #Calculate intermediaries
        ct = np.cos(t)
        st = np.sin(t)

        #Function
        func_ans = func[:,None] * np.stack((ct, st), 1)

        #Derivative
        diff_ans = np.stack((diff*ct - func*st, diff*st + func*ct),1)

        #Second derivative
        if with_2nd:
            diff_2nd_ans = np.hstack((diff_2nd*ct - 2.*diff*st - func*ct,
                                      diff_2nd*st + 2.*diff*ct - func*st))
            #Cleanup
            del func, diff, ct, st, diff_2nd

            return func_ans, diff_ans, diff_2nd_ans

        else:
            #Cleanup
            del func, diff, ct, st

            return func_ans, diff_ans

    ############################################

    def inv_cart(self, xy):
        #Inverse to go from cartesian to parameter, function
        rad = np.hypot(xy[:,0], xy[:,1])
        the = np.arctan2(xy[:,1], xy[:,0])
        return the, rad

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
