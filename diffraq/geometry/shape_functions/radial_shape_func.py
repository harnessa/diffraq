"""
radial_shape_func.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-15-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Class that is the function that describes the shape of a radial (starshade)
    occulter's edge.

"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from diffraq.geometry import Shape_Function

class Radial_Shape_Func(Shape_Function):

    kind = 'radial'

############################################
#####  Wrappers for Cartesian coordinate systems #####
############################################

    def cart_func(self, r):
        func = self.func(r)
        return r * np.hstack((np.cos(func), np.sin(func)))

    def cart_diff(self, r):
        func = self.func(r)
        diff = self.diff(r)
        return np.hstack((np.cos(func) - r*np.sin(func)*diff, \
                          np.sin(func) + r*np.cos(func)*diff))

    def cart_diff_2nd(self, r):
        func = self.func(r)
        diff = self.diff(r)
        dif2 = self.diff_2nd(r)
        return np.hstack(( \
            -np.sin(func)*diff - ((np.sin(func) + r*np.cos(func)*diff)*diff + \
                r*np.sin(func)*dif2), \
             np.cos(func)*diff + ((np.cos(func) - r*np.sin(func)*diff)*diff + \
                r*np.cos(func)*dif2)  ))

############################################
############################################
