"""
polar_shape_func.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-15-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Class that is the function that describes the shape of a polar
    occulter's edge.

"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from diffraq.geometry import Shape_Function

class Polar_Shape_Func(Shape_Function):

    kind = 'polar'

############################################
#####  Wrappers for Cartesian coordinate systems #####
############################################

    def cart_func(self, t):
        return self.func(t) * np.hstack((np.cos(t), np.sin(t)))

    def cart_diff(self, t):
        return np.hstack((self.diff(t)*np.cos(t) - self.func(t)*np.sin(t), \
                          self.diff(t)*np.sin(t) + self.func(t)*np.cos(t)))

    def cart_diff_2nd(self, t):
        return np.hstack((
            self.diff_2nd(t)*np.cos(t) - 2.*self.diff(t)*np.sin(t) - self.func(t)*np.cos(t),
            self.diff_2nd(t)*np.sin(t) + 2.*self.diff(t)*np.cos(t) - self.func(t)*np.sin(t),
        ))

############################################
############################################
