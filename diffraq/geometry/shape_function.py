"""
shape_function.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-15-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Class that is the function that describes the shape of the occulter's edge.

"""

import numpy as np
from scipy.misc import derivative
from scipy.interpolate import InterpolatedUnivariateSpline

class Shape_Function(object):

    def __init__(self, func, diff=None, diff_2nd=-1):

        #Order with second derivative (diff_2nd == -1: won't calculate 2nd diff)
        k = [2, 1][diff_2nd == -1]

        #Set function
        self.func, self.is_interp = self.get_function(func, k)

        #Set derivative
        self.diff = self.get_derivative(diff, 1)

        #Set second derivative (if applicable)
        if diff_2nd != -1:
            self.diff_2nd = self.get_derivative(diff_2nd, 2)

############################################
#####  Set Main Functions #####
############################################

    def get_function(self, func, k):
        #If function is actually data, we need to interpolate first
        if isinstance(func, np.ndarray):
            func = InterpolatedUnivariateSpline(func[:,0], func[:,1], k=k, ext=3)
            is_interp = True
        else:
            is_interp = False

        return func, is_interp

    def get_derivative(self, diff, order):
        #If not given, need to create
        if diff is None:
            #Handle interpolated function differently
            if self.is_interp:
                diff = self.func.derivative(order)
            else:
                diff = lambda t: derivative(self.func, t, dx=t[1]-t[0], n=order)

        return diff

############################################
############################################

############################################
#####  Miscellaneous Functions #####
############################################

    def closest_point(self, point):
        from scipy.optimize import newton, root_scalar
        #Build minimizing function

        # min_func = lambda t:
        breakpoint()

############################################
############################################
