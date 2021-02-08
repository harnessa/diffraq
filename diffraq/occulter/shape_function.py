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

    def __init__(self, func, deriv=None, deriv_2nd=-1):

        #Order with second derivative (deriv_2nd == -1: won't calculate 2nd deriv)
        k = [2, 1][deriv_2nd == -1]

        #Set function
        self.func, self.is_interp = self.get_function(func, k)

        #Set derivative
        self.deriv = self.get_derivative(deriv, 1)

        #Set second derivative (if applicable)
        if deriv_2nd != -1:
            self.deriv_2nd = self.get_derivative(deriv_2nd, 2)

    def get_function(self, func, k):
        #If function is actually data, we need to interpolate first
        if isinstance(func, np.ndarray):
            func = InterpolatedUnivariateSpline(func[:,0], func[:,1], k=k, ext=3)
            is_interp = True
        else:
            is_interp = False

        return func, is_interp

    def get_derivative(self, deriv, order):
        #If not given, need to create
        if deriv is None:
            #Handle interpolated function differently
            if self.is_interp:
                deriv = self.func.derivative(order)
            else:
                deriv = lambda t: derivative(self.func, t, dx=t[1]-t[0], n=order)

        return deriv
