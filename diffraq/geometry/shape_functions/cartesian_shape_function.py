"""
cartesian_shape_func.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-15-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Class that is the function that describes the shape of a cartesian
    occulter's edge.

"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from diffraq.geometry import ShapeFunction

class CartesianShapeFunction(ShapeFunction):

    kind = 'cartesian'

    def get_function(self, func, k):
        """Overwrite function to handle multi-dimensional function"""
        #If function is actually data, we need to interpolate first
        if isinstance(func, np.ndarray):
            #Use wrapper for Cartesian (multidimensional)
            func = Cart_InterpFunc(func, k)
            is_interp = True
        else:
            is_interp = False

        return func, is_interp

    ############################################
    #####  Wrappers for Cartesian coordinate systems #####
    ############################################

    def cart_func(self, t):
        return self.func(t)

    def cart_diff(self, t):
        return self.diff(t)

    def cart_diff_solo(self, t):
        return self.diff_solo(t)

    def cart_diff_2nd(self, t):
        return self.diff_2nd(t)

    ############################################
    ############################################

############################################
#####  Cartesian Interpolated Function #####
############################################

class Cart_InterpFunc(object):

    def __init__(self, func, k):
        self._func = [InterpolatedUnivariateSpline(func[:,0], func[:,1+i], \
            k=k, ext=3) for i in range(2)]

    def __call__(self, t):
        return np.hstack((self._func[0](t), self._func[1](t)))

    def derivative(self, order):
        return getattr(self, f'set_derivative_{order}')()

    def set_derivative_1(self):
        self._diff = [self._func[i].derivative(1) for i in range(2)]
        return lambda t: np.hstack((self._diff[0](t), self._diff[1](t)))

    def set_derivative_2(self):
        self._diff_2nd = [self._func[i].derivative(2) for i in range(2)]
        return lambda t: np.hstack((self._diff_2nd[0](t), self._diff_2nd[1](t)))

############################################
############################################
