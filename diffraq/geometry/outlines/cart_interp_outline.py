"""
cart_interp_outline.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 03-16-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Class that holds the interpolated function (and its derivatives) that describes
    the 2D outline of an occulter's edge.

"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

class Cart_InterpOutline(object):

    def __init__(self, data, diff=None, with_2nd=False, etch_error=None):

        #TODO: add etch error

        #Store data
        self._data = data

        #Use second order interpolant
        k = 2

        #Interpolate data for both dimensions and store
        self._func = [InterpolatedUnivariateSpline(data[:,0], data[:,1+i], \
            k=k, ext=3) for i in range(2)]

        #Store derivative functions
        self._diff = [self._func[i].derivative(1) for i in range(2)]

        #Sometimes we wont use 2nd derivative
        if with_2nd:
            self._diff_2nd = [self._func[i].derivative(2) for i in range(2)]
        else:
            self._diff_2nd = [lambda t:np.zeros_like(t), lambda t:np.zeros_like(t)]

    ############################################
    ############################################

    def func(self, t):
        return np.hstack((self._func[0](t), self._func[1](t)))

    def diff(self, t):
        return np.hstack((self._diff[0](t), self._diff[1](t)))

    def diff_2nd(self, t):
        return np.hstack((self._diff_2nd[0](t), self._diff_2nd[1](t)))

    ############################################
    ############################################
