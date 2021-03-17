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

    def __init__(self, parent, data, with_2nd=False, etch_error=None):

        self.parent = parent   #Shape
        self._data = data

        #Use second order interpolant
        self.k = 2

        #Interpolate data for both dimensions and store
        self._func = [InterpolatedUnivariateSpline(data[:,0], data[:,1+i], \
            k=self.k, ext=3) for i in range(2)]

        #Store derivative functions
        self._diff = [self._func[i].derivative(1) for i in range(2)]

        #Add etch error
        if etch_error is not None:
            self.add_etch_error(etch_error)

        #Sometimes we wont use 2nd derivative
        if with_2nd:
            self._diff_2nd = [self._func[i].derivative(2) for i in range(2)]
        else:
            self._diff_2nd = [lambda t:np.zeros_like(t), lambda t:np.zeros_like(t)]

    ############################################

    def func(self, t):
        return np.hstack((self._func[0](t), self._func[1](t)))

    def diff(self, t):
        return np.hstack((self._diff[0](t), self._diff[1](t)))

    def diff_2nd(self, t):
        return np.hstack((self._diff_2nd[0](t), self._diff_2nd[1](t)))

############################################
############################################

############################################
#####  Etch Error #####
############################################

    def add_etch_error(self, etch):
        #Return if etch is 0
        if np.abs(etch) < 1e-9:
            return

        #Get old function
        func = self.func(self._data[:,:1])

        #Create normal from derivative
        normal = self.diff(self._data[:,:1])[:,::-1]
        normal /= np.hypot(*normal.T)[:,None]

        #Get proper etch direction
        etch = etch * np.array([1, -1])

        #Build new data
        new_func = func + etch*normal

        #Reinterpolate data
        self._func = [InterpolatedUnivariateSpline(self._data[:,0], new_func[:,i], \
            k=self.k, ext=3) for i in range(2)]

        #Remake derivative
        self._diff = [self._func[i].derivative(1) for i in range(2)]

        #Cleanup
        del func, normal, etch, new_func

############################################
############################################
