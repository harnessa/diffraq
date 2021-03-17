"""
interp_outline.py

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

class InterpOutline(object):

    def __init__(self, parent, data, with_2nd=False, etch_error=None):

        self.parent = parent   #Shape
        self._data = data

        #Use second order interpolant
        self.k = 2

        #Interpolate data and store
        self.func = InterpolatedUnivariateSpline(data[:,0], data[:,1], k=self.k, ext=3)

        #Derivatives are easy
        self.diff = self.func.derivative(1)

        #Add etch error
        if etch_error is not None:
            self.add_etch_error(etch_error)

        #Sometimes we wont use 2nd derivative
        if with_2nd:
            self.diff_2nd = self.func.derivative(2)
        else:
            self.diff_2nd = lambda t: np.zeros_like(t)

############################################
############################################

############################################
#####  Etch Error #####
############################################

    def add_etch_error(self, etch):
        #Return if etch is 0
        if np.abs(etch) < 1e-9:
            return

        #Get old function and diff (radius if polar parent, apodization if petal parent)
        func = self.func(self._data[:,:1])
        diff = self.diff(self._data[:,:1])

        old = func.copy()

        #Get cartesian coordinates and derivative depending on parent
        kind = self.parent.kind
        cart_func = getattr(self, f'{kind}_cart_func')(func, self._data[:,:1])
        cart_diff = getattr(self, f'{kind}_cart_diff')(func, diff, self._data[:,:1])

        #Create normal from derivative
        normal = cart_diff[:,::-1]
        normal /= np.hypot(*normal.T)[:,None]

        #Get proper etch direction
        etch = etch * np.array([1, -1])

        #Build new data
        new_cart_func = cart_func + etch*normal

        #Turn back to coordinates
        new_func = getattr(self, f'{kind}_inv_cart')(new_cart_func)

        #Reinterpolate new function
        self.func = InterpolatedUnivariateSpline(self._data[:,0], new_func, k=self.k, ext=3)

        #Derivatives are easy
        self.diff = self.func.derivative(1)

        #Cleanup
        del func, diff, cart_func, cart_diff, normal, etch, new_cart_func, new_func

    ##########################################

    def polar_cart_func(self, func, t):
        return func * np.hstack((np.cos(t), np.sin(t)))

    def polar_cart_diff(self, func, diff, t):
        ct = np.cos(t)
        st = np.sin(t)
        ans = np.hstack((diff*ct - func*st, diff*st + func*ct))
        del ct, st
        return ans

    def polar_inv_cart(self, xy):
        return np.hypot(*xy.T)

    def petal_cart_func(self, func, r):
        pet_mul = np.pi/self.parent.num_petals
        return r * np.hstack((np.cos(func*pet_mul), np.sin(func*pet_mul)))

    def petal_cart_diff(self, func, diff, r):
        pet_mul = np.pi/self.parent.num_petals
        cf = np.cos(func*pet_mul)
        sf = np.sin(func*pet_mul)
        #Derivative has negative for trailing petal
        ans = -1 * np.hstack((cf - r*sf*diff*pet_mul, sf + r*cf*diff*pet_mul))
        del cf, sf
        return ans

    def petal_inv_cart(self, xy):
        return np.arctan2(*xy[:,::-1].T) * self.parent.num_petals/np.pi

############################################
############################################
