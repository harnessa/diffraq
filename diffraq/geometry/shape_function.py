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

############################################
#####  Normal Shape Function #####
############################################

class Shape_Function(object):

    def __init__(self, kind, func, diff=None, diff_2nd=-1):

        #Kind of function. Options = [polar, cart, apod, loci]
        self.kind = kind

        #Bail if loci
        if kind == 'loci':
            self._func = func     #Points to loading loci function
            return

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
            #Use wrapper for Cartesian (multidimensional)
            if func.shape[-1] > 2:
                func = Cart_InterpFunc(func, k)
            else:
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
    #####  Wrappers for Cartesian coordinate systems #####
    ############################################

    def cart_func(self, t):
        return getattr(self, f'cart_func__{self.kind}')(t)

    def cart_diff(self, t):
        return getattr(self, f'cart_diff__{self.kind}')(t)

    def cart_diff_2nd(self, t):
        return getattr(self, f'cart_diff_2nd__{self.kind}')(t)

    def cart_func__cart(self, t):
        return self.func(t)

    def cart_diff__cart(self, t):
        return self.diff(t)

    def cart_diff_2nd__cart(self, t):
        return self.diff_2nd(t)

    def cart_func__polar(self, t):
        return self.func(t) * np.hstack((np.cos(t), np.sin(t)))

    def cart_diff__polar(self, t):
        return np.hstack(( self.diff(t)*np.cos(t) - self.func(t)*np.sin(t), \
                           self.diff(t)*np.sin(t) + self.func(t)*np.cos(t)))

    def cart_diff_2nd__polar(self, t):
        return np.hstack((
            self.diff_2nd(t)*np.cos(t) - 2.*self.diff(t)*np.sin(t) - self.func(t)*np.cos(t),
            self.diff_2nd(t)*np.sin(t) + 2.*self.diff(t)*np.cos(t) - self.func(t)*np.sin(t),
        ))

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
