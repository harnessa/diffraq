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
from scipy.optimize import newton

class Shape_Function(object):

    def __init__(self, func, diff=None, diff_2nd=-1):

        #Order with second derivative (diff_2nd == -1: won't calculate 2nd diff)
        k = [2, 1][diff_2nd == -1]

        #Set function
        self.func, self.is_interp = self.get_function(func, k)

        #Set derivative
        self.diff = self.get_derivative(diff, 1)

        #Set solo derivative, i.e., doesn't need array as input (assumes small dx)
        self.diff_solo = self.get_derivative(diff, 1, is_solo=True)

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

    def get_derivative(self, diff, order, is_solo=False):
        #If not given, need to create
        if diff is None:
            #Handle interpolated function differently
            if self.is_interp:
                diff = self.func.derivative(order)
            else:
                if is_solo:
                    diff = lambda t: derivative(self.func, t, dx=1e-3, n=order)
                else:
                    diff = lambda t: derivative(self.func, t, dx=t[1]-t[0], n=order)

        return diff

############################################
############################################

############################################
#####  Perturbation Functions #####
############################################

    def find_closest_point(self, point):
        #Build minimizing function (derivative of distance equation)
        min_diff = lambda t: np.sum((self.cart_func(t) - point)*self.cart_diff_solo(t))

        #Get initial guess
        if self.kind == 'radial':
            x0 = np.hypot(*point)
        else:
            x0 = np.arctan2(point[1], point[0])

        #Find root
        out, msg = newton(min_diff, x0, full_output=True)

        #Check
        if not msg.converged:
            print('\n!Closest point not Converged!\n')
            breakpoint()

        return out

    def find_width_point(self, t0, width):
        #Build distance = width equation
        dist = lambda t: np.hypot(*(self.cart_func(t) - self.cart_func(t0))) - width

        #Get initial guess
        #TODO: get better guess
        x0 = 0

        #Solve
        out, msg = newton(dist, x0, full_output=True)

        #Check
        if not msg.converged:
            print('\n!Closest point not Converged!\n')
            breakpoint()

        #Make sure it is going the right direction (CW)
        if np.arctan2(*self.cart_func(out)[::-1]) > np.arctan2(*self.cart_func(t0)[::-1]):
            #Go same distance, but opposite direction
            out = 2*t0 - out

        return out

############################################
############################################
