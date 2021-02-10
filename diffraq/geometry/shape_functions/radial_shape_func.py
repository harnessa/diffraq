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
        cf = np.cos(func)
        sf = np.sin(func)
        ans = np.hstack((cf - r*sf*diff, sf + r*cf*diff))
        del func, diff, cf, sf
        return ans

    def cart_diff_solo(self, r):
        func = self.func(r)
        diff = self.diff_solo(r)
        cf = np.cos(func)
        sf = np.sin(func)
        ans = np.hstack((cf - r*sf*diff, sf + r*cf*diff))
        del func, diff, cf, sf
        return ans

    def cart_diff_2nd(self, r):
        func = self.func(r)
        diff = self.diff(r)
        dif2 = self.diff_2nd(r)
        cf = np.cos(func)
        sf = np.sin(func)
        ans = np.hstack((-sf*diff - ((sf + r*cf*diff)*diff + r*sf*dif2), \
                          cf*diff + ((cf - r*sf*diff)*diff + r*cf*dif2)))
        del func, diff, dif2, cf, sf
        return ans

############################################
############################################
