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
from diffraq.geometry import Shape_Function

class Polar_Shape_Func(Shape_Function):

    kind = 'polar'

############################################
#####  Wrappers for Cartesian coordinate systems #####
############################################

    def cart_func(self, t):
        return self.func(t) * np.hstack((np.cos(t), np.sin(t)))

    def cart_diff(self, t):
        func = self.func(t)
        diff = self.diff(t)
        ct = np.cos(t)
        st = np.sin(t)
        ans = np.hstack((diff*ct - func*st, diff*st + func*ct))
        del func, diff, ct, st
        return ans

    def cart_diff_solo(self, t):
        func = self.func(t)
        diff = self.diff_solo(t)
        ct = np.cos(t)
        st = np.sin(t)
        ans = np.hstack((diff*ct - func*st, diff*st + func*ct))
        del func, diff, ct, st
        return ans

    def cart_diff_2nd(self, t):
        func = self.func(t)
        diff = self.diff(t)
        dif2 = self.diff_2nd(t)
        ct = np.cos(t)
        st = np.sin(t)
        ans =  np.hstack((dif2*ct - 2.*diff*st - func*ct,
                          dif2*st + 2.*diff*ct - func*st))
        del func, diff, dif2, ct, st
        return ans

############################################
############################################
