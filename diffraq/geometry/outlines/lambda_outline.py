"""
lambda_outline.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 03-16-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Class that holds the lambda function (and its derivatives) that describes
    the 2D outline of an occulter's edge.

"""

import numpy as np
from scipy.misc import derivative

class LambdaOutline(object):

    def __init__(self, parent, func, diff=None, diff_2nd=None, etch_error=None):

        self.parent = parent   #Shape
        self._func = func

        #Store derivative is supplied, otherwise build
        if diff is not None:
            self._diff = diff
        else:
            #Point to unknown function
            self._diff = self.unknown_diff

        #Store derivative is supplied, otherwise build
        if diff_2nd is not None:
            self._diff_2nd = diff_2nd
        else:
            #Point to unknown function
            self._diff_2nd = self.unknown_diff_2nd

        #Add etch error
        if etch_error is not None:
            self.add_etch_error(etch_error)

############################################
############################################

    def func(self, t):
        return self._func(t)

    def diff(self, t):
        return self._diff(t)

    def diff_2nd(self, t):
        return self._diff_2nd(t)

    ############################################

    def unknown_diff(self, t):
        return derivative(self._func, t, dx=t[1]-t[0], n=1)

    def unknown_diff_2nd(self, t):
        return derivative(self._func, t, dx=t[1]-t[0], n=2)

############################################
############################################

############################################
#####  Etch Error #####
############################################

    def add_etch_error(self, etch):
        #Return if etch is 0
        if np.abs(etch) < 1e-9:
            return

        tt = np.linspace(0, 2*np.pi, 1000)[:,None]
        old = self.func(tt).copy()

        #Store etch
        self.etch = etch * np.array([1., -1])

        #Point to new functions depending on parent's kind (ignore 2nd for now)
        self.func = getattr(self, f'{self.parent.kind}_etch_func')
        self.diff = getattr(self, f'{self.parent.kind}_etch_diff')

        #Need to repoint parent's cartesian functions too
        # if self.parent.kind == 'polar':
            # self.parent.cart_func =


        # new = self.func(tt)
        # import matplotlib.pyplot as plt;plt.ion()
        # plt.cla()
        # plt.plot(old*np.cos(tt), old*np.sin(tt))
        # plt.plot(new*np.cos(tt), new*np.sin(tt), '--')
        # # plt.plot(*old.T)
        # # plt.plot(*new.T, '--')
        # breakpoint()

    ############################################

    def cartesian_etch_func(self, t):
        diff = self._diff(t)
        return self._func(t) + self.etch*diff[:,::-1]/np.hypot(*diff.T)[:,None]

    def cartesian_etch_diff(self, t):
        diff = self._diff(t)
        diff_2nd = self.diff_2nd(t)
        norm = np.hypot(*diff.T)
        return diff + self.etch*(diff_2nd[:,::-1] - diff[:,::-1]* \
            (np.sum(diff*diff_2nd,1)/norm**2)[:,None])/norm[:,None]

    ############################################
        ### Polar Parent ###

    def polar_etch_func(self, t):
        func = self._func(t)

        #TODO: add this for polar coords -- lots of calculus! esp. in derivative
        #see non_package_sandbox/lab_starshades/normal_lines.py for working function
        breakpoint()
        return func - self.etch * func / np.hypot(func, self._diff(t))

############################################
############################################
