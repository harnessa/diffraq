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

from scipy.misc import derivative

class LambdaOutline(object):

    def __init__(self, func, diff=None, diff_2nd=None, etch_error=None):

        #TODO: add etch error

        #Store function
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
