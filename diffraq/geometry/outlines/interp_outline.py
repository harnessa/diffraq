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

from scipy.interpolate import InterpolatedUnivariateSpline

class InterpOutline(object):

    def __init__(self, data, diff=None, with_2nd=False, etch_error=None):

        #TODO: add etch error

        #Store data
        self._data = data

        #Use second order interpolant
        k = 2

        #Interpolate data and store
        self.func = InterpolatedUnivariateSpline(data[:,0], data[:,1], k=k, ext=3)

        #Derivatives are easy
        self.diff = self.func.derivative(1)

        #Sometimes we wont use 2nd derivative
        if with_2nd:
            self.diff_2nd = self.func.derivative(2)
        else:
            self.diff_2nd = lambda t: np.zeros_like(t)
