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

        self.parent = parent    #Shape
        self._data = data       #polar parent - (theta, apod), petal parent - (radius, apod)

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
        func = self.func(self._data[:,0])
        diff = self.diff(self._data[:,0])

        #Get cartesian coordinates and derivative from parent
        cart_func, cart_diff = self.parent.cart_func_diffs(self._data[:,0], func=func, diff=diff)

        #Create normal from derivative
        normal = cart_diff[:,::-1]
        normal /= np.hypot(normal[:,0], normal[:,1])[:,None]

        #Build new data (negative etch adds more material)
        new_cart_func = cart_func + etch*normal*np.array([1,-1])*self.parent.opq_sign

        #Turn back to coordinates (parameter and function value)
        new_para, new_func = self.parent.inv_cart(new_cart_func)

        #Sort by parameter
        new_func = new_func[np.argsort(new_para)]
        new_para = new_para[np.argsort(new_para)]

        #Reinterpolate new function
        self.func = InterpolatedUnivariateSpline(new_para, new_func, k=self.k, ext=3)

        #Derivatives are easy
        self.diff = self.func.derivative(1)

        #Cleanup
        del func, diff, cart_func, cart_diff, normal, etch, new_cart_func, new_para, new_func

############################################
############################################
