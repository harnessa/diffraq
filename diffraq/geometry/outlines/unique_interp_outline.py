"""
unique_interp_outline.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 06-01-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Class that holds the interpolated function (and its derivatives)
    -- unique for different petal edges -- that describes the 2D outline of an occulter's edge.

"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

class Unique_InterpOutline(object):

    def __init__(self, parent, edge_data, combos, with_2nd=False):

        self.parent = parent    #Shape
        self._edge_data = edge_data       #polar parent - (theta, apod), petal parent - (radius, apod)
        self._combos = combos

        #Use second order interpolant
        self.k = 2

        #Loop through each combination and build functions and derivatives
        self.func, self.diff, self.diff_2nd = [], [], []
        for data_ind, etch_error in combos:

            #Get current edge data
            cur_data = edge_data[int(data_ind)]

            #Interpolate data
            func = InterpolatedUnivariateSpline(cur_data[:,0], \
                cur_data[:,1], k=self.k, ext=3)

            #Derivatives are easy
            diff = func.derivative(1)

            #Get new function with etch error
            if not np.isclose(etch_error, 0, atol=1e-9):
                #Get new function
                func, diff = self.add_etch_error(func, diff, cur_data, etch_error)

            #Sometimes we wont use 2nd derivative
            if with_2nd:
                diff_2nd = func.derivative(2)
            else:
                diff_2nd = lambda t: np.zeros_like(t)

            #Append
            self.func.append(func)
            self.diff.append(diff)
            self.diff_2nd.append(diff_2nd)

#########################################
############################################

############################################
#####  Etch Error #####
############################################

    def add_etch_error(self, func, diff, cur_data, etch):

        #Get old function and diff (radius if polar parent, apodization if petal parent)
        func = func(cur_data[:,0])
        diff = diff(cur_data[:,0])

        #Get cartesian coordinates and derivative from parent
        cart_func, cart_diff = self.parent.cart_func_diffs(cur_data[:,0], \
            func=func, diff=diff)

        #Create normal from derivative
        normal = cart_diff[:,::-1]
        normal /= np.hypot(normal[:,0], normal[:,1])[:,None]

        #Get new etch function
        func, diff = self.get_etch_func(cart_func, normal, etch)

        #Cleanup
        del cart_func, cart_diff, normal, etch

        return func, diff

    def get_etch_func(self, cart_func, normal, etch):

        #Build new data (negative etch adds more material)
        new_cart_func = cart_func + etch*normal*np.array([1,-1])*self.parent.opq_sign

        #Turn back to coordinates (parameter and function value)
        new_para, new_func = self.parent.inv_cart(new_cart_func)

        #Sort by parameter
        new_func = new_func[np.argsort(new_para)]
        new_para = new_para[np.argsort(new_para)]

        #Reinterpolate new function
        cur_func = InterpolatedUnivariateSpline(new_para, new_func, k=self.k, ext=3)

        #Derivatives are easy
        cur_diff = cur_func.derivative(1)

        #Cleanup
        del new_cart_func, new_para, new_func

        return cur_func, cur_diff

############################################
############################################
