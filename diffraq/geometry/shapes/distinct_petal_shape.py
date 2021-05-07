"""
distinct_petal_shape.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 05-06-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Derived class of Shape with outline parameterized in radial coordinates,
    but allowing for distinct petals.

"""

import numpy as np
import diffraq.quadrature as quad
from diffraq.geometry.shapes import PetalShape
from scipy.optimize import fmin

class DistinctPetalShape(PetalShape):

    kind = 'petal'
    param_code = 1000

############################################
#####  Wrappers for Cartesian coordinate systems #####
############################################

    def cart_func(self, r, func=None):
        #Grab function if not specified (usually by etch_error)
        if func is None:
            r, pet, pet_mul, pet_add = self.unpack_param(np.atleast_1d(r))
            func = np.empty((0,) + (1,)*(r.ndim-1))
            for p in np.unique(pet_add):
                func = np.concatenate((func, self.outline.func[int(p)//2](r) ))
            func = func*pet_mul + pet_add

        pang = np.pi/self.num_petals
        return r * np.stack((np.cos(func*pang), np.sin(func*pang)), func.ndim).squeeze()

    def cart_diff(self, r, func=None, diff=None):
        #Grab function and derivative if not specified (usually by etch_error)
        if func is None or diff is None:
            r, pet, pet_mul, pet_add = self.unpack_param(np.atleast_1d(r))
            func = np.empty((0,) + (1,)*(r.ndim-1))
            diff = np.empty_like(func)
            for p in np.unique(pet_add):
                func = np.concatenate((func, self.outline.func[int(p)//2](r) ))
                diff = np.concatenate((diff, self.outline.diff[int(p)//2](r) ))
            func = func*pet_mul + pet_add
            diff *= pet_mul

        pang = np.pi/self.num_petals
        cf = np.cos(func*pang)
        sf = np.sin(func*pang)
        ans = np.stack((cf - r*sf*diff*pang, sf + r*cf*diff*pang), diff.ndim).squeeze()
        del func, diff, cf, sf
        return ans

    def cart_diff_2nd(self, r, func=None, diff=None, diff_2nd=None):
        if func is None or diff is None:
            r, pet, pet_mul, pet_add = self.unpack_param(np.atleast_1d(r))
            func = np.empty((0,) + (1,)*(r.ndim-1))
            diff = np.empty_like(func)
            diff_2nd = np.empty_like(func)
            for p in np.unique(pet_add):
                func = np.concatenate((func, self.outline.func[int(p)//2](r) ))
                diff = np.concatenate((diff, self.outline.diff[int(p)//2](r) ))
                diff_2nd = np.concatenate((diff_2nd, self.outline.diff_2nd[int(p)//2](r) ))
            func = func*pet_mul + pet_add
            diff *= pet_mul
            diff_2nd *= pet_mul

        pang = np.pi/self.num_petals
        cf = np.cos(func*pang)
        sf = np.sin(func*pang)
        shr = (2*diff + r*diff_2nd)*pang
        ans = np.stack((-(diff*pang)**2*r*cf - sf*shr, -(diff*pang)**2*r*sf + cf*shr), func.ndim).squeeze()
        del func, diff, diff_2nd, cf, sf, shr
        return ans

    ############################################

    def cart_func_diffs(self, r, func=None, diff=None, diff_2nd=None, with_2nd=False):
        """Same functions as above, just calculate all at once to save time"""
        if func is None:
            r, pet, pet_mul, pet_add = self.unpack_param(np.atleast_1d(r))
            func = np.empty((0,) + (1,)*(r.ndim-1))
            diff = np.empty_like(func)
            diff_2nd = np.empty_like(func)
            for p in np.unique(pet_add):
                func = np.concatenate((func, self.outline.func[int(p)//2](r) ))
                diff = np.concatenate((diff, self.outline.diff[int(p)//2](r) ))
                if with_2nd:
                    diff_2nd = np.concatenate((diff_2nd, self.outline.diff_2nd[int(p)//2](r) ))
            func = func*pet_mul + pet_add
            diff *= pet_mul
            if with_2nd:
                diff_2nd *= pet_mul

        #Calculate intermediaries
        pang = np.pi/self.num_petals
        cf = np.cos(func*pang)
        sf = np.sin(func*pang)

        #Function
        func_ans = r[:,None]*np.stack((cf, sf), func.ndim).squeeze()

        #Derivative
        diff_ans = np.stack((cf - r*sf*diff*pang, sf + r*cf*diff*pang), diff.ndim).squeeze()

        #Second derivative
        if with_2nd:
            shr1 = -pang * diff**2 * r
            shr2 = 2*diff + r*diff_2nd
            diff_2nd_ans = pang*np.stack((shr1*cf - sf*shr2, shr1*sf + cf*shr2), func.ndim).squeeze()

            #Cleanup
            del func, diff, cf, sf, diff_2nd, shr1, shr2

            return func_ans, diff_ans, diff_2nd_ans

        else:

            #Cleanup
            del func, diff, cf, sf

            return func_ans, diff_ans

############################################
############################################
