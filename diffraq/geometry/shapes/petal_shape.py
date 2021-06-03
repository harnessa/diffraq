"""
petal_shape.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-08-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Derived class of Shape with outline parameterized in radial coordinates.

"""

import numpy as np
import diffraq.quadrature as quad
from diffraq.geometry import Shape
from scipy.optimize import fminbound

class PetalShape(Shape):

    kind = 'petal'
    param_code = 1000

############################################
#####  Main Shape #####
############################################

    def build_local_shape_quad(self):
        #Calculate petal quadrature
        xq, yq, wq = quad.petal_quad(self.outline.func, self.num_petals, \
            self.min_radius, self.max_radius, self.radial_nodes, self.theta_nodes, \
            has_center=self.has_center)

        return xq, yq, wq

    def build_local_shape_edge(self, npts=None):
        #Radial nodes
        if npts is None:
            npts = self.radial_nodes

        #Calculate petal edge
        edge = quad.petal_edge(self.outline.func, self.num_petals, \
            self.min_radius, self.max_radius, npts)

        return edge

############################################
############################################

############################################
#####  Wrappers for Cartesian coordinate systems #####
############################################

    #Parameter t has radius and petal number packed in it.
    #Petal can be positive or negative, depending on if trailing (+) or leading (-)
    #Petals = [1, num_petals], but don't rotate at 1 position

    def unpack_param(self, t):
        r = t % self.param_code
        pet = t//self.param_code
        pet += (pet == 0)               #Don't allow to be zero (used without parameterization)
        pet_mul = np.sign(pet)
        pet_add = 2*(np.abs(pet) - 1)

        return r, pet, pet_mul, pet_add

    def pack_param(self, r, pet):
        return r + self.param_code*pet

    ################################

    def cart_func(self, r, func=None):
        #TODO: speed this up since I don't need to accomodate multiple petals
        #Grab function if not specified (usually by etch_error)
        if func is None:
            r, pet, pet_mul, pet_add = self.unpack_param(r)
            func = self.outline.func(r)*pet_mul + pet_add

        pang = np.pi/self.num_petals
        return r * np.stack((np.cos(func*pang), np.sin(func*pang)), func.ndim).squeeze()

    def cart_diff(self, r, func=None, diff=None):
        #Grab function and derivative if not specified (usually by etch_error)
        if func is None or diff is None:
            r, pet, pet_mul, pet_add = self.unpack_param(r)
            func = self.outline.func(r)*pet_mul + pet_add
            diff = self.outline.diff(r)*pet_mul

        pang = np.pi/self.num_petals
        cf = np.cos(func*pang)
        sf = np.sin(func*pang)
        ans = np.stack((cf - r*sf*diff*pang, sf + r*cf*diff*pang), diff.ndim).squeeze()
        del func, diff, cf, sf
        return ans

    def cart_diff_2nd(self, r, func=None, diff=None, diff_2nd=None):
        if func is None or diff is None:
            r, pet, pet_mul, pet_add = self.unpack_param(r)
            func = self.outline.func(r)*pet_mul + pet_add
            diff = self.outline.diff(r)*pet_mul
            diff_2nd = self.outline.diff_2nd(r)*pet_mul

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
            r, pet, pet_mul, pet_add = self.unpack_param(r)
            func = self.outline.func(r)*pet_mul + pet_add
            diff = self.outline.diff(r)*pet_mul
            if with_2nd:
                diff_2nd = self.outline.diff_2nd(r)*pet_mul

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

    def inv_cart(self, xy):
        #Inverse to go from cartesian to parameter, function
        rad = np.hypot(xy[:,0], xy[:,1])
        the = np.arctan2(xy[:,1], xy[:,0]) * self.num_petals/np.pi
        return rad, the

############################################
############################################

############################################
#####  Overwritten Perturbation Functions #####
############################################

    def find_closest_point(self, point):
        #Build minimizing function (distance equation)
        min_diff = lambda r, pet: \
            np.hypot(*(self.cart_func(self.pack_param(r, pet)) - point))

        #Get initial guess
        r_guess = np.hypot(*point)

        #Find best fit to equation
        r0, p0 = self.minimize_over_petals(min_diff, np.min(self.min_radius))

        #Get best fit parameter
        t0 = self.pack_param(r0, p0)

        return t0

    def find_width_point(self, t0, width):
        #Start radius
        r0 = self.unpack_param(t0)[0]

        #Build |distance - width| equation
        min_diff = lambda r, pet: np.abs(np.hypot(*(self.cart_func( \
            self.pack_param(r, pet)) - self.cart_func(t0))) - width)

        #Find best fit to equation (set r0 as minimum radius to force increasing radius)
        rf, pf = self.minimize_over_petals(min_diff, r0)

        #Get best fit parameter
        tf = self.pack_param(rf, pf)

        #Check that width is within 1% of requested width
        if np.abs(np.hypot(*(self.cart_func(tf) - \
            self.cart_func(t0))) - width)/width > 0.01:
            print('\n!Closest point (width) not Converged!\n')

        return tf

    def minimize_over_petals(self, min_eqn, r0):
        #Loop over petals and find roots
        fits = np.empty((0,3))
        pets = np.arange(1, 1 + self.num_petals)
        pets = np.concatenate((pets, -pets[::-1]))
        r1 = np.max(self.max_radius)
        for i in pets:
            out = fminbound(min_eqn, r0, r1, args=(i,), disp=0, xtol=1e-8)
            fits = np.concatenate((fits, [[out, i, min_eqn(out, i)]]))

        #Find best fit index
        ind = np.argmin(np.abs(fits[:,2]))

        return fits[ind][:2]

############################################
############################################

class StarshadeShape(PetalShape):
    """ Just another name for PetalShape """
    pass
