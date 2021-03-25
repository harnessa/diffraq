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
from scipy.optimize import fmin, newton

class PetalShape(Shape):

    kind = 'petal'
    param_code = 1000

    #TODO: rename 'petal' to 'radial' ?

############################################
#####  Main Shape #####
############################################

    def build_local_shape_quad(self):
        #Calculate starshade quadrature
        xq, yq, wq = quad.starshade_quad(self.outline.func, self.num_petals, \
            self.min_radius, self.max_radius, self.radial_nodes, self.theta_nodes, \
            has_center=self.has_center)

        return xq, yq, wq

    def build_local_shape_edge(self, npts=None):
        #Radial nodes
        if npts is None:
            npts = self.radial_nodes

        #Calculate starshade edge
        edge = quad.starshade_edge(self.outline.func, self.num_petals, \
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
        pet = pet + (pet == 0)          #Don't allow to be zero (used without parameterization)
        pet_mul = np.pi/self.num_petals*np.sign(pet)
        pet_add = (np.abs(pet)-1)*2*np.pi/self.num_petals
        return r, pet, pet_mul, pet_add

    def pack_param(self, r, pet):
        return r + self.param_code*pet

    ################################

    def cart_func(self, r, func=None):
        #Grab function if not specified (usually by etch_error)
        if func is None:
            r, pet, pet_mul, pet_add = self.unpack_param(r)
            func = self.outline.func(r)*pet_mul + pet_add
        else:
            func *= np.pi/self.num_petals

        return r * np.stack((np.cos(func), np.sin(func)), func.ndim).squeeze()

    def cart_diff(self, r, func=None, diff=None):
        #Grab function and derivative if not specified (usually by etch_error)
        if func is None or diff is None:
            r, pet, pet_mul, pet_add = self.unpack_param(r)
            func = self.outline.func(r)*pet_mul + pet_add
            diff = self.outline.diff(r)*pet_mul
        else:
            func *= np.pi/self.num_petals
            diff *= np.pi/self.num_petals
        cf = np.cos(func)
        sf = np.sin(func)
        ans = np.stack((cf - r*sf*diff, sf + r*cf*diff), diff.ndim).squeeze()
        del func, diff, cf, sf
        return ans

    def cart_diff_2nd(self, t, func=None, diff=None, diff_2nd=None):
        if func is None or diff is None:
            r, pet, pet_mul, pet_add = self.unpack_param(t)
            func = self.outline.func(r)*pet_mul + pet_add
            diff = self.outline.diff(r)*pet_mul
            diff_2nd = self.outline.diff_2nd(r)*pet_mul
        else:
            func *= np.pi/self.num_petals
            diff *= np.pi/self.num_petals
            diff_2nd *= np.pi/self.num_petals
        cf = np.cos(func)
        sf = np.sin(func)
        shr = 2*diff + r*diff_2nd
        ans = np.stack((-diff**2*r*cf - sf*shr, -diff**2*r*sf + cf*shr), func.ndim).squeeze()
        del func, diff, diff_2nd, cf, sf, shr
        return ans

    def inv_cart(self, xy):
        #Inverse to go from cartesian to parameter, function
        rad = np.hypot(*xy.T)
        the = np.arctan2(*xy[:,::-1].T) * self.num_petals/np.pi
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
        r0, p0 = self.minimize_over_petals(min_diff, r_guess)

        #Get best fit parameter
        t0 = self.pack_param(r0, p0)

        return t0

    def find_width_point(self, t0, width):
        #Start radius
        r0 = self.unpack_param(t0)[0]

        #Build |distance - width| equation (force increasing radius w/ heaviside)
        min_diff = lambda r, pet:  1/(1+1e-9-np.heaviside(r0-r, 0.5)) * \
            np.abs(np.hypot(*(self.cart_func(self.pack_param(r, pet)) - \
            self.cart_func(t0))) - width)

        #Find best fit to equation (guess with nudge towards positive radius)
        rf, pf = self.minimize_over_petals(min_diff, r0 + width/2)

        #Get best fit parameter
        tf = self.pack_param(rf, pf)

        #Check that width is within 1% of requested width
        if np.abs(np.hypot(*(self.cart_func(tf) - \
            self.cart_func(t0))) - width)/width > 0.01:
            print('\n!Closest point (width) not Converged!\n')

        return tf

    def minimize_over_petals(self, min_eqn, x0):
        #Loop over petals and find roots
        fits = np.empty((0,3))
        pets = np.arange(1, 1 + self.num_petals)
        pets = np.concatenate((pets, -pets[::-1]))
        for i in pets:
            out = fmin(min_eqn, x0, args=(i,), disp=0, xtol=1e-8, ftol=1e-8)[0]
            fits = np.concatenate((fits, [[out, i, min_eqn(out, i)]]))

        #Find best fit index
        ind = np.argmin(np.abs(fits[:,2]))

        return fits[ind][:2]

############################################
############################################

class StarshadeShape(PetalShape):
    """ Just another name for PetalShape """
    pass
