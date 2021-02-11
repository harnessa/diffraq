"""
petal_shape_func.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-15-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Class that is the function that describes the shape of a petalized (starshade)
    occulter's edge.

"""

import numpy as np
from diffraq.geometry import Shape_Function
from scipy.optimize import fmin

class Petal_Shape_Func(Shape_Function):

    kind = 'petal'
    param_code = 1000

    def __init__(self, func, diff=None, diff_2nd=-1, num_petals=1):
        super().__init__(func, diff=diff, diff_2nd=diff_2nd)

        self.num_petals = num_petals

############################################
#####  Wrappers for Cartesian coordinate systems #####
############################################

    #Parameter t has radius and petal number packed in it.
    #Petal can be positive or negative, depending on if trailing or leading

    def unpack_param(self, t):
        r = t % self.param_code
        pet = t//self.param_code
        pet_mul = np.pi/self.num_petals*(np.sign(pet)+(pet==0))
        pet_add = 2*np.abs(pet)*np.pi/self.num_petals
        return r, pet_mul, pet_add

    def pack_param(self, r, pet):
        return r + self.param_code*pet

    ################################

    def cart_func(self, t):
        r, pet_mul, pet_add = self.unpack_param(t)
        func = self.func(r)*pet_mul + pet_add
        return r * np.hstack((np.cos(func), np.sin(func)))

    def cart_diff(self, t):
        r, pet_mul, pet_add = self.unpack_param(t)
        func = self.func(r)*pet_mul + pet_add
        diff = self.diff(r)*pet_mul
        cf = np.cos(func)
        sf = np.sin(func)
        ans = np.hstack((cf - r*sf*diff, sf + r*cf*diff))
        del func, diff, cf, sf
        return ans

    def cart_diff_solo(self, t):
        r, pet_mul, pet_add = self.unpack_param(t)
        func = self.func(r)*pet_mul + pet_add
        diff = self.diff_solo(r)*pet_mul
        cf = np.cos(func)
        sf = np.sin(func)
        ans = np.hstack((cf - r*sf*diff, sf + r*cf*diff))
        del func, diff, cf, sf
        return ans

    def cart_diff_2nd(self, t):
        r, pet_mul, pet_add = self.unpack_param(t)
        func = self.func(r)*pet_mul + pet_add
        diff = self.diff(r)*pet_mul
        dif2 = self.diff_2nd(r)*pet_mul
        cf = np.cos(func)
        sf = np.sin(func)
        ans = np.hstack((-sf*diff - ((sf + r*cf*diff)*diff + r*sf*dif2), \
                          cf*diff + ((cf - r*sf*diff)*diff + r*cf*dif2)))
        del func, diff, dif2, cf, sf
        return ans

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
        #Build |distance - width| equation
        min_diff = lambda r, pet: np.abs(np.hypot(*( \
            self.cart_func(self.pack_param(r, pet)) - self.cart_func(t0))) - width)

        #Find best fit to equation
        r0, p0 = self.minimize_over_petals(min_diff, self.unpack_param(t0)[0])

        #Get best fit parameter
        tf = self.pack_param(r0, p0)

        #Make sure it is going the right direction (CW)
        if np.arctan2(*self.cart_func(tf)[::-1]) > np.arctan2(*self.cart_func(t0)[::-1]):
            #Go same distance, but opposite direction
            tf = 2*t0 - tf

        #Check that width is within 10% of requested width
        if np.abs(np.hypot(*(self.cart_func(tf) - self.cart_func(t0))) - width)/width > 0.1:
            print('\n!Closest point (width) not Converged!\n')
            breakpoint()

        return tf

    def minimize_over_petals(self, min_eqn, x0):
        #Loop over petals and find roots
        fits = np.empty((0,3))
        for i in range(-self.num_petals, self.num_petals):
            out = fmin(min_eqn, x0, args=(i,), disp=0, xtol=1e-8, ftol=1e-8)[0]
            fits = np.concatenate((fits, [[out, i, min_eqn(out, i)]]))

        #Find best fit index
        ind = np.argmin(np.abs(fits[:,2]))

        return fits[ind][:2]

############################################
############################################
