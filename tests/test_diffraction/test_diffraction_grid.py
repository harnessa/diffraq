"""
test_diffraction_grid.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-08-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Test of diffraction to grid target.

"""

import diffraq
import numpy as np

class Test_diffraction_grid(object):

    ### HARDWIRED ###

    fresnum = 10.           #Fresnel Number
    n=350; m=120            #Number of quadrature points
    tol = 1e-9              #tolerance

############################################
####	Tests ####
############################################

    def test_grid(self):
        lambdaz = 1./self.fresnum

        #Smooth radial function on [0, 2pi)
        gfunc = lambda t: 1 + 0.3*np.cos(3*t)

        #Get quadratures
        xq, yq, wq = diffraq.quad.polar_quad(gfunc, self.m, self.n)

        # import time
        # tic = time.perf_counter()
        #
        # #Get quadratures
        # for i in range(1000):
        #     xq, yq, wq = diffraq.quad.polar_quad(gfunc, self.m, self.n)
        # print(f'{time.perf_counter() - tic:.2f}')

        breakpoint()

############################################
############################################

if __name__ == '__main__':

    tst = Test_diffraction_grid()
    tst.test_grid()
