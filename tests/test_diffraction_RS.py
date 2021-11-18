"""
test_diffraction_RS.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 11-18-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Test of diffraction to via Rayleigh sommerfeld equations.

"""

import diffraq
import numpy as np

class Test_diffraction_RS(object):

    def test_RS(self):

        #Numerics
        n=350; m=120            #Number of quadrature points
        tol = 5e-3

        #Specify target grid
        ngrid = 20
        grid_width = 3
        wave = 0.5e-6

        #Build grid
        grid_pts = diffraq.utils.image_util.get_grid_points(ngrid, grid_width)
        grid_2D = np.tile(grid_pts, (ngrid,1))

        #Loop over Fresnel number
        for fresnum in [1,10,50]:

            #Fresnel number
            lambdaz = 1./fresnum
            zz = lambdaz / wave

            #Run polar and cartesian
            for shape in ['polar', 'cartesian']:

                #Get quadratures
                xq, yq, wq = getattr(self, f'get_quad_{shape}')(m, n)

                #Calculate diffraction RS1
                u1 = diffraq.diffraction.diffract_RS1(xq, yq, wq, wave, zz, grid_2D)
                u2 = diffraq.diffraction.diffract_RS2(xq, yq, wq, wave, zz, grid_2D)

                #Calculate theoretical value
                utru = diffraq.utils.solution_util.direct_integration(fresnum, \
                    grid_2D.shape, xq, yq, wq, grid_2D) * np.exp(1j*2*np.pi/wave*1e19)

                #Assert max difference is close to specified tolerance
                max_diff = tol / np.sqrt(fresnum)

                assert(np.abs(utru - u1).max() < max_diff)
                assert(np.abs(utru - u2).max() < max_diff)

        #Cleanup
        del grid_pts, grid_2D, u1, u2, utru

############################################

    def get_quad_polar(self, m, n):
        #Smooth radial function on [0, 2pi)
        gfunc = lambda t: 1 + 0.3*np.cos(3*t)

        #Get quadratures
        xq, yq, wq = diffraq.quadrature.polar_quad(gfunc, m, n)

        return xq, yq, wq

    def get_quad_cartesian(self, m, n):
        #Kite occulter
        func = lambda t: np.hstack((0.5*np.cos(t) + 0.5*np.cos(2*t), np.sin(t)))
        diff = lambda t: np.hstack((-0.5*np.sin(t) - np.sin(2*t), np.cos(t)))

        #Get quad
        xq, yq, wq = diffraq.quadrature.cartesian_quad(func, diff, m, n)

        return xq, yq, wq

############################################

if __name__ == '__main__':

    tst = Test_diffraction_RS()
    tst.test_RS()
