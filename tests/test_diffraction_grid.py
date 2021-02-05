"""
test_diffraction_grid.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-08-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Test of diffraction to target grid.

"""

import diffraq
import numpy as np

class Test_diffraction_grid(object):

    def test_grid(self):

        #Numerics
        n=350; m=120            #Number of quadrature points
        tol = 1e-9              #tolerance

        #Specify target grid
        ngrid = 20
        grid_width = 3

        #Build grid
        grid_pts = diffraq.utils.image_util.get_grid_points(ngrid, grid_width)
        grid_2D = np.tile(grid_pts, (ngrid,1)).T

        #Loop over Fresnel number
        for fresnum in [10, 100, 1000]:

            #Fresnel number
            lambdaz = 1./fresnum

            #Run polar and cartesian
            for shape in ['polar', 'cartesian']:

                #Get quadratures
                xq, yq, wq = getattr(self, f'get_quad_{shape}')(m, n)

                #Calculate diffraction
                uu = diffraq.diffraction.diffract_grid(xq, yq, wq, lambdaz, grid_pts, tol)

                #Calculate theoretical value
                utru = diffraq.utils.solution_util.direct_integration(fresnum, \
                    uu.shape, xq, yq, wq, grid_2D)

                #Assert max difference is close to specified tolerance
                max_diff = tol * fresnum
                assert(np.abs(utru - uu).max() < max_diff)

        #Cleanup
        del grid_pts, grid_2D, uu, utru

############################################

    def get_quad_polar(self, m, n):
        #Smooth radial function on [0, 2pi)
        gfunc = lambda t: 1 + 0.3*np.cos(3*t)

        #Get quadratures
        xq, yq, wq = diffraq.quadrature.polar_quad(gfunc, m, n)

        return xq, yq, wq

    def get_quad_cartesian(self, m, n):
        #Kite occulter
        func  = lambda t: np.hstack((0.5*np.cos(t) + 0.5*np.cos(2*t), np.sin(t)))
        deriv = lambda t: np.hstack((-0.5*np.sin(t) - np.sin(2*t), np.cos(t)))

        #Get quad
        xq, yq, wq = diffraq.quadrature.cartesian_quad(func, deriv, m, n)

        return xq, yq, wq

############################################

if __name__ == '__main__':

    tst = Test_diffraction_grid()
    tst.test_grid()
