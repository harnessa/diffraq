"""
test_diffraction_points.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-15-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Test of diffraction to arbitrary points.

"""

import diffraq
import numpy as np

class Test_diffraction_points(object):

    def test_points(self):

        #Numerics
        n=350; m=120            #Number of quadrature points
        tol = 1e-9              #tolerance

        #Specify target grid
        ngrid = 20
        grid_width = 3

        #Build grid
        grid_pts = diffraq.utils.image_util.get_grid_points(ngrid, grid_width)
        grid_2D = np.tile(grid_pts, (ngrid,1)).T

        #Flatten grid
        xi = grid_2D.flatten()
        eta = grid_2D.T.flatten()

        #Loop over Fresnel number
        for fresnum in [10,100,1000]:

            #Fresnel number
            lambdaz = 1./fresnum

            #Smooth radial function on [0, 2pi)
            gfunc = lambda t: 1 + 0.3*np.cos(3*t)

            #Get quadratures
            xq, yq, wq = diffraq.quadrature.polar_quad(gfunc, m, n)

            #Calculate diffraction
            uu = diffraq.diffraction.diffract_points(xq, yq, wq, lambdaz, xi, eta, tol)

            #Reshape to match grid
            uu = uu.reshape(grid_2D.shape)

            #Calculate theoretical value
            utru = np.empty_like(uu)
            for j in range(uu.shape[0]):
                for k in range(uu.shape[1]):
                    utru[j,k] = 1/(1j*lambdaz) * np.sum(np.exp((1j*np.pi/lambdaz)* \
                        ((xq - grid_2D[j,k])**2 + (yq - grid_2D[k,j])**2))*wq)

            #Assert max difference is close to specified tolerance
            max_diff = tol * fresnum

            assert(np.abs(utru - uu).max() < max_diff)

        #Cleanup
        del grid_pts, grid_2D, uu, utru, xi, eta

if __name__ == '__main__':

    tst = Test_diffraction_points()
    tst.test_points()
