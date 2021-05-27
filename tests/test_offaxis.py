"""
test_offaxis.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 04-22-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Test of shifting the target off-axis using diffract_grid.

"""

import diffraq
import numpy as np
import time

class Test_Offaxis(object):

    def test_pupil_field(self):

        ngrid = 15
        grid_width = 3
        fresnums = [10]
        tol = 1e-9              #tolerance

        #Simulator params
        params = {
            'zz':               1,
            'waves':            1/np.array(fresnums),
            'radial_nodes':     250,
            'theta_nodes':      350,
            'tel_diameter':     grid_width,
            'num_pts':          ngrid,
            'skip_image':       True,
        }

        shape = {'kind':'circle', 'max_radius':1}

        for x0 in [-1, 0.25]:
            for y0 in [-0.5, 0.17]:

                #Add target center
                params['target_center'] = [x0, y0]

                #Build simulator
                sim = diffraq.Simulator(params, shape)

                #Get pupil field
                pupil, grid_pts = sim.calc_pupil_field()

                #Get 2D grid for theoretical calculation
                gx_2D = grid_pts[0].reshape((ngrid,ngrid))
                gy_2D = grid_pts[1].reshape((ngrid,ngrid))

                #Compare to theoretical value for each Fresnel number
                for i in range(len(fresnums)):

                    #Theoretical value
                    utru = diffraq.utils.solution_util.direct_integration( \
                        fresnums[i], pupil[i].shape, sim.occulter.xq, \
                        sim.occulter.yq, sim.occulter.wq, gx_2D, gy_2D=gy_2D)

                    #Assert max difference is close to specified tolerance
                    max_diff = tol * fresnums[i]
                    assert(np.abs(utru - pupil[i]).max() < max_diff)

        #Cleanup
        sim.clean_up()
        del grid_pts, gx_2D, gy_2D, pupil, utru

############################################
############################################

    def test_grid_points(self):

        ngrid = 15
        grid_width = 3
        lamzz = 1/10
        tol = 1e-9              #tolerance

        #Simulator params
        params = {
            'radial_nodes':     250,
            'theta_nodes':      350,
            'tel_diameter':     grid_width,
            'num_pts':          ngrid,
            'skip_image':       True,
        }

        shape = {'kind':'circle', 'max_radius':1}

        #Build simulator and points
        sim = diffraq.Simulator(params, shape)
        sim.occulter.build_quadrature()

        #Build grid
        grid_pts = diffraq.utils.image_util.get_grid_points(ngrid, grid_width)
        grid_2D = np.tile(grid_pts, (ngrid,1))

        #Flatten grid
        xi = grid_2D.flatten()
        eta = grid_2D.T.flatten()

        for x0 in [-1, 0.25]:
            for y0 in [-0.5, 0.17]:

                #Run diffraction with target points
                pup_pts = diffraq.diffraction.diffract_points( \
                    sim.occulter.xq, sim.occulter.yq, sim.occulter.wq, lamzz, xi + x0, \
                    eta + y0, sim.fft_tol, is_babinet=False).reshape((ngrid,ngrid))

                #Run diffraction with target grid
                pup_grd = diffraq.diffraction.diffract_grid( \
                    sim.occulter.xq - x0, sim.occulter.yq - y0, sim.occulter.wq, lamzz, \
                    grid_pts, sim.fft_tol, is_babinet=False)

                #Assert true
                assert(np.allclose(pup_pts, pup_grd))

        #Cleanup
        sim.clean_up()
        del grid_pts, grid_2D, xi, eta, pup_pts, pup_grd

############################################
############################################

if __name__ == '__main__':

    tst = Test_Offaxis()
    tst.test_pupil_field()
    tst.test_grid_points()
