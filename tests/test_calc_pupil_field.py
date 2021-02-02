"""
test_calc_pupil_field.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-18-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Test of pupil field calculation.

"""

import diffraq
import numpy as np

class Test_calc_pupil_field(object):

    def test_pupil_field(self):

        ngrid = 20
        grid_width = 3
        fresnums = [10,100,1000]
        shapes = ['circle', 'polar']
        tol = 1e-9              #tolerance

        for shape in shapes:

            #Load simulator
            params = {
                'zz':               1,
                'waves':            1/np.array(fresnums),
                'occulter_shape':   shape,
                'circle_rad':       1,
                'apod_func':        lambda t: 1 + 0.3*np.cos(3*t),
                'radial_nodes':     120,
                'theta_nodes':      350,
                'tel_diameter':     grid_width,
                'num_pts':          ngrid,
                'skip_image':       True,
            }

            sim = diffraq.Simulator(params)

            #Get pupil field
            pupil, grid_pts = sim.calc_pupil_field()

            #Get 2D grid for theoretical calculation
            grid_2D = np.tile(grid_pts, (ngrid,1)).T

            #Compare to theoretical value for each Fresnel number
            for i in range(len(fresnums)):

                #Theoretical value
                utru = diffraq.utils.solution_util.direct_integration( \
                    fresnums[i], pupil[i].shape, sim.occulter.xq, sim.occulter.yq, \
                    sim.occulter.wq, grid_2D)

                #Assert max difference is close to specified tolerance
                max_diff = tol * fresnums[i]
                assert(np.abs(utru - pupil[i]).max() < max_diff)

        #Cleanup
        sim.clean_up()
        del grid_pts, grid_2D, pupil, utru

if __name__ == '__main__':

    tst = Test_calc_pupil_field()
    tst.test_pupil_field()
