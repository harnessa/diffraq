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

        ngrid = 512
        grid_width = 3
        fresnums = [10,100,1000]
        shapes = ['circle', 'polar']
        tol = 1e-9              #tolerance

        #Simulator params
        params = {
            'zz':               1,
            'waves':            1/np.array(fresnums),
            'radial_nodes':     120,
            'theta_nodes':      350,
            'tel_diameter':     grid_width,
            'num_pts':          ngrid,
            'skip_image':       True,
        }

        shapes = [{'kind':'circle', 'max_radius':1},
            {'kind':'polar', 'edge_func':lambda t: 1 + 0.3*np.cos(3*t)}]

        for shape in shapes:

            #Build simulator
            sim = diffraq.Simulator(params, shape)

            #Get pupil field
            pupil, grid_pts = sim.calc_pupil_field()

            #Get 2D grid for theoretical calculation
            grid_2D = np.tile(grid_pts, (ngrid,1))

            #Compare to theoretical value for each Fresnel number
            for i in range(len(fresnums)):

                #Theoretical value
                # utru = diffraq.utils.solution_util.direct_integration( \
                    # fresnums[i], pupil[i].shape, sim.occulter.xq, sim.occulter.yq, \
                    # sim.occulter.wq, grid_2D)

                #Assert max difference is close to specified tolerance
                max_diff = tol * fresnums[i]
                # assert(np.abs(utru - pupil[i]).max() < max_diff)
                import matplotlib.pyplot as plt;plt.ion()
                # plt.figure()
                # plt.imshow(abs(utru))
                plt.figure()
                plt.imshow(abs(pupil[i]))
                breakpoint()

        #Cleanup
        sim.clean_up()
        del grid_pts, grid_2D, pupil, utru

if __name__ == '__main__':

    tst = Test_calc_pupil_field()
    tst.test_pupil_field()
