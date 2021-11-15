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
        radius = 7e-3
        fresnums = [1, 10]
        shapes = ['circle', 'polar']
        tol = 1e-9              #tolerance

        #Simulator params
        params = {
            'waves':            0.5e-6,
            'radial_nodes':     200,
            'theta_nodes':      200,
            'tel_diameter':     3e-3,
            'num_pts':          ngrid,
            'skip_image':       True,
            'fft_tol':1e-12,
        }

        shapes = [{'kind':'circle', 'max_radius':radius},
            {'kind':'polar', 'edge_func':lambda t: radius*(1 + 0.3*np.cos(3*t))}]

        for shape in shapes[1:]:

            #Compare to theoretical value for each Fresnel number
            for i in range(len(fresnums))[1:]:

                params['zz'] = radius**2/(params['waves'] * fresnums[i])

                #Build simulator
                sim = diffraq.Simulator(params, shape)
                print(sim.zz)

                #Get pupil field
                pupil, grid_pts = sim.calc_pupil_field()

                #Get 2D grid for theoretical calculation
                grid_2D = np.tile(grid_pts, (ngrid,1))

                #Theoretical value
                # utru = diffraq.utils.solution_util.direct_integration( \
                #     fresnums[i], pupil[0].shape, sim.occulter.xq, sim.occulter.yq, \
                #     sim.occulter.wq, grid_2D)

                #Assert max difference is close to specified tolerance
                max_diff = tol * fresnums[i]
                # assert(np.abs(utru - pupil[0]).max() < max_diff)

                import matplotlib.pyplot as plt;plt.ion()

                # plt.figure()
                # plt.plot(abs(utru[len(utru)//2]))
                # plt.plot(abs(pupil[0][len(pupil[0])//2]),'--')
                # plt.figure()
                # plt.imshow(abs(utru))
                plt.figure()
                plt.imshow(abs(pupil[0]))
                breakpoint()

        #Cleanup
        sim.clean_up()
        del grid_pts, grid_2D, pupil, utru

if __name__ == '__main__':

    tst = Test_calc_pupil_field()
    tst.test_pupil_field()
